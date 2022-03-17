# Copyright (c) Yiming Wang, Yiwen Shao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from typing import List, Optional

import numpy as np

import torch

from fairseq.data import FairseqDataset
from fairseq.data import BaseWrapperDataset, data_utils

import espresso.tools.utils as speech_utils

try:
    from pychain import ChainGraph
    import simplefst
except ImportError:
    raise ImportError(
        "Please install OpenFST and PyChain by `make openfst pychain` after entering espresso/tools"
    )

logger = logging.getLogger(__name__)


class AsrWavChainMemoryDataset(BaseWrapperDataset):
    """
    A dataset of numerator graphs for LF-MMI. It loads all graphs into memory at
    once as its relatively small.
    """

    def __init__(
        self,
        dataset,
        batch_targets,
        utt_ids: Optional[List[str]] = None,
        rxfiles: Optional[List[str]] = None,
        text: Optional[List[str]] = None,
    ):
        super().__init__(dataset)
        self.batch_targets = batch_targets
        self.read_fsts(utt_ids, rxfiles, text)

    def read_fsts(
        self,
        utt_ids: Optional[List[str]] = None,
        rxfiles: Optional[List[str]] = None,
        text: Optional[List[str]] = None,
    ):
        self.dummy_targets = True
        if utt_ids is not None:
            self.dummy_targets = False
            self.utt_ids = []
            self.rxfiles = []
            self.numerator_graphs = []
            self.fst_sizes = []
            self.transcripts = []
            for i, rxfile in enumerate(rxfiles):
                # Only load if present in line_inds
                if i in self.dataset.line_inds:
                    file_path, offset = self._parse_rxfile(rxfile)
                    fst = simplefst.StdVectorFst.read_ark(file_path, offset)
                    graph = ChainGraph(
                        fst, initial_mode="fst", final_mode="fst", log_domain=True
                    )
                    if not graph.is_empty:  # skip empty graphs
                        self.utt_ids.append(utt_ids[i])
                        self.rxfiles.append(rxfile)
                        self.fst_sizes.append(fst.num_states())
                        self.numerator_graphs.append(graph)
                        self.transcripts.append(text[i] if text else None)

            assert len(self.numerator_graphs) == len(self.dataset)

    def _parse_rxfile(self, rxfile):
        # separate offset from filename
        m = re.match(r"(\S+):([0-9]+)", rxfile)
        assert m is not None, "Illegal rxfile: {}".format(rxfile)
        return m.group(1), int(m.group(2))

    def _get_numerator(self, index):
        fst = None if self.dummy_targets else self.numerator_graphs[index]
        return fst

    def _get_utt_id(self, index):
        utt_id = None if self.dummy_targets else self.utt_ids[index]
        return utt_id

    def _get_transcript(self, index):
        transcript = None if self.dummy_targets else self.transcripts[index]
        return transcript

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self._get_numerator(index)
        item["transcript"] = self._get_transcript(index)
        if not self.dummy_targets:
            assert item["utt_id"] == self._get_utt_id(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = -1 if self.dummy_targets else self._get_numerator(index).num_states
        return (sz, own_sz)

    def collater(self, samples):
        try:
            from pychain import ChainGraphBatch
        except ImportError:
            raise ImportError(
                "Please install OpenFST and PyChain by `make openfst pychain` after entering espresso/tools"
            )

        def merge_target():
            max_num_transitions = max(s["label"].num_transitions for s in samples)
            max_num_states = max(s["label"].num_states for s in samples)
            return ChainGraphBatch(
                [s["label"] for s in samples],
                max_num_transitions=max_num_transitions,
                max_num_states=max_num_states,
            )

        # Sort in descending order to work with rnn packed sequence
        samples.sort(key=lambda x: -len(x["source"]))

        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())

        # TODO: Shall we change ntokens to total source tokens??
        if not self.dummy_targets:
            if self.batch_targets:
                collated["target_lengths"] = torch.LongTensor(
                    [s["label"].num_states for s in samples]
                )
                collated["target_ntokens"] = collated["target_lengths"].sum().item()
            else:
                collated["target_ntokens"] = sum(
                    [s["label"].num_states for s in samples]
                )

            target = merge_target()
        else:
            target = None
            collated["target_lengths"] = None
            collated["target_ntokens"] = None

        collated["transcripts"] = [s["transcript"] for s in samples]
        collated["target"] = target
        collated["nsentences"] = len(samples)

        return collated


class AsrWavChainDataset(BaseWrapperDataset):
    """
    A dataset of numerator graphs for LF-MMI. It loads all graphs into memory at
    once as its relatively small.
    """

    def __init__(
        self,
        dataset,
        batch_targets,
        utt_ids: Optional[List[str]] = None,
        rxfiles: Optional[List[str]] = None,
        text: Optional[List[str]] = None,
    ):
        super().__init__(dataset)
        self.batch_targets = batch_targets
        self.dummy_fst = self.dummy_text = self.dummy_uttid = True

        if rxfiles is not None:
            self.read_fsts(rxfiles)

        if utt_ids is not None:
            self.read_uttids(utt_ids)

        if text is not None:
            self.read_transcripts(text)

    def read_uttids(self, utt_ids):
        self.dummy_uttid = False
        self.utt_ids = []
        for i, uttid in enumerate(utt_ids):
            # Only load if present in line_inds
            if i in self.dataset.line_inds:
                self.utt_ids.append(utt_ids[i])
        assert len(self.utt_ids) == len(self.dataset)

    def read_transcripts(self, text):
        self.dummy_text = False
        self.transcripts = []
        for i, transcript in enumerate(text):
            # Only load if present in line_inds
            if i in self.dataset.line_inds:
                self.transcripts.append(text[i] if text else None)
        assert len(self.transcripts) == len(self.dataset)

    def read_fsts(
        self,
        rxfiles: Optional[List[str]] = None,
    ):
        try:
            from pychain import ChainGraph
            import simplefst
        except ImportError:
            raise ImportError(
                "Please install OpenFST and PyChain by `make openfst pychain` after entering espresso/tools"
            )

        self.dummy_fst = False
        self.rxfiles = []
        self.files = []
        self.offsets = []
        for i, rxfile in enumerate(rxfiles):
            # Only load if present in line_inds
            if i in self.dataset.line_inds:
                file_path, offset = self._parse_rxfile(rxfile)
                self.files.append(file_path)
                self.offsets.append(offset)
                self.rxfiles.append(rxfile)

        assert len(self.files) == len(self.dataset)

    def _parse_rxfile(self, rxfile):
        # separate offset from filename
        m = re.match(r"(\S+):([0-9]+)", rxfile)
        assert m is not None, "Illegal rxfile: {}".format(rxfile)
        return m.group(1), int(m.group(2))

    def _get_numerator(self, index):
        if self.dummy_fst:
            return None

        fst = simplefst.StdVectorFst.read_ark(self.files[index], self.offsets[index])
        graph = ChainGraph(fst, initial_mode="fst", final_mode="fst", log_domain=True)
        if graph.is_empty:
            raise Exception(
                "Utterantce : {}, Length: {}, has empty graph, states: {}".format(
                    self.utt_ids[index], self.sizes[index], fst.num_states()
                )
            )
        return graph

    def _get_utt_id(self, index):
        utt_id = None if self.dummy_uttid else self.utt_ids[index]
        return utt_id

    def _get_transcript(self, index):
        transcript = None if self.dummy_text else self.transcripts[index]
        return transcript

    def __getitem__(self, index):
        item = self.dataset[index]

        item["label"] = self._get_numerator(index)
        item["transcript"] = self._get_transcript(index)
        if not self.dummy_uttid:
            assert item["utt_id"] == self._get_utt_id(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = -1 if self.dummy_fst else self._get_numerator(index).num_states
        return (sz, own_sz)

    def collater(self, samples):
        try:
            from pychain import ChainGraphBatch
        except ImportError:
            raise ImportError(
                "Please install OpenFST and PyChain by `make openfst pychain` after entering espresso/tools"
            )

        def merge_target():
            max_num_transitions = max(s["label"].num_transitions for s in samples)
            max_num_states = max(s["label"].num_states for s in samples)
            return ChainGraphBatch(
                [s["label"] for s in samples],
                max_num_transitions=max_num_transitions,
                max_num_states=max_num_states,
            )

        # Sort in descending order to work with rnn packed sequence
        samples.sort(key=lambda x: -len(x["source"]))

        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())

        # TODO: Shall we change ntokens to total source tokens??
        if not self.dummy_fst:
            if self.batch_targets:
                collated["target_lengths"] = torch.LongTensor(
                    [s["label"].num_states for s in samples]
                )
                collated["target_ntokens"] = collated["target_lengths"].sum().item()
            else:
                collated["target_ntokens"] = sum(
                    [s["label"].num_states for s in samples]
                )

            target = merge_target()
        else:
            target = None
            collated["target_lengths"] = None
            collated["target_ntokens"] = None

        collated["transcripts"] = [s["transcript"] for s in samples]
        collated["target"] = target
        collated["nsentences"] = len(samples)
        return collated
