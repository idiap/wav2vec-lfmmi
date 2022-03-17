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

from fairseq.data import FairseqDataset, data_utils

import espresso.tools.utils as speech_utils

logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=False,
    left_pad_target=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        if key == "source":
            return speech_utils.collate_frames(
                [s[key] for s in samples],
                0.0,
                left_pad,
                pad_to_length=pad_to_length,
                pad_to_multiple=pad_to_multiple,
            )
        elif key == "target":
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad,
                move_eos_to_beginning,
                pad_to_length=pad_to_length,
                pad_to_multiple=pad_to_multiple,
            )
        else:
            raise ValueError("Invalid key.")

    id = torch.LongTensor([s["id"] for s in samples])
    src_frames = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.IntTensor([s["source"].size(0) for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    utt_id = [samples[i]["utt_id"] for i in sort_order.numpy()]
    src_frames = src_frames.index_select(0, sort_order)

    target = None
    if samples[0].get("target_unprocessed", None) is not None:
        target_unprocessed = [
            samples[i]["target_unprocessed"] for i in sort_order.numpy()
        ]
    else:
        target_unprocessed = None

    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        target_lengths = torch.tensor(
            [
                samples[i]["target"].ne(pad_idx).int().sum().item()
                for i in sort_order.numpy()
            ]
        )
        ntokens = sum(target_lengths).item()
    else:
        ntokens = src_lengths.sum().item()
        target_lengths = None

    target_raw_text = None
    if samples[0].get("target_raw_text", None) is not None:
        target_raw_text = [samples[i]["target_raw_text"] for i in sort_order.numpy()]

    batch = {
        "id": id,
        "utt_id": utt_id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_frames,
            "src_lengths": src_lengths,
        },
        "target": target,
        "target_unprocessed": target_unprocessed,
        "target_lengths": target_lengths,
        "target_raw_text": target_raw_text,
    }

    return batch


class AsrCtcDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        labels: labels for the CTC
        pad,
        eos,
        batch_targets,
        process_label=None,
        add_to_input=False,
    """

    def __init__(
        self,
        src,
        src_sizes,
        labels,
        pad,
        eos,
        transcripts=None,
        shuffle=True,
        process_label=None,
    ):
        self.src = src
        self.labels = labels
        self.src_sizes = np.array(src_sizes)
        self.shuffle = shuffle
        self.labels = labels
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.transcripts = transcripts

    def get_label(self, index):
        if self.process_label is None:
            return self.labels[index], self.labels[index]
        else:
            return self.labels[index], self.process_label(self.labels[index])

    def get_text(self, index):
        return None if self.transcripts is None else self.transcripts[index]

    def __getitem__(self, index):
        src_item = self.src[index]
        tgt_item, tgt_item_processed = self.get_label(index)
        text_item = self.get_text(index)
        item = {
            "id": index,
            "utt_id": self.src.utt_ids[index],
            "source": src_item,
            "target": tgt_item_processed,
            "target_unprocessed": tgt_item,
            "target_raw_text": text_item,
        }
        return item

    def size(self, index):
        sz = self.src_sizes[index]
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {"source": source_pad_to_length, "target": target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `utt_id` (List[str]): list of utterance ids
                - `nsentences` (int): batch size
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (FloatTensor): a padded 3D Tensor of features in
                    the source of shape `(bsz, src_len, feat_dim)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (IntTensor): 1D Tensor of the unpadded
                    lengths of each source sequence of shape `(bsz)`
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `target_raw_text` (List[str]): list of original text
        """
        res = collate(
            samples,
            pad_idx=self.pad,
            eos_idx=self.eos,
            left_pad_source=False,
            left_pad_target=False,
            pad_to_length=pad_to_length,
            pad_to_multiple=1,
        )
        return res

    def num_tokens(self, index):
        """Return the number of frames in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.src_sizes[index]

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        return sizes

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self)).astype(np.int64)]
        else:
            order = [np.arange(len(self), dtype=np.int64)]
        order.append(self.src.sizes)
        return np.lexsort(order)[::]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False)

    def prefetch(self, indices):
        """Only prefetch src."""
        self.src.prefetch(indices)

    @property
    def supports_fetch_outside_dataloader(self):
        """Whether this dataset supports fetching outside the workers of the dataloader."""
        return False

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False  # to avoid running out of CPU RAM

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.src, "set_epoch"):
            self.src.set_epoch(epoch)
