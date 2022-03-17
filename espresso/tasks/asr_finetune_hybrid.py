# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import itertools
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch

from fairseq import utils
from fairseq.data import BaseWrapperDataset, ConcatDataset
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.tasks import FairseqTask, register_task
from omegaconf import II

from espresso.data import (
    WavAudioDataset,
    AsrWavChainDataset,
    AsrDictionary,
)

try:
    import kaldi_io
except ImportError:
    raise ImportError("Please install kaldi_io with: pip install kaldi_io")


logger = logging.getLogger(__name__)


@dataclass
class AsrWavFinetuneHybridConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    dict: Optional[str] = field(
        default=None, metadata={"help": "path to the dictionary"}
    )
    non_lang_syms: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to a file listing non-linguistic symbols, e.g., <NOISE> "
            "etc. One entry per line. To be filtered out when calculating WER/CER"
        },
    )
    wer_output_filter: Optional[str] = field(
        default=None,
        metadata={"help": "path to wer_output_filter file for WER evaluation"},
    )
    # max_source_positions: Optional[int] = field(
    #     default=1024, metadata={"help": "max number of tokens in the source sequence"}
    # )
    # max_target_positions: Optional[int] = field(
    #     default=1024, metadata={"help": "max number of tokens in the target sequence"}
    # )
    upsample_primary: int = field(
        default=1,
        metadata={"help": "amount to upsample primary dataset"},
    )
    num_targets: int = field(
        default=3000,
        metadata={"help": "number of targets for training (e.g., num pdf-ids)"},
    )

    # Wav dataset related
    sample_rate: int = field(
        default=16000,
        metadata={"help": "Sampling rate for the wav files"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Pass maximum sample size in seconds to cropped input"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Pass minimum duration sample size in seconds"},
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    train_subset: str = II("dataset.train_subset")
    valid_subset: str = II("dataset.valid_subset")
    gen_subset: str = II("dataset.gen_subset")
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")
    criterion_name: str = II("criterion._name")
    max_epoch: int = II(
        "optimization.max_epoch"
    )  # to determine whether in trainig stage


def get_asr_dataset_from_json(
    data_path,
    split,
    dictionary,
    sample_rate,
    upsample_primary=1,
    max_sample_size=None,
    min_sample_size=0,
    shuffle=True,
    pad=True,
    normalize=False,
    seed=1,
):
    """
    Parse data json and create dataset.
    See espresso/tools/asr_prep_json.py which pack json from raw files
    Json example:
    {
        "011c0202": {
            "feat": "data/train_si284_spe2e_hires/data/raw_mfcc_train_si284_spe2e_hires.1.ark:24847",
            "numerator_fst": "exp/chain/e2e_bichar_tree_tied1a/fst.1.ark:6704",
            "alignment": "exp/tri3/ali.ark:8769",
            "text": "THE HOTELi OPERATOR'S EMBASSY",
            "utt2num_frames": "693",
        },
        "011c0203": {
            ...
        }
    }
    """
    src_datasets = []
    tgt_datasets = []
    text_datasets = []

    data_json_path = os.path.join(data_path, "{}.json".format(split))
    if not os.path.isfile(data_json_path):
        raise FileNotFoundError("Dataset not found: {}".format(data_json_path))

    with open(data_json_path, "rb") as f:
        loaded_json = json.load(f, object_pairs_hook=OrderedDict)

    utt_ids, wavs, numerator_fsts, alignments, text, utt2len = [], [], [], [], [], []
    for utt_id, val in loaded_json.items():
        utt_ids.append(utt_id)
        wavs.append(val["wav"])
        if "numerator_fst" in val:
            numerator_fsts.append(val["numerator_fst"])
        if "alignment" in val:
            alignments.append(val["alignment"])
        if "text" in val:
            text.append(val["text"])
        if "utt2len" in val:
            utt2len.append(int(val["utt2len"]))

    assert len(utt2len) == 0 or len(utt_ids) == len(utt2len)
    dataset = WavAudioDataset(
        utt_ids,
        wavs,
        utt2len,
        sample_rate=sample_rate,
        max_sample_size=max_sample_size,
        min_sample_size=min_sample_size,
        pad=pad,
        normalize=normalize,
    )

    return AsrWavChainDataset(
        dataset,
        batch_targets=True,
        utt_ids=utt_ids if len(utt_ids) else None,
        rxfiles=numerator_fsts if len(numerator_fsts) else None,
        text=text if len(text) else None,
    )


@register_task("asr_wav_finetune_hybrid", dataclass=AsrWavFinetuneHybridConfig)
class AsrWavFinetuneHybridTask(FairseqTask):
    """
    Hybrid speech recognition with lattice-free MMI or cross-entropy loss.
    Currently it dumps posteriors from neural networks' output on-the-fly or
    as an ark file for Kaldi to decode.

    Args:
        dictionary (~fairseq.data.AsrDictionary): dictionary for the final text

    .. note::

        The speech recognition with lattice-free MMI task is compatible with
        :mod:`speech-train`, and :mod:`dump-posteriors`. The results are not
        strictly reproducible (i.e., there is some randomness among different
        runs with the same exprimental setting) due to the use of `atomicAdd`
        while accumulating gradients w.r.t. pdf-ids in backprop of LF-MMI loss.
        See https://pytorch.org/docs/stable/notes/randomness.html for details.

    The speech recognition task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.speech_recognition_parser
        :prog:
    """

    @classmethod
    def load_dictionary(cls, filename, non_lang_syms=None):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
            non_lang_syms (str): non_lang_syms filename
        """
        return AsrDictionary.load(filename, f_non_lang_syms=non_lang_syms)

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Disable this method"""
        raise NotImplementedError

    def __init__(self, cfg: AsrWavFinetuneHybridConfig, dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary
        self.num_targets = cfg.num_targets
        # self.training_stage = (cfg.max_epoch > 0)  # a hack

        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.deterministic = False

    @classmethod
    def setup_task(cls, cfg: AsrWavFinetuneHybridConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AsrWavFinetuneHybridConfig): configuration of this task
        """
        # load dictionaries
        dict_path = cfg.dict
        dictionary = (
            cls.load_dictionary(dict_path, non_lang_syms=cfg.non_lang_syms)
            if dict_path is not None
            else None
        )
        if dictionary is not None:
            logger.info("dictionary: {} types".format(len(dictionary)))

        return cls(cfg, dictionary)

    def load_dataset(
        self,
        split: str,
        epoch: int = 1,
        task_cfg: FairseqDataclass = None,
        **kwargs,
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            epoch (int): epoch number determining which shard of training data to load
            task_cfg (FairseqDataclass): optional task configuration stored in the checkpoint that can be used
                                          to load datasets
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]
        task_cfg = task_cfg or self.cfg

        self.datasets[split] = get_asr_dataset_from_json(
            data_path,
            split,
            self.dictionary,
            sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
            upsample_primary=self.cfg.upsample_primary,
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            shuffle=(split != self.cfg.gen_subset),
            pad=True,
            normalize=task_cfg.normalize,
        )

    def build_generator(self, models, cfg: GenerationConfig):
        if cfg.score_reference:
            cfg.score_reference = False
            logger.warning(
                "--score-reference is not applicable to speech recognition, ignoring it."
            )
        from espresso.tools.generate_log_probs_for_w2v_decoding import (
            GenerateLogProbsForW2vDecoding,
        )

        apply_log_softmax = getattr(cfg, "apply_log_softmax", False)
        return GenerateLogProbsForW2vDecoding(
            models, apply_log_softmax=apply_log_softmax
        )

    # def build_dataset_for_inference(self, src_tokens, src_lengths):
    #     return AsrChainDataset(src_tokens, src_lengths)

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        state_post = []
        for log in logging_outputs:
            post = log.get("state_post", None)
            if post is not None:
                state_post.append(post)
        if len(state_post) > 0:
            # collect state priors from all workers and do weighted average
            weights = state_post[0].new(
                [log.get("ntokens", 0) for log in logging_outputs]
            )
            weights = weights / weights.sum()  # N
            with torch.no_grad():
                stacked_state_post = torch.stack(state_post, dim=1)  # V x N
                self.averaged_state_post = stacked_state_post.mv(weights)  # V
        else:
            self.averaged_state_post = None

    def update_state_prior(self, model):
        if self.averaged_state_post is not None:
            assert hasattr(model, "update_state_prior")
            model.update_state_prior(
                self.averaged_state_post, self.state_prior_update_smoothing
            )

    #  def max_positions(self):
    #      """Return the max sentence length allowed by the task."""
    #      return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.AsrDictionary`."""
        # Note: padding idx for criterions would be self.target_dictionary.pad() if it
        # returns not None.
        return None
