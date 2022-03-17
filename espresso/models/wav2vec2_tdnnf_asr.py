# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import contextlib
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Any

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
    GradMultiply,
)

import espresso.tools.utils as speech_utils
from .speech_tdnnf import SpeechTdnnfEncoder


@dataclass
class Wav2Vec2AsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )

    # masking
    mask_updates: int = field(
        default=-1, metadata={"help": "apply masking for these many updates"}
    )
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded wav2vec args
    w2v_args: Any = None

    # Tdnnf args
    tdnnf_dropout: float = field(
        default=0.1, metadata={"help": "dropout rate for tdnnf"}
    )
    tdnnf_hidden_size: int = field(
        default=1024, metadata={"help": "hidden layer size for tdnnf"}
    )
    bottleneck_size: int = field(
        default=128, metadata={"help": "bottleneck dimension for tdnnf"}
    )
    prefinal_bottleneck_size: int = field(
        default=256, metadata={"help": "prefinal layer bottleneck dimension for tdnnf"}
    )
    kernel_sizes: str = field(
        default="[3, 3, 3, 1, 3, 3, 3]",
        metadata={"help": "kernel sizes to be used for tdnnf layers"},
    )
    subsampling_factors: str = field(
        default="[1, 1, 1, 3, 1, 1, 1]",
        metadata={"help": "subsampling factors for tdnnf layers"},
    )
    nsgd: bool = field(default=False, metadata={"help": "Use nsgd training for tdnnf"})
    num_layers: int = field(default=7, metadata={"help": "Number of layers for tdnnf"})
    output_subsampling: int = field(
        default=1, metadata={"help": "subsample the output by this rate"}
    )
    tdnnf_grad_mult: float = field(
        default=1.0, metadata={"help": "scaled tdnnf gradient by this"}
    )


@dataclass
class Wav2Vec2TdnnfLfmmiConfig(Wav2Vec2AsrConfig):
    pass


@register_model("wav2vec_tdnnf_lfmmi", dataclass=Wav2Vec2TdnnfLfmmiConfig)
class Wav2VecTdnnfLfmmi(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2TdnnfLfmmiConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2TdnnfLfmmiConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoder(cfg, task.num_targets)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"][0]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"][0]
        padding = net_output["padding_mask"][0]
        # TODO: Mask logits
        # if padding is not None and padding.any():
        #     padding = padding.T
        #     logits[padding][...,0] = 0
        #     logits[padding][...,1:] = float('-inf')

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, n_targets=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0
        self.tdnnf_grad_mult = cfg.tdnnf_grad_mult
        self.mask_updates = cfg.mask_updates

        if n_targets is not None:
            kernel_sizes = speech_utils.eval_str_nested_list_or_tuple(
                cfg.kernel_sizes, type=int
            )
            subsampling_factors = speech_utils.eval_str_nested_list_or_tuple(
                cfg.subsampling_factors, type=int
            )
            self.proj = SpeechTdnnfEncoder(
                input_size=d,
                output_size=n_targets,
                hidden_size=cfg.tdnnf_hidden_size,
                bottleneck_size=cfg.bottleneck_size,
                prefinal_bottleneck_size=cfg.prefinal_bottleneck_size,
                kernel_sizes=kernel_sizes,
                subsampling_factors=subsampling_factors,
                nsgd=cfg.nsgd,
                output_subsampling=cfg.output_subsampling,
                num_layers=cfg.num_layers,
                dropout=cfg.tdnnf_dropout,
                dropout_in=cfg.tdnnf_dropout,
                training_stage=getattr(task, "training_stage", True),
            )
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        mask_this_iter = self.apply_mask and self.training
        if self.mask_updates > -1:
            mask_this_iter = mask_this_iter and (self.mask_updates > self.num_updates)

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": mask_this_iter,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)
            x, padding_mask = res['x'], res['padding_mask']

        x = self.final_dropout(x)

        x = GradMultiply.apply(x, 1.0 / self.tdnnf_grad_mult)

        src_lengths = (~padding_mask).sum(-1) if padding_mask is not None else x.new_full((x.shape[0], ), x.shape[1], dtype=torch.long)

        if self.proj:
            x, x_lengths, x_padding_mask = self.proj.extract_features(
                x, src_lengths
            )

        # Multiply to get a higher learning rate for TDNNF module
        # TODO: Use composite optimizer as pointed out here
        # https://github.com/pytorch/fairseq/issues/3225
        x = GradMultiply.apply(x, self.tdnnf_grad_mult)

        if not tbc:
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [x_padding_mask.transpose(0, 1)]
            if x_padding_mask.any()
            else [],  # T x B
            "padding_mask": [
                x_padding_mask
            ],  # TODO: verify should it be x_padding_mask look at criterion
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [x_lengths],  # B
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if len(encoder_out["encoder_out"]) != 0:
            encoder_out["encoder_out"] = [
                encoder_out["encoder_out"][0].index_select(1, new_order)
            ]
        if len(encoder_out["encoder_padding_mask"]) != 0:
            encoder_out["encoder_padding_mask"] = [
                encoder_out["encoder_padding_mask"][0].index_select(1, new_order)
            ]  # note: transposed

        if len(encoder_out["src_lengths"]) != 0:
            encoder_out["src_lengths"] = [
                (encoder_out["src_lengths"][0]).index_select(0, new_order)
            ]
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
