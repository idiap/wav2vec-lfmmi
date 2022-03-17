# Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>
#
# Modified from espresso/speech_tdnn.py

from argparse import Namespace
import logging
from typing import Dict, List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import Linear
from fairseq.modules import FairseqDropout
from omegaconf import DictConfig

import espresso.tools.utils as speech_utils
from .tdnnf_nsgd import TdnnfModel


logger = logging.getLogger(__name__)


@register_model("speech_tdnnf")
class SpeechTdnnfEncoderModel(FairseqEncoderModel):
    def __init__(self, encoder, state_prior: Optional[torch.FloatTensor] = None):
        super().__init__(encoder)
        self.num_updates = 0
        self.state_prior = state_prior

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--dropout", type=float, metavar="D",
                            help="dropout probability")
        parser.add_argument("--hidden-size", type=int,
                            help="hidden size to be used for all Tdnnf layers")
        parser.add_argument("--bottleneck-size", type=int,
                            help="bottleneck dim to be used for all Tdnnf layers")
        parser.add_argument("--prefinal-bottleneck-size", type=int,
                            help="bottleneck dim to be used for prefinal layer")
        parser.add_argument("--kernel-sizes", type=str, metavar="EXPR",
                            help="list of all Tdnnf layer\'s kernel sizes")
        parser.add_argument("--subsampling-factors", type=str, metavar="EXPR",
                            help="list of subsampling factors Tdnnf layer\'s")
        parser.add_argument("--nsgd", type=bool,
                            help="use natural gradient descent")
        parser.add_argument("--default-init", type=bool,
                            help="Use default initializtion of Natural Affine layers")
        parser.add_argument("--output-subsampling", type=int,
                            help="output subsampling rate")
        parser.add_argument("--num-layers", type=int, metavar="N",
                            help="number of Tdnn layers")

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument("--dropout-in", type=float, metavar="D",
                            help="dropout probability for encoder\'s input")
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        kernel_sizes = speech_utils.eval_str_nested_list_or_tuple(
            args.kernel_sizes, type=int
        )
        subsampling_factors = speech_utils.eval_str_nested_list_or_tuple(
            args.subsampling_factors, type=int
        )
        logger.info(
            "input feature dimension: {}, output dimension: {}".format(
                task.feat_dim, task.num_targets
            )
        )

        encoder = SpeechTdnnfEncoder(
            input_size=task.feat_dim,
            output_size=task.num_targets,
            hidden_size=args.hidden_size,
            bottleneck_size=args.bottleneck_size,
            prefinal_bottleneck_size=args.prefinal_bottleneck_size,
            kernel_sizes=kernel_sizes,
            subsampling_factors=subsampling_factors,
            nsgd=args.nsgd,
            default_init=args.default_init,
            output_subsampling=args.output_subsampling,
            num_layers=args.num_layers,
            dropout=args.dropout,
            dropout_in=args.dropout_in,
            training_stage=getattr(task, "training_stage", True),
        )
        return cls(encoder, state_prior=getattr(task, "initial_state_prior", None))

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

    def output_lengths(self, in_lengths):
        return self.encoder.output_lengths(in_lengths)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output["encoder_out"][0]
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)

    # def get_logits(self, net_output):
    #     logits = net_output["encoder_out"][0]
    #     padding = net_output["padding_mask"][0]
    #     if padding is not None and padding.any():
    #         padding = padding.T
    #         logits[padding][...,0] = 0
    #         logits[padding][...,1:] = float('-inf')
    #     # logits = encoder_out.transpose(0, 1).squeeze(2)  # T x B x 1 -> B x T
    #     return logits

    def update_state_prior(self, new_state_prior, factor=0.1):
        assert self.state_prior is not None
        self.state_prior = self.state_prior.to(new_state_prior)
        self.state_prior = (1.0 - factor) * self.state_prior + factor * new_state_prior
        self.state_prior = self.state_prior / self.state_prior.sum()  # re-normalize

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["state_prior"] = self.state_prior
        return state_dict

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        state_dict_subset = state_dict.copy()
        self.state_prior = state_dict.get("state_prior", None)
        if "state_prior" in state_dict:
            self.state_prior = state_dict["state_prior"]
            del state_dict_subset["state_prior"]
        super().load_state_dict(
            state_dict_subset, strict=strict, model_cfg=model_cfg, args=args
        )


class SpeechTdnnfEncoder(FairseqEncoder):
    """Tdnnf encoder."""

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        bottleneck_size=128,
        prefinal_bottleneck_size=256,
        kernel_sizes=[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
        subsampling_factors=[1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        nsgd=True,
        default_init=False,
        output_subsampling=3,
        num_layers=12,
        dropout=0.1,
        dropout_in=0.0,
        training_stage=True,
    ):
        super().__init__(None)  # no src dictionary
        self.num_layers = num_layers
        assert len(kernel_sizes) == num_layers
        assert len(subsampling_factors) == num_layers
        assert (output_subsampling == 3) or (output_subsampling == 1)
        self.output_subsampling = output_subsampling
        padding = self.get_padding(kernel_sizes, subsampling_factors)
        # self.dropout_in_module = FairseqDropout(
        #     dropout_in, module_name=self.__class__.__name__
        # )

        self.tdnnf = TdnnfModel(
            input_size,
            output_size,
            padding=padding,
            hidden_dim=hidden_size,
            bottleneck_dim=bottleneck_size,
            prefinal_bottleneck_dim=prefinal_bottleneck_size,
            kernel_size_list=kernel_sizes,
            subsampling_factor_list=subsampling_factors,
            frame_subsampling_factor=output_subsampling,
            p_dropout=dropout,
            natural_affine=nsgd,
            zero_init=not default_init,
        )

        receptive_field_radius = padding
        self.training_stage = training_stage

    def get_padding(self, kernel_sizes, subsampling_factors):
        pad = 0
        global_subsampling = 1
        for k, s in zip(kernel_sizes, subsampling_factors):
            pad += (k - 1) * global_subsampling
            global_subsampling *= s
        self.padding = pad // 2
        assert global_subsampling == self.output_subsampling
        return self.padding

    def output_lengths(self, in_lengths):
        out_lengths = (
            in_lengths + self.output_subsampling - 1
        ) // self.output_subsampling
        return out_lengths

    def forward(self, src_tokens, src_lengths: Tensor, **unused):
        x, x_lengths, padding_mask = self.extract_features(src_tokens, src_lengths)
        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `foward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask.transpose(0, 1)]
            if padding_mask.any()
            else [],  # T x B
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [x_lengths],  # B
            "output_subsampling": self.output_subsampling,
        }

    def extract_features(self, src_tokens, src_lengths, **unused):
        x, x_lengths = src_tokens, src_lengths
        # x = self.dropout_in_module(x)

        x = self.tdnnf(x)
        x_lengths = self.output_lengths(x_lengths)

        padding_mask = ~speech_utils.sequence_mask(x_lengths, x.size(1))
        x = x.transpose(0, 1)  # B x T x C -> T x B x C
        return x, x_lengths, padding_mask

    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(
                    1, new_order
                )  # note: transposed
            ]
        if len(encoder_out["src_lengths"]) == 0:
            new_src_lengths = []
        else:
            new_src_lengths = [
                (encoder_out["src_lengths"][0]).index_select(0, new_order)
            ]

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": new_src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


@register_model_architecture("speech_tdnnf", "speech_tdnnf")
def base_architecture(args):

    args.dropout = getattr(args, "dropout", 0.1)
    args.hidden_size = getattr(args, "hidden_size", "1024")
    args.bottleneck_size = getattr(args, "bottleneck_size", "128")
    args.prefinal_bottleneck_size = getattr(args, "prefinal_bottleneck_size", "256")
    args.kernel_sizes = getattr(args, "kernel_sizes", "[3, 3, 3, 1, 3, 3, 3]")
    args.subsampling_factors = getattr(
        args, "subsampling_factors", "[1, 1, 1, 3, 1, 1, 1]"
    )
    args.nsgd = getattr(args, "nsgd", False)
    args.default_init = getattr(args, "default_init", False)
    args.num_layers = getattr(args, "num_layers", 7)
    args.output_subsampling = getattr(args, "output_subsampling", 3)
    args.dropout_in = getattr(args, "dropout_in", args.dropout)


@register_model_architecture("speech_tdnnf", "speech_tdnnf_wsj")
def tdnnf_wsj(args):
    base_architecture(args)
