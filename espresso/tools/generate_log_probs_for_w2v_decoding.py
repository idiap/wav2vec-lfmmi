# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GenerateLogProbsForW2vDecoding(nn.Module):
    def __init__(self, models, apply_log_softmax=False):
        """Generate the neural network's output intepreted as log probabilities
        for decoding with Kaldi.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            apply_log_softmax (bool, optional): apply log-softmax on top of the
                network's output (default: False)
        """
        super().__init__()
        self.model = models[0]
        self.apply_log_softmax = apply_log_softmax

        self.model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
        """
        return self._generate(sample, **kwargs)

    def _generate(self, sample: Dict[str, Dict[str, Tensor]], **kwargs):

        encoder_outs = self.model(**sample["net_input"])

        if "ntokens" not in sample:
            sample["ntokens"] = (~encoder_outs['padding_mask'][0]).sum().item()
        
        if "utt_ids" in sample and "utt_id" not in sample:
            sample["utt_id"] = sample["utt_ids"] 

        logits = encoder_outs["encoder_out"][0].transpose(0, 1).float()  # T x B x V -> B x T x V
        padding_mask = (
            encoder_outs["encoder_padding_mask"][0].t()
            if len(encoder_outs["encoder_padding_mask"]) > 0
            else None
        )
        if self.apply_log_softmax:
            return F.log_softmax(logits, dim=-1), padding_mask
        return logits, padding_mask
