#!/usr/bin/env python3
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>


import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from espresso.optim.nsg import OnlineNaturalGradient, NGState, constrain_orthonormal


class OnlineNaturalGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, in_state, out_state):
        ctx.save_for_backward(input, weight, bias)
        ctx.states = [in_state, out_state]
        # the code below is based on pytorch's F.linear
        if input.dim() == 2 and bias is not None:
            output = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias
        return output

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output):
        """Backward pass for NG-SGD layer

        We pass the gradients computed by Pytorch to Kaldi's precondition_directions
        given the states of the layer."""
        input, weight, _ = ctx.saved_tensors
        in_state, out_state = ctx.states
        if input.dim() == 3:
            mb, T, D = input.shape
            mb_T = mb * T
        else:
            mb_T, D = input.shape
        input_temp = torch.zeros(mb_T, D + 1, device=input.device, requires_grad=False)
        input_temp[:, -1] = 1.0
        input_temp[:, :-1].copy_(input.reshape(mb_T, D))
        grad_weight = grad_bias = None
        if grad_output.dim() == 3:
            grad_input = grad_output.matmul(weight)
            grad_input = grad_input.reshape(mb, T, D)
        else:
            grad_input = grad_output.mm(weight)
        in_scale = in_state.precondition_directions(input_temp)
        out_dim = grad_output.shape[-1]
        grad_output_temp = grad_output.reshape(-1, out_dim)
        out_scale = out_state.precondition_directions(
            grad_output_temp
        )  # hope grad_output is continguous!
        scale = in_scale * out_scale
        grad_output.data.mul_(scale)
        # TODO: check if we should use data member instead?
        grad_weight = grad_output_temp.t().mm(input_temp[:, :-1])
        grad_bias = grad_output_temp.t().mm(input_temp[:, -1].reshape(-1, 1))
        grad_weight.data.mul_(scale)
        grad_bias.data.mul_(scale)
        return grad_input, grad_weight, grad_bias.t(), None, None, None, None


class NaturalAffineTransform(nn.Module):
    def __init__(
        self,
        feat_dim,
        out_dim,
        bias=True,
        ngstate=None,
    ):
        """Initialize NaturalGradientAffineTransform layer

        The function initializes NG-SGD states and parameters of the layer

        Args:
            feat_dim: (int, required) input dimension of the transformation
            out_dim: (int, required) output dimension of the transformation
            bias: (bool, optional) set False to not use bias. True by default.
            ngstate: a dictionary containing the following keys
                alpha: a floating point value (default is 4.0)
                num_samples_history: a floating point value (default is 2000.)
                update_period: an integer (default is 4)

        Returns:
            NaturalAffineTransform object
        """
        super(NaturalAffineTransform, self).__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.preconditioner_in = OnlineNaturalGradient()
        self.preconditioner_out = OnlineNaturalGradient()
        if ngstate is None:
            ngstate = NGState()
        self.weight = nn.Parameter(torch.Tensor(out_dim, feat_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_dim))
        else:
            self.register_parameter("bias", None)
        self.init_parameters()

    def init_parameters(self):
        """Initialize the parameters (weight and bias) of the layer"""

        self.weight.data.normal_()
        self.weight.data.mul_(1.0 / pow(self.feat_dim * self.out_dim, 0.5))
        self.bias.data.normal_()

    def forward(self, input):
        """Forward pass"""
        return OnlineNaturalGradientFunction.apply(
            input,
            self.weight,
            self.bias,
            self.preconditioner_in,
            self.preconditioner_out,
        )


class OrthonormalLinear(nn.Module):
    def __init__(self, linear_transform, scale=0.0):
        super(OrthonormalLinear, self).__init__()
        self.scale = torch.tensor(scale, requires_grad=False)
        self.linear_transform = linear_transform

    def forward(self, input):
        """Forward pass"""
        # do it before forward pass
        if self.training:
            with torch.no_grad():
                constrain_orthonormal(self.linear_transform.weight, self.scale)
        x = self.linear_transform(input)
        return x


class TDNNF(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_dim,
        bottleneck_dim,
        context_len=1,
        subsampling_factor=1,
        orthonormal_constraint=0.0,
        floating_scale=True,
        bypass_scale=0.66,
        natural_affine=True,
    ):
        super(TDNNF, self).__init__()
        # lets keep it context_len for now
        if natural_affine:
            linear_transform = NaturalAffineTransform(
                feat_dim * context_len, bottleneck_dim, bias=True
            )

        else:
            linear_transform = Linear(feat_dim * context_len, bottleneck_dim, bias=True)
        # wrap linear with Orthonormal linear
        self.linearB = OrthonormalLinear(linear_transform, scale=orthonormal_constraint)
        self.linearA = nn.Linear(bottleneck_dim, output_dim)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)
        self.bottleneck_dim = torch.tensor(bottleneck_dim, requires_grad=False)
        self.feat_dim = torch.tensor(feat_dim, requires_grad=False)
        self.subsampling_factor = torch.tensor(subsampling_factor, requires_grad=False)
        self.context_len = torch.tensor(context_len, requires_grad=False)
        self.orthonormal_constraint = torch.tensor(
            orthonormal_constraint, requires_grad=False
        )
        self.bypass_scale = torch.tensor(bypass_scale, requires_grad=False)
        if bypass_scale > 0.0 and feat_dim == output_dim:
            self.use_bypass = True
            if self.context_len > 1:
                if self.context_len % 2 == 1:
                    self.identity_lidx = self.context_len // 2
                    self.identity_ridx = -self.identity_lidx
                else:
                    self.identity_lidx = self.context_len // 2
                    self.identity_ridx = -self.identity_lidx + 1
            else:
                self.use_bypass = False
        else:
            self.use_bypass = False

    def forward(self, input):
        mb, T, D = input.shape
        padded_input = (
            input.reshape(mb, -1)
            .unfold(1, D * self.context_len, D * self.subsampling_factor)
            .contiguous()
        )
        x = self.linearB(padded_input)
        x = self.linearA(x)
        if self.use_bypass:
            x = (
                x
                + input[
                    :,
                    self.identity_lidx : self.identity_ridx : self.subsampling_factor,
                    :,
                ]
                * self.bypass_scale
            )
        return x


class TDNNFBatchNorm(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_dim,
        bottleneck_dim,
        context_len=1,
        subsampling_factor=1,
        orthonormal_constraint=0.0,
        bypass_scale=0.66,
        p_dropout=0.1,
        natural_affine=True,
    ):
        super(TDNNFBatchNorm, self).__init__()
        self.tdnn = TDNNF(
            feat_dim,
            output_dim,
            bottleneck_dim,
            context_len=context_len,
            subsampling_factor=subsampling_factor,
            orthonormal_constraint=orthonormal_constraint,
            bypass_scale=bypass_scale,
            natural_affine=natural_affine,
        )
        self.bn = nn.BatchNorm1d(output_dim, affine=False)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, input):
        mb, T, D = input.shape
        x = self.tdnn(input)
        x = x.permute(0, 2, 1).contiguous()
        x = self.bn(x)
        x = x.permute(0, 2, 1).contiguous()
        x = F.relu(x)
        x = self.drop(x)
        return x


# Create a network like the above one
class TdnnfModel(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_dim,
        padding=27,
        ivector_dim=0,
        hidden_dim=1024,
        bottleneck_dim=128,
        prefinal_bottleneck_dim=256,
        kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
        subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        frame_subsampling_factor=3,
        p_dropout=0.1,
        natural_affine=True,
        zero_init=True,
    ):
        super().__init__()

        # at present, we support only frame_subsampling_factor to be 3 or 1
        assert frame_subsampling_factor == 3 or frame_subsampling_factor == 1

        assert len(kernel_size_list) == len(subsampling_factor_list)
        num_layers = len(kernel_size_list)
        input_dim = feat_dim + ivector_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_subsampling = frame_subsampling_factor

        self.padding = padding
        self.frame_subsampling_factor = frame_subsampling_factor

        self.tdnn = TDNNFBatchNorm(
            input_dim,
            hidden_dim,
            bottleneck_dim=bottleneck_dim,
            context_len=kernel_size_list[0],
            subsampling_factor=subsampling_factor_list[0],
            orthonormal_constraint=-1.0,
            p_dropout=p_dropout,
            natural_affine=natural_affine,
        )
        tdnnfs = []
        for i in range(1, num_layers):
            kernel_size = kernel_size_list[i]
            subsampling_factor = subsampling_factor_list[i]
            layer = TDNNFBatchNorm(
                hidden_dim,
                hidden_dim,
                bottleneck_dim=bottleneck_dim,
                context_len=kernel_size,
                subsampling_factor=subsampling_factor,
                orthonormal_constraint=-1.0,
                p_dropout=p_dropout,
                natural_affine=natural_affine,
            )
            tdnnfs.append(layer)

        # tdnnfs requires [N, C, T]
        self.tdnnfs = nn.ModuleList(tdnnfs)

        # prefinal_l affine requires [N, C, T]
        self.prefinal_chain = TDNNFBatchNorm(
            hidden_dim,
            hidden_dim,
            bottleneck_dim=prefinal_bottleneck_dim,
            context_len=1,
            orthonormal_constraint=-1.0,
            p_dropout=0.0,
            natural_affine=natural_affine,
        )
        if natural_affine:
            self.chain_output = NaturalAffineTransform(hidden_dim, output_dim)
            if zero_init:
                self.chain_output.weight.data.zero_()
                self.chain_output.bias.data.zero_()
        else:
            self.chain_output = Linear(hidden_dim, output_dim)

        self.validate_model()

    def validate_model(self):
        N = 1
        T = 10 * self.frame_subsampling_factor
        C = self.input_dim
        x = torch.arange(N * T * C).reshape(N, T, C).float()
        nnet_output = self.forward(x)
        assert nnet_output.shape[1] == 10

    def pad_input(self, x):
        N, T, F = x.shape
        if self.padding > 0:
            x = torch.cat(
                [
                    x[:, 0:1, :].repeat(1, self.padding, 1),
                    x,
                    x[:, -1:, :].repeat(1, self.padding, 1),
                ],
                axis=1,
            )
        return x

    def forward(self, x):
        # input x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
        assert x.ndim == 3
        x = self.pad_input(x)
        # at this point, x is [N, T, C]
        x = self.tdnn(x)
        for i in range(len(self.tdnnfs)):
            x = self.tdnnfs[i](x)
        chain_prefinal_out = self.prefinal_chain(x)
        chain_out = self.chain_output(chain_prefinal_out)
        return chain_out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    feat_dim = 40
    output_dim = 3456
    model = TdnnfModel(feat_dim=feat_dim, output_dim=output_dim, padding=27)
    N, T = 1, 150
    C = feat_dim
    x = torch.arange(N * T * C).reshape(N, T, C).float()
    nnet_output = model(x)
    print(x.shape, nnet_output.shape)
