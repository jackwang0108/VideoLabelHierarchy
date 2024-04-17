# Standard Library
import math
from typing import Callable

# Third-Party Library
import timm

# Torch Library
import torch
import torchvision
import torch.nn as nn
from torch.autograd.function import FunctionCtx


class TemporalShift(nn.Module):
    """ Temporal Shift Module (TSM) adapted from TSM: Temporal Shift Module for Efficient Video Understanding, arxiv: https://arxiv.org/abs/1811.08383 """

    class InplaceShift(torch.autograd.Function):

        @staticmethod
        def forward(ctx: FunctionCtx, input: torch.FloatTensor, num_channel):
            """
            input shape: [Batch, Temporal, Channel, Height, Width]
            output shape: [Batch, Temporal, Channel, Height, Width]
            num_channel: number of channels to shift
            """
            # not support higher order gradient
            # input = input.detach_()
            ctx.num_channel = num_channel
            n, t, c, h, w = input.size()
            # new creates a new tensor according to the shape we specified with same device and type, but values stored in different memories
            # the values are not copied when creating, so we need to fill the value manually
            buffer = input.data.new(n, t, num_channel, h, w).zero_()

            # shift backward
            # copy input along the temporal dim [1:] to buffer
            buffer[:, :-1] = input.data[:, 1:, :num_channel]
            # copy buffer to input to temporal dim [:], i.e., temporal shift by 1 frame
            input.data[:, :, :num_channel] = buffer

            # shift forward
            buffer.zero_()
            buffer[:, 1:] = input.data[:, :-1, num_channel: 2 * num_channel]
            input.data[:, :, num_channel: 2 * num_channel] = buffer
            return input

        @staticmethod
        def backward(ctx, grad_output: torch.FloatTensor) -> tuple[torch.FloatTensor, None]:
            # grad_output = grad_output.detach_()
            fold = ctx.num_channel
            n, t, c, h, w = grad_output.size()
            buffer = grad_output.data.new(n, t, fold, h, w).zero_()

            # get the gradient of backward shifted numbers
            buffer[:, 1:] = grad_output.data[:, :-1, :fold]
            grad_output.data[:, :, :fold] = buffer

            # get the gradient of forward shifted numbers
            buffer.zero_()
            buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
            grad_output.data[:, :, fold: 2 * fold] = buffer

            return grad_output, None

    def __init__(self, net: nn.Module, clip_len: int, n_div: int, inplace: bool = True):
        """
        warps the original module with TSM

        when calling:
            input shape: [Batch * Temporal, Channel(in), Height, Width]
            output shape: [Batch * Temporal, Channel(out), Height, Width]

        Args:
            net (nn.Module): original module to warp
            clip_len (int): length of the input clip, i.e., size of Temporal dim
            n_div (int): number of channel group to shift, for example, channel=64, n_div=4, 16 channels will be shifted as a result
            inplace (bool, optional): if use inplace TSM. Defaults to True.
        """
        super(TemporalShift, self).__init__()
        self.net = net
        self.clip_len = clip_len
        self.fold_div = n_div
        self.inplace = inplace

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        input shape: [Batch * Temporal, Channel(in), Height, Width]
        output shape: [Batch * Temporal, Channel(out), Height, Width]
        """
        x = self.shift(x, self.clip_len, fold_div=self.fold_div,
                       inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x: torch.FloatTensor, clip_len: int, fold_div: int = 3, inplace: bool = False) -> torch.FloatTensor:
        nt, c, h, w = x.size()
        n_batch = nt // clip_len
        x = x.view(n_batch, clip_len, c, h, w)

        num_channel = c // fold_div
        if inplace:
            # Inplace-TSM from original paper
            out = TemporalShift.InplaceShift.apply(x, num_channel)
        else:
            # Residual-TSM from original paper
            out = torch.zeros_like(x)
            out[:, :-1, :num_channel] = x[:, 1:, :num_channel]  # shift left
            out[:, 1:, num_channel: 2 * num_channel] = x[:, :-
                                                         1, num_channel: 2 * num_channel]  # shift right
            out[:, :, 2 * num_channel:] = x[:, :, 2 * num_channel:]  # not shift

        return out.view(nt, c, h, w)


class GatedShift(nn.Module):
    """ Gated Shift Module (GSM) adapted from Gate-Shift Networks for Video Action Recognition, arxiv: https://arxiv.org/abs/1912.00381 """

    class _GSM(nn.Module):

        def __init__(self, in_channel: int, clip_len: int):
            """
            wrap the original module with GSM

            Args:
                in_channel (int): input channels
                clip_len (int): length of the input clip, i.e., size of Temporal dim
            """
            super(GatedShift._GSM, self).__init__()

            self.conv3D: Callable[[torch.FloatTensor], torch.FloatTensor] = \
                nn.Conv3d(in_channel, 2, (3, 3, 3), stride=1,
                          padding=(1, 1, 1), groups=2)

            nn.init.constant_(self.conv3D.weight, 0)
            nn.init.constant_(self.conv3D.bias, 0)

            self.tanh = nn.Tanh()
            self.in_channel = in_channel
            self.clip_len = clip_len
            self.bn = nn.BatchNorm3d(num_features=in_channel)
            self.relu = nn.ReLU()

        def lshift_zeroPad(self, x: torch.FloatTensor) -> torch.FloatTensor:
            n, c, t, h, w = x.size()
            temp = x.data.new(n, c, 1, h, w).fill_(0)
            # return torch.cat((x[:, :, 1:], ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0)), dim=2)
            return torch.cat((x[:, :, 1:], temp), dim=2)

        def rshift_zeroPad(self, x: torch.FloatTensor) -> torch.FloatTensor:
            n, t, c, h, w = x.size()
            temp = x.data.new(n, t, 1, h, w).fill_(0)
            # return torch.cat((ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0), x[:, :, :-1]), dim=2)
            return torch.cat((temp, x[:, :, :-1]), dim=2)

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            """
            input shape: [Batch * Temporal, Channel, Height, Width]
            output shape: [Batch * Temporal, Channel, Height, Width]
            """
            nt, c, h, w = x.size()
            n_batch = nt // self.clip_len
            assert c == self.in_channel

            # permute to [Batch, Channel, Temporal, Height, Width]
            x = x.view(n_batch, self.clip_len, c, h, w).permute(
                0, 2, 1, 3, 4).contiguous()

            x_bn = self.bn(x)
            x_bn_relu = self.relu(x_bn)

            # shape: [Batch, 2, Temporal, Height, Width]
            gate = self.tanh(self.conv3D(x_bn_relu))
            gate_group1 = gate[:, 0].unsqueeze(1)
            gate_group2 = gate[:, 1].unsqueeze(1)

            # split the input into two groups
            y_group1: torch.FloatTensor
            y_group2: torch.FloatTensor

            x_group1 = x[:, :self.in_channel // 2]
            # shape: [Batch, Channel, Temporal, Height, Width]
            y_group1 = gate_group1 * x_group1
            r_group1 = x_group1 - y_group1
            y_group1 = self.lshift_zeroPad(y_group1) + r_group1
            y_group1 = y_group1.view(n_batch, 2, self.in_channel // 4,
                                     self.clip_len, h, w).permute(0, 2, 1, 3, 4, 5)

            x_group2 = x[:, self.in_channel // 2:]
            y_group2 = gate_group2 * x_group2
            r_group2 = x_group2 - y_group2
            y_group2 = self.rshift_zeroPad(y_group2) + r_group2
            y_group2 = y_group2.view(n_batch, 2, self.in_channel // 4,
                                     self.clip_len, h, w).permute(0, 2, 1, 3, 4, 5)

            y = torch.cat((y_group1.contiguous().view(n_batch, self.in_channel//2, self.clip_len, h, w),
                           y_group2.contiguous().view(n_batch, self.in_channel//2, self.clip_len, h, w)), dim=1)

            # permute back to [Batch * Temporal, Channel, Height, Width]
            return y.permute(0, 2, 1, 3, 4).contiguous().view(n_batch * self.clip_len, c, h, w)

    def __init__(
        self,
        net: timm.layers.ConvNormAct | nn.Module,
        channels: int,
        clip_len: int,
        n_div: int
    ):
        """
        wraps the original module with GSM

        when calling:
            input shape: [Batch * Temporal, Channel(in), Height, Width]
            output shape: [Batch * Temporal, Channel(out), Height, Width]

        Args:
            net(nn.Module): original module to warp
            clip_len (int): length of the input clip, i.e., size of Temporal dim
            n_div (int): number of channel group to shift, for example, channel=64, n_div=4, 16 channels will be shifted as a result

        Raises:
            NotImplementedError: if the module to warp is not supported
        """

        super(GatedShift, self).__init__()

        self.net = net
        self.in_channel = math.ceil(channels // n_div / 4) * 4
        # Gated Shift Module
        self.gsm = GatedShift._GSM(self.in_channel, clip_len)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        input shape: [Batch * Temporal, Channel(in), Height, Width]
        output shape: [Batch * Temporal, Channel(out), Height, Width]
        """
        y = torch.zeros_like(x)
        y[:, :self.in_channel, :, :] = self.gsm(x[:, :self.in_channel, :, :])
        y[:, self.in_channel:, :, :] = x[:, self.in_channel:, :, :]
        return self.net(y)


def get_shift_module_builder(clip_len: int, n_div: int, is_gsm: bool = False, inplace_tsm: bool = False) -> Callable[[nn.Module], TemporalShift | GatedShift]:
    """
    get the builder of temporal shift module

    Args:
        clip_len (int): length of the input clip, i.e., size of Temporal dim
        n_div (int): number of channel group to shift, for example, channel=64, n_div=4, 16 channels will be shifted as a result
        is_gsm (bool, optional): build GSM or TSM. Defaults to False.
        inplace_tsm (bool, optional): if use inplace TSM. Defaults to False.

    Returns:
        Callable[[nn.Module], TemporalShift | GatedShift]: shift module builder
    """
    def builder(net: nn.Module) -> TemporalShift | GatedShift:
        # sourcery skip: lift-return-into-if, remove-unnecessary-else, swap-if-else-branches
        if is_gsm:
            # get in_channels, GSM split the channels of input into two groups
            # so the in_channels need to be known
            if isinstance(net, torchvision.models.resnet.BasicBlock):
                channels = net.conv1.in_channels
            elif isinstance(net, torchvision.ops.misc.ConvNormActivation):
                channels = net[0].in_channels
            elif isinstance(net, timm.layers.conv_bn_act.ConvBnAct):
                channels = net.conv.in_channels
            elif isinstance(net, nn.Conv2d):
                channels = net.in_channels
            else:
                raise NotImplementedError(
                    f"Cannot decide input channels of {type(net)=}")

            shift_module = GatedShift(net, channels, clip_len, n_div)
        else:
            shift_module = TemporalShift(net, clip_len, n_div, inplace_tsm)

        return shift_module

    return builder


def insert_temporal_shift(
        backbone: torchvision.models.ResNet | timm.models.RegNet | timm.models.ConvNeXt,
        shift_module_builder: Callable[[nn.Module], TemporalShift | GatedShift]
) -> torchvision.models.ResNet | timm.models.RegNet | timm.models.ConvNeXt:
    """ insert temporal shift module inside the backbone """

    if isinstance(backbone, torchvision.models.ResNet):
        # insert TSM/GSM every 1 block if using resnet18/32/50 else 2 for resnet101/152
        n_round = 2 if len(list(backbone.layer3.children())) >= 23 else 1

        def insert_before_blocks(stage: nn.Sequential) -> nn.Sequential:
            blocks: list[torchvision.models.resnet.BasicBlock |
                         torchvision.models.resnet.Bottleneck]
            blocks = list(stage.children())

            for i, block in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = shift_module_builder(block.conv1)

            return nn.Sequential(*blocks)

        backbone.layer1 = insert_before_blocks(backbone.layer1)
        backbone.layer2 = insert_before_blocks(backbone.layer2)
        backbone.layer3 = insert_before_blocks(backbone.layer3)
        backbone.layer4 = insert_before_blocks(backbone.layer4)

    elif isinstance(backbone, timm.models.RegNet):
        n_round = 1

        def insert_before_blocks(stage: timm.models.regnet.RegStage) -> None:
            blocks: list[timm.models.regnet.Bottleneck]
            blocks = list(stage.children())

            for i, block in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = shift_module_builder(block.conv1)

        insert_before_blocks(backbone.s1)
        insert_before_blocks(backbone.s2)
        insert_before_blocks(backbone.s3)
        insert_before_blocks(backbone.s4)

    elif isinstance(backbone, timm.models.ConvNeXt):
        n_round = 1

        def insert_before_blocks(stage: timm.models.convnext.ConvNeXtBlock) -> nn.Sequential:
            blocks: list[timm.models.convnext.ConvNeXtBlock]
            blocks = list(stage.blocks)

            for i, block in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv_dw = shift_module_builder(block.conv_dw)

            return nn.Sequential(*blocks)

        insert_before_blocks(backbone.stages[0])
        insert_before_blocks(backbone.stages[1])
        insert_before_blocks(backbone.stages[2])
        insert_before_blocks(backbone.stages[3])

    else:
        raise NotImplementedError(
            f"insert_temporal_shift didn't support {type(backbone)=} yet")

    return backbone


if __name__ == "__main__":
    from .backbone import get_backbone

    for backbone in ["convnext_tiny", "convnext_large"]:
        convnext: timm.models.ConvNeXt = get_backbone(
            backbone=backbone, modality="rgb")

        print(f"{backbone=}, {type(convnext.stages[0])}, {
              len(list(convnext.stages[0].children()))=}")

        blocks = list(convnext.stages[0].blocks)
        for block in blocks:
            print(f"{type(block)=}, {type(block.conv_dw)=}")
