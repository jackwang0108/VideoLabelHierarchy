# Standard Library
import math

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

        Args:
            net (nn.Module): original module to warp
            clip_len (int): length of the input clip, i.e., size of Temporal dim
            n_div (int): number of channel group to shift, for example, channel=64, n_div=4, 16 channels will be shifted as a result
            inplace (bool, optional): if use inplace TSM. Defaults to True.
        """
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = clip_len
        self.fold_div = n_div
        self.inplace = inplace

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        input shape: [Batch * Temporal, Channel, Height, Width]
        output shape: [Batch * Temporal, Channel, Height, Width]
        """
        x = self.shift(x, self.n_segment, fold_div=self.fold_div,
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
    """
    GatedShift 就是对传入的net进行一个wrap, forward计算的时候先通过GSM模块, 然后再通过net. GSM模块就是Gate-Shift Networks for Video Action Recognition提出的一个3D卷积模块, 把相邻两帧的2D卷积信息加进来
    """

    class _GSM(nn.Module):
        """
        Gate-Shift Networks for Video Action Recognition论文中提出的3D卷积模块
        """

        def __init__(self, fPlane, num_segments=3):
            super(GatedShift._GSM, self).__init__()

            self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1,
                                    padding=(1, 1, 1), groups=2)
            nn.init.constant_(self.conv3D.weight, 0)
            nn.init.constant_(self.conv3D.bias, 0)
            self.tanh = nn.Tanh()
            self.fPlane = fPlane
            self.num_segments = num_segments
            self.bn = nn.BatchNorm3d(num_features=fPlane)
            self.relu = nn.ReLU()

        def lshift_zeroPad(self, x: torch.FloatTensor) -> torch.FloatTensor:
            n, t, c, h, w = x.size()
            temp = x.data.new(n, t, 1, h, w).fill_(0)
            # return torch.cat((x[:, :, 1:], ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0)), dim=2)
            return torch.cat((x[:, :, 1:], temp), dim=2)

        def rshift_zeroPad(self, x: torch.FloatTensor) -> torch.FloatTensor:
            n, t, c, h, w = x.size()
            temp = x.data.new(n, t, 1, h, w).fill_(0)
            # return torch.cat((ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0), x[:, :, :-1]), dim=2)
            return torch.cat((temp, x[:, :, :-1]), dim=2)

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            batchSize = x.size(0) // self.num_segments
            shape = x.size(1), x.size(2), x.size(3)
            assert shape[0] == self.fPlane
            x = x.view(batchSize, self.num_segments, *
                       shape).permute(0, 2, 1, 3, 4).contiguous()
            x_bn = self.bn(x)
            x_bn_relu = self.relu(x_bn)
            gate = self.tanh(self.conv3D(x_bn_relu))
            gate_group1 = gate[:, 0].unsqueeze(1)
            gate_group2 = gate[:, 1].unsqueeze(1)
            x_group1 = x[:, :self.fPlane // 2]
            x_group2 = x[:, self.fPlane // 2:]
            y_group1 = gate_group1 * x_group1
            y_group2 = gate_group2 * x_group2

            r_group1 = x_group1 - y_group1
            r_group2 = x_group2 - y_group2

            y_group1 = self.lshift_zeroPad(y_group1) + r_group1
            y_group2 = self.rshift_zeroPad(y_group2) + r_group2

            y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4,
                                     self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
            y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4,
                                     self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)

            y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                           y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)

            return y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize*self.num_segments, *shape)

    def __init__(
        self,
        net: timm.layers.ConvNormAct | nn.Module,
        n_segment: int,
        n_div: int
    ):
        """
        为给定的神经网络模块添加一个门控移位模块。

        参数:
            net (nn.Module): 神经网络模型。
            n_segment (int): 视频片段的数量。
            n_div (int): 分割数。

        属性:
            fold_dim (int): 基于输入通道数的折叠维度。
            gsm (_GSM): 门控移位模块。
            net (nn.Module): 神经网络模型。
            n_segment (int): 视频片段的数量。

        方法:
            forward(x): 执行门控移位模块的前向传播。

        示例:
            >>> gated_shift = GatedShift(net, n_segment=16, n_div=4)
            >>> output = gated_shift(input)
        """

        super(GatedShift, self).__init__()

        # 针对torchvision的ResNet中的模块进行处理
        if isinstance(net, torchvision.models.resnet.BasicBlock):
            channels = net.conv1.in_channels

        # 针对torchvision的ConvNormActivation模块进行处理
        elif isinstance(net, torchvision.ops.misc.ConvNormActivation):
            channels = net[0].in_channels

        # 针对timm的ConvBnAct模块进行处理
        elif isinstance(net, timm.layers.conv_bn_act.ConvBnAct):
            channels = net.conv.in_channels

        # 针对Pytorch的Con2d模块进行处理
        elif isinstance(net, nn.Conv2d):
            channels = net.in_channels
        else:
            raise NotImplementedError(type(net))

        self.fold_dim = math.ceil(channels // n_div / 4) * 4
        # Gate-Shift Networks for Video Action Recognition中提出的模块, 本质就是一个3D卷积
        self.gsm = GatedShift._GSM(self.fold_dim, n_segment)
        self.net = net
        self.n_segment = n_segment
        print(f'=> Using GSM, fold dim: {self.fold_dim} / {channels}')

    def forward(self, x):
        y = torch.zeros_like(x)
        y[:, :self.fold_dim, :, :] = self.gsm(x[:, :self.fold_dim, :, :])
        y[:, self.fold_dim:, :, :] = x[:, self.fold_dim:, :, :]
        return self.net(y)
