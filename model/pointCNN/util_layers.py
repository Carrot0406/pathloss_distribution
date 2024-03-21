import torch.nn as nn
from typing import Callable, Union, Tuple

from util_funcs import UFloatTensor


def EndChannels(f, make_contiguous=False):
    """ Class decorator to apply 2D convolution along end channels. """

    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x = self.f(x)
            x = x.permute(0, 2, 3, 1)
            return x

    return WrappedLayer()


class Dense(nn.Module):
    """
    Single layer perceptron with optional activation, batch normalization, and dropout.
    单层感知器，具有可选激活、批量规范化和丢弃功能
    """

    def __init__(self, in_features: int, out_features: int,
                 drop_rate: int = 0, with_bn: bool = True,
                 activation: Callable[[UFloatTensor], UFloatTensor] = nn.ReLU()
                 # 这段代码定义了一个变量 activation，它是一个类型为 Callable[[UFloatTensor], UFloatTensor] 的变量，并且将其初始化为 nn.ReLU()。
                 ) -> None:
        """
        :param in_features: Length of input featuers (last dimension).
        :param out_features: Length of output features (last dimension).
        :param drop_rate: Drop rate to be applied after activation.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Dense, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        # self.bn = LayerNorm(out_channels) if with_bn else None
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, x: UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Linear.
        :return: Tensor with linear layer and optional activation, batchnorm,
        and dropout applied.
        """
        x = x.cuda()
        self.linear = self.linear.cuda()
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        # if self.bn:
        #     x = self.bn(x)
        if self.drop:
            x = self.drop(x)
        return x


class Conv(nn.Module):
    """
    2D convolutional layer with optional activation and batch normalization.
    具有可选激活和批量规范化的2D卷积层
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]], with_bn: bool = True,
                 # 类型注释指定了 kernel_size 这个变量可以是整数类型或者是一个包含两个整数的元组类型
                 activation: Callable[[UFloatTensor], UFloatTensor] = nn.ReLU()
                 ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9) if with_bn else None

    def forward(self, x: UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with convolutional layer and optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x


class SepConv(nn.Module):
    """ Depthwise separable convolution with optional activation and batch normalization
    具有可选激活和批量归一化的深度可分离卷积
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 depth_multiplier: int = 1, with_bn: bool = True,
                 activation: Callable[[UFloatTensor], UFloatTensor] = nn.ReLU()
                 ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :depth_multiplier: Depth multiplier for middle part of separable convolution.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(SepConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups=in_channels),
            nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, bias=not with_bn)
        )

        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9) if with_bn else None

    def forward(self, x: UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with depthwise separable convolutional layer and
        optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x


class LayerNorm(nn.Module):
    """
    Batch Normalization over ONLY the mini-batch layer (suitable for nn.Linear layers).
    仅在小批量层上进行batch归一化（适用于nn.Linear层）
    """

    def __init__(self, N: int, dim: int, *args, **kwargs) -> None:
        """
        :param N: Batch size.
        :param D: Dimensions.
        """
        super(LayerNorm, self).__init__()
        if dim == 1:
            self.bn = nn.BatchNorm1d(N, *args, **kwargs)
        elif dim == 2:
            self.bn = nn.BatchNorm2d(N, *args, **kwargs)
        elif dim == 3:
            self.bn = nn.BatchNorm3d(N, *args, **kwargs)
        else:
            raise ValueError("Dimensionality %i not supported" % dim)

        self.forward = lambda x: self.bn(x.unsqueeze(0)).squeeze(0)
