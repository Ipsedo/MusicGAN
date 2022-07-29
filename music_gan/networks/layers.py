from typing import Tuple, Union, Optional, List
from math import sqrt

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t

from .constants import LEAKY_RELU_SLOPE


class Conv2dPadding(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            dilation: Tuple[int, int] = (1, 1),
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            (0, 0),
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype
        )

        assert len(padding) == 2
        assert padding[0] == padding[1]

        self.__repl = nn.ReplicationPad2d(
            (padding[0], padding[0], 0, padding[1])
        )

        self.__zero = nn.ZeroPad2d(
            (0, 0, padding[1], 0)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_padded = self.__repl(x)
        x_padded = self.__zero(x_padded)
        return super()._conv_forward(x_padded, self.weight, self.bias)


class LayerNorm2d(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super(LayerNorm2d, self).__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        var = x.var(dim=[1, 2, 3], keepdim=True)

        return (x - mean) / th.sqrt(var + self.__epsilon)

    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.__epsilon})"

    def __str__(self):
        return self.__repr__()


class PixelNorm(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super(PixelNorm, self).__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        norm = th.sqrt(
            x.pow(2.).mean(dim=1, keepdim=True) +
            self.__epsilon
        )

        return x / norm

    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.__epsilon})"

    def __str__(self):
        return self.__repr__()


class AdaIN(nn.Module):
    def __init__(
            self,
            channels: int,
            style_channels: int
    ):
        super(AdaIN, self).__init__()

        self.__to_style = nn.Linear(
            style_channels,
            2 * channels
        )

        self.__inst_norm = nn.InstanceNorm2d(
            channels, affine=False
        )

        self.__channels = channels
        self.__style_channels = style_channels

    def forward(self, x: th.Tensor, z: th.Tensor) -> th.Tensor:
        b, c, _, _ = x.size()

        out_lin = self.__to_style(z).view(b, 2 * c, 1, 1)
        gamma, beta = out_lin.chunk(2, 1)

        out_norm = self.__inst_norm(x)

        out = gamma * out_norm + beta

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}" + \
               f"(channels={self.__channels}, " \
               f"style={self.__style_channels})"


class NoiseLayer(nn.Module):
    def __init__(self, channels: int):
        super(NoiseLayer, self).__init__()

        self.__channels = channels

        self.__to_noise = nn.Linear(1, channels, bias=False)

    def forward(self, x: th.Tensor) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        b, c, w, h = x.size()

        rand_per_pixel = th.randn(b, w, h, 1, device=device)

        out = x + self.__to_noise(rand_per_pixel).permute(0, 3, 1, 2)

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__channels})"

    def __str__(self):
        return self.__repr__()


class MiniBatchStdDev(nn.Module):
    def __init__(
            self,
            epsilon: float = 1e-8
    ):
        super(MiniBatchStdDev, self).__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, _, h, w = x.size()

        std = th.sqrt(
            th.mean(
                (x - th.mean(x, dim=0, keepdim=True)) ** 2,
                dim=0, keepdim=True
            )
            + self.__epsilon
        )

        std_mean = (
            th.mean(std, dim=(1, 2, 3), keepdim=True)
            .expand(b, -1, h, w)
        )

        return th.cat([x, std_mean], dim=1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.__epsilon})"

    def __str__(self) -> str:
        return self.__repr__()


class ToMagnPhase(nn.Sequential):
    def __init__(self, in_channels: int):
        super(ToMagnPhase, self).__init__(
            EqualLrConv2d(
                in_channels, 2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            ),
            nn.Tanh()
        )


class FromMagnPhase(nn.Sequential):
    def __init__(self, out_channels: int):
        super(FromMagnPhase, self).__init__(
            EqualLrConv2d(
                2,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            ),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
        )


class EqualLR:
    def __init__(
            self,
            module: nn.Module,
            name: str,
            alpha: float,
            idx: int
    ) -> None:
        self.__name = name
        self.__alpha = alpha
        self.__idx = idx

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(self)

    def __compute_weight(self, module: nn.Module) -> th.Tensor:
        weight = getattr(module, self.__name + '_orig')
        fan_in = weight.data.size()[self.__idx] * weight.data.size()[2:].numel()
        return weight * sqrt(self.__alpha / fan_in)

    def __call__(self, module: nn.Module, x: th.Tensor) -> None:
        weight = self.__compute_weight(module)
        setattr(module, self.__name, weight)


class EqualLrConvTr2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int],
            padding: Tuple[int, int],
            output_padding: Tuple[int, int],
            alpha: float = 2.
    ) -> None:
        super().__init__()

        self.__conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

        nn.init.zeros_(self.__conv.bias.data)
        nn.init.normal_(self.__conv.weight.data)

        self.__equal_lr_w = EqualLR(self.__conv, "weight", alpha, 0)
        self.__equal_lr_b = EqualLR(self.__conv, "bias", alpha, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.__conv(x)


class EqualLrConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int],
            padding: Tuple[int, int],
            alpha: float = 2.
    ) -> None:
        super().__init__()

        self.__conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        nn.init.zeros_(self.__conv.bias.data)
        nn.init.normal_(self.__conv.weight.data)

        self.__equal_lr_w = EqualLR(self.__conv, "weight", alpha, 1)
        self.__equal_lr_b = EqualLR(self.__conv, "bias", alpha, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.__conv(x)


class EqualLrLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            alpha: float = 2.
    ) -> None:
        super().__init__()

        self.__lin = nn.Linear(in_features, out_features)

        nn.init.zeros_(self.__lin.bias.data)
        nn.init.normal_(self.__lin.weight.data)

        self.__equal_lr_w = EqualLR(self.__lin, "weight", alpha, 1)
        self.__equal_lr_b = EqualLR(self.__lin, "bias", alpha, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.__lin(x)
