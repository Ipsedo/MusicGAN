import torch as th
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super(LayerNorm2d, self).__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        var = x.var(dim=[1, 2, 3], keepdim=True)

        return (x - mean) / th.sqrt(var + self.__epsilon)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.__epsilon})"

    def __str__(self) -> str:
        return self.__repr__()


class PixelNorm(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super(PixelNorm, self).__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x / th.sqrt(
            x.pow(2.0).mean(dim=1, keepdim=True) + self.__epsilon
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.__epsilon})"

    def __str__(self) -> str:
        return self.__repr__()


class AdaIN(nn.Module):
    def __init__(self, channels: int, style_channels: int):
        super(AdaIN, self).__init__()

        self.__to_style = nn.Linear(style_channels, 2 * channels)

        self.__inst_norm = nn.InstanceNorm2d(channels, affine=False)

        self.__channels = channels
        self.__style_channels = style_channels

    def forward(self, x: th.Tensor, z: th.Tensor) -> th.Tensor:
        b, c, _, _ = x.size()

        out_lin = self.__to_style(z).view(b, 2 * c, 1, 1)
        gamma, beta = out_lin.chunk(2, 1)

        out_norm = self.__inst_norm(x)

        out: th.Tensor = gamma * out_norm + beta

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}" + f"(channels={self.__channels}, "
            f"style={self.__style_channels})"
        )


class NoiseLayer(nn.Module):
    def __init__(self, channels: int):
        super(NoiseLayer, self).__init__()

        self.__channels = channels

        self.__to_noise = nn.Linear(1, channels, bias=False)

    def forward(self, x: th.Tensor) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        b, c, w, h = x.size()

        rand_per_pixel = th.randn(b, w, h, 1, device=device)

        out: th.Tensor = x + self.__to_noise(rand_per_pixel).permute(
            0, 3, 1, 2
        )

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__channels})"

    def __str__(self) -> str:
        return self.__repr__()


class MiniBatchStdDev(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super(MiniBatchStdDev, self).__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, _, w, h = x.size()

        std = th.sqrt(
            th.mean(
                (x - th.mean(x, dim=0, keepdim=True)) ** 2,
                dim=0,
                keepdim=True,
            )
            + self.__epsilon
        )

        std_mean = th.mean(std, dim=(1, 2, 3), keepdim=True).expand(b, 1, w, h)

        return th.cat([x, std_mean], dim=1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.__epsilon})"

    def __str__(self) -> str:
        return self.__repr__()
