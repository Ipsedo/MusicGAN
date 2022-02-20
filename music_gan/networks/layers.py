import torch as th
import torch.nn as nn


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
        return f"PixelNorm(eps={self.__epsilon})"

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
        return "AdaIN" + \
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
        return f"NoiseLayer({self.__channels})"

    def __str__(self):
        return self.__repr__()
