import torch as th
import torch.nn as nn

from typing import Iterator


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

    def __repr__(self) -> str:
        return "AdaIN" + \
               f"(channels={self.__channels}, " \
               f"style={self.__style_channels})"

    def __str__(self) -> str:
        return self.__repr__()


class _Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_channels: int
    ):
        super(_Block, self).__init__()

        self.__conv_1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1)
        )

        self.__noise_1 = NoiseLayer(
            out_channels
        )

        self.__adain_1 = AdaIN(
            out_channels,
            style_channels
        )

        self.__lr_relu_1 = nn.LeakyReLU(2e-1)

        self.__conv_2 = nn.ConvTranspose2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.__noise_2 = NoiseLayer(
            out_channels
        )

        self.__adain_2 = AdaIN(
            out_channels,
            style_channels
        )

        self.__lr_relu_2 = nn.LeakyReLU(2e-1)

    def forward(self, x: th.Tensor, style: th.Tensor) -> th.Tensor:
        out = self.__conv_1(x)
        out = self.__noise_1(out)
        out = self.__adain_1(out, style)
        out = self.__lr_relu_1(out)

        out = self.__conv_2(out)
        out = self.__noise_2(out)
        out = self.__adain_2(out, style)
        out = self.__lr_relu_2(out)

        return out


class Block(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(Block, self).__init__(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1)
            ),
            PixelNorm(),
            nn.LeakyReLU(2e-1),
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            PixelNorm(),
            nn.LeakyReLU(2e-1)
        )


class ToMagnPhaseLayer(nn.Sequential):
    def __init__(self, in_channels: int):
        super(ToMagnPhaseLayer, self).__init__(
            nn.ConvTranspose2d(
                in_channels, 2,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
            nn.Tanh()
        )


class LinearBlock(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(LinearBlock, self).__init__(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(2e-1)
        )


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            #rand_style_channels: int,
            #style_channels: int,
            end_layer: int = 0
    ):
        super(Generator, self).__init__()

        self.__curr_layer = end_layer

        self.__nb_downsample = 7

        channels = [
            (rand_channels, 64),
            (64, 56),
            (56, 48),
            (48, 40),
            (40, 32),
            (32, 24),
            (24, 16),
            (16, 8)
        ]

        self.__channels = channels

        assert 0 <= end_layer < len(channels)

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            Block(
                c[0], c[1]
            )
            for i, c in enumerate(channels)
        ])

        # for progressive gan
        self.__end_block = ToMagnPhaseLayer(
            channels[self.curr_layer][1]
        )

        self.__last_end_block = (
            None if self.__curr_layer == 0
            else nn.Sequential(
                nn.Upsample(
                    scale_factor=2.,
                    mode="bilinear",
                    align_corners=True
                ),
                ToMagnPhaseLayer(
                    channels[self.curr_layer - 1][1]
                )
            )
        )

        """style_layers = [
            (rand_style_channels, style_channels),
            (style_channels, style_channels),
            (style_channels, style_channels),
            (style_channels, style_channels),
            (style_channels, style_channels),
            (style_channels, style_channels),
            (style_channels, style_channels),
            (style_channels, style_channels),
        ]

        self.__style_network = nn.Sequential(*[
            LinearBlock(size_in, size_out)
            for size_in, size_out in style_layers
        ])"""

    def forward(
            self,
            z: th.Tensor,
            #z_style: th.Tensor,
            alpha: float
    ) -> th.Tensor:

        #style = self.__style_network(z_style)

        out = z

        for i in range(self.curr_layer):
            m = self.__gen_blocks[i]
            out = m(out)

        out_block = self.__gen_blocks[self.curr_layer](out)

        out_mp = self.__end_block(out_block)

        if self.__last_end_block is not None:
            out_old_mp = self.__last_end_block(out)
            return alpha * out_mp + (1. - alpha) * out_old_mp

        return out_mp

    def next_layer(self) -> bool:
        if self.growing:
            self.__curr_layer += 1

            self.__last_end_block = nn.Sequential(
                nn.Upsample(
                    scale_factor=2.,
                    mode="bilinear",
                    align_corners=True
                ),
                self.__end_block
            )

            self.__end_block = ToMagnPhaseLayer(
                self.__channels[self.curr_layer][1]
            )

            device = "cuda" \
                if next(self.__gen_blocks.parameters()).is_cuda \
                else "cpu"

            self.__end_block.to(device)

            return True

        return False

    @property
    def down_sample(self) -> int:
        return self.__nb_downsample

    @property
    def curr_layer(self) -> int:
        return self.__curr_layer

    @property
    def growing(self) -> bool:
        return self.curr_layer < len(self.__gen_blocks) - 1

    def end_block_params(self) -> Iterator[nn.Parameter]:
        return self.__end_block.parameters()

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return iter(
            list(self.__gen_blocks.parameters(recurse)) +
            list(self.__end_block.parameters(recurse))
            #list(self.__style_network.parameters(recurse))
        )

    def zero_grad(self, set_to_none: bool = False) -> None:
        for p in self.parameters():
            p.grad = None
