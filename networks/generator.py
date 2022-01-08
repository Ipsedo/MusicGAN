import torch as th
import torch.nn as nn

from typing import Iterator

from .layers import AdaIN


class Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_channels: int
    ):
        super(Block, self).__init__()

        self.__conv_tr = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1)
        )

        self.__adain = AdaIN(
            out_channels,
            style_channels
        )

        self.__lr_relu = nn.LeakyReLU(2e-1)

    def forward(self, x: th.Tensor, style: th.Tensor) -> th.Tensor:
        out = self.__conv_tr(x)
        out = self.__adain(out, style)
        out = self.__lr_relu(out)

        return out


class StyleBlock(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(StyleBlock, self).__init__(
            nn.Linear(in_channels, out_channels),
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


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            style_rand_channels: int,
            end_layer: int = 0
    ):
        super(Generator, self).__init__()

        self.__curr_layer = end_layer

        self.__nb_downsample = 7

        channels = [
            (rand_channels, 256),
            (256, 224),
            (224, 192),
            (192, 160),
            (160, 128),
            (128, 96),
            (96, 64),
            (64, 32)
        ]

        self.__channels = channels

        self.__style_channels = 256

        assert 0 <= end_layer < len(channels)

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            Block(
                c[0], c[1],
                self.__style_channels
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

        style_channels = [
            (style_rand_channels, self.__style_channels),
            (self.__style_channels, self.__style_channels),
            (self.__style_channels, self.__style_channels),
            (self.__style_channels, self.__style_channels),
            (self.__style_channels, self.__style_channels),
            (self.__style_channels, self.__style_channels),
            (self.__style_channels, self.__style_channels),
            (self.__style_channels, self.__style_channels),
        ]

        self.__style_network = nn.Sequential(*[
            StyleBlock(c[0], c[1])
            for c in style_channels
        ])

    def forward(
            self,
            z: th.Tensor,
            z_style: th.Tensor,
            alpha: float
    ) -> th.Tensor:

        style = self.__style_network(z_style)

        out = z

        for i in range(self.curr_layer):
            m = self.__gen_blocks[i]
            out = m(out, style)

        out_block = self.__gen_blocks[self.curr_layer](out, style)

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
            list(self.__end_block.parameters(recurse)) +
            list(self.__style_network.parameters(recurse))
        )

    def zero_grad(self, set_to_none: bool = False) -> None:
        for p in self.parameters():
            p.grad = None
