import torch as th
import torch.nn as nn

from typing import Iterator

from .layers import PixelNorm


class Block(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(Block, self).__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.LeakyReLU(2e-1),
            PixelNorm(),

            nn.Upsample(
                scale_factor=2.,
                mode="nearest"
            ),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.LeakyReLU(2e-1),
            PixelNorm()
        )


class ToMagnPhaseLayer(nn.Sequential):
    def __init__(self, in_channels: int):
        super(ToMagnPhaseLayer, self).__init__(
            nn.Conv2d(
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
            end_layer: int = 0
    ):
        super(Generator, self).__init__()

        self.__curr_layer = end_layer

        self.__nb_downsample = 7

        channels = [
            (rand_channels, 128),
            (128, 112),
            (112, 96),
            (96, 80),
            (80, 64),
            (64, 48),
            (48, 32),
            (32, 16)
        ]

        self.__channels = channels

        assert 0 <= end_layer < len(channels), f"0 <= {end_layer} < {len(channels)}"

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            Block(c[0], c[1])
            for i, c in enumerate(channels)
        ])

        # for progressive gan
        self.__end_block = ToMagnPhaseLayer(
            channels[self.curr_layer][1]
        )

        self.__last_end_block = (
            None if self.__curr_layer == 0
            else nn.Sequential(
                ToMagnPhaseLayer(
                    channels[self.curr_layer - 1][1]
                ),
                nn.Upsample(
                    scale_factor=2.,
                    mode="nearest"
                )
            )
        )

    def forward(
            self,
            z: th.Tensor,
            alpha: float
    ) -> th.Tensor:

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
                self.__end_block,
                nn.Upsample(
                    scale_factor=2.,
                    mode="nearest"
                )
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

    def zero_grad(self, set_to_none: bool = False) -> None:
        for p in self.parameters():
            p.grad = None
