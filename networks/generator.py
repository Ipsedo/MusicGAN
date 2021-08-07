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


class Block(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(Block, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            PixelNorm(),
            nn.LeakyReLU(2e-1)
        )


class UpSampleBlock(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(UpSampleBlock, self).__init__(
            nn.Upsample(
                scale_factor=2.,
                mode="bilinear",
                align_corners=True
            ),
            Block(
                in_channels,
                out_channels
            )
        )


class ToMagnPhaseLayer(nn.Sequential):
    def __init__(self, in_channels: int):
        super(ToMagnPhaseLayer, self).__init__(
            nn.Conv2d(
                in_channels, 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
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

        self.__nb_downsample = 8

        channels = [
            (144, 128),
            (128, 112),
            (112, 96),
            (96, 80),
            (80, 64),
            (64, 48),
            (48, 32),
            (32, 16)
        ]

        self.__channels = channels

        assert 0 <= end_layer < len(channels)

        self.__first_block = Block(
            rand_channels,
            channels[0][0]
        )

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            UpSampleBlock(
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
                ToMagnPhaseLayer(
                    channels[self.curr_layer - 1][1]
                ),
                nn.Upsample(
                    scale_factor=2.,
                    mode="bilinear",
                    align_corners=True
                )
            )
        )

    def forward(
            self,
            z: th.Tensor,
            alpha: float
    ) -> th.Tensor:

        out = self.__first_block(z)

        for i in range(self.curr_layer):
            m = self.__gen_blocks[i]
            out = m(out)

        m = self.__gen_blocks[self.curr_layer]
        out_blocks = m(out)

        out_mp = self.__end_block(out_blocks)

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
                    mode="bilinear",
                    align_corners=True
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

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return iter(
            list(self.__first_block.parameters()) +
            list(self.__gen_blocks.parameters(recurse)) +
            list(self.__end_block.parameters(recurse))
        )
