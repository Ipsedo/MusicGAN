from os.path import join
from typing import List

import matplotlib.pyplot as plt
import torch as th
from torchvision.transforms import Compose, Resize
from tqdm import tqdm

from . import audio
from .networks import Discriminator, Generator


class Grower:
    def __init__(
        self, n_grow: int, fadein_lengths: List[int], train_lengths: List[int]
    ):
        self.__curr_grow = 0
        self.__n_grow = n_grow

        # view sample counter
        self.__sample_idx = 0
        # current grow step counter
        self.__step_sample_idx = 0

        # image downscale factor
        self.__downscale = 7
        # image/tensor transformation
        self.__transform = Grower.__get_transform(self.__downscale)

        # +1 because of last layer
        assert len(fadein_lengths) == self.__n_grow + 1
        assert len(train_lengths) == self.__n_grow

        self.__fadein_lengths = fadein_lengths

        self.__train_lengths = train_lengths
        self.__train_lengths_cumsum = (
            th.tensor(train_lengths).cumsum(dim=0).tolist()
        )

        self.__init_tqdm_bars()

    def __init_tqdm_bars(self) -> None:
        self.__tqdm_bar_fadein = tqdm(
            range(self.__fadein_lengths[self.__curr_grow]),
            position=1,
            leave=False,
        )

        if self.__curr_grow < self.__n_grow:
            self.__tqdm_bar_grow = tqdm(
                range(self.__train_lengths[self.__curr_grow]),
                position=2,
                leave=False,
            )

    def __update_bars(self) -> None:

        self.__tqdm_bar_fadein.set_description(f"⌙> fade in ")

        self.__tqdm_bar_grow.set_description(
            f"⌙> grow [{self.__curr_grow} / {self.__n_grow}] "
        )

        if self.__step_sample_idx <= self.__fadein_lengths[self.__curr_grow]:
            self.__tqdm_bar_fadein.update(1)

        if self.__curr_grow < self.__n_grow:
            self.__tqdm_bar_grow.update(1)

    def grow(self) -> bool:
        self.__sample_idx += 1
        self.__step_sample_idx += 1

        self.__update_bars()

        if self.__curr_grow >= self.__n_grow:
            return False

        if self.__train_lengths_cumsum[self.__curr_grow] < self.__sample_idx:
            self.__step_sample_idx = 0
            self.__curr_grow += 1

            self.__downscale -= 1
            self.__transform = Grower.__get_transform(self.__downscale)

            self.__init_tqdm_bars()

            return True

        return False

    @property
    def alpha(self) -> float:
        return min(
            1.0,
            (1.0 + self.__step_sample_idx)
            / self.__fadein_lengths[self.__curr_grow],
        )

    @staticmethod
    def __get_transform(downscale_factor: int) -> Compose:
        size = 512

        target_size = size // 2**downscale_factor

        compose = Compose(
            [
                audio.ChannelMinMaxNorm(),
                audio.ChangeRange(-1.0, 1.0),
                Resize(target_size),
            ]
        )

        return compose

    @property
    def scale_transform(self) -> Compose:
        return self.__transform


class Saver:
    def __init__(
        self,
        output_dir: str,
        save_every: int,
        rand_channels: int,
        rand_height: int = 2,
        rand_width: int = 2,
    ):
        self.__output_dir = output_dir

        # forward / backward pass
        self.__counter = 0
        # the current save index
        self.__curr_save = 0
        # save every each N pass
        self.__save_every = save_every

        # for sounds/images generation
        self.__rand_channels = rand_channels

        # input random image
        # usually 2 * 2 rand pixels
        self.__height = rand_height
        self.__width = rand_width

        # produce 6 outputs per save
        self.__nb_output_images = 6

    def __save_models(
        self,
        gen: Generator,
        disc: Discriminator,
        optim_gen: th.optim.Adam,
        optim_disc: th.optim.Adam,
    ) -> None:
        # Save discriminator
        th.save(
            disc.state_dict(),
            join(self.__output_dir, f"disc_{self.__curr_save}.pt"),
        )

        th.save(
            optim_disc.state_dict(),
            join(self.__output_dir, f"optim_disc_{self.__curr_save}.pt"),
        )

        # save generator
        th.save(
            gen.state_dict(),
            join(self.__output_dir, f"gen_{self.__curr_save}.pt"),
        )

        th.save(
            optim_gen.state_dict(),
            join(self.__output_dir, f"optim_gen_{self.__curr_save}.pt"),
        )

    def __save_outputs(self, gen: Generator, alpha: float) -> None:
        # Generate sound
        with th.no_grad():

            for gen_idx in range(self.__nb_output_images):

                z = th.randn(
                    1,
                    self.__rand_channels,
                    self.__height,
                    self.__width,
                    device="cuda",
                )

                x_fake = gen(z, alpha)

                magn = x_fake[0, 0, :, :].detach().cpu().numpy()
                phase = x_fake[0, 1, :, :].detach().cpu().numpy()

                # create two subplots
                fig, (magn_ax, phase_ax) = plt.subplots(1, 2)

                # Plot magnitude
                magn_ax.matshow(
                    magn / (magn.max() - magn.min()), cmap="plasma"
                )

                magn_ax.set_title(
                    "gen magn "
                    + str(self.__curr_save)
                    + " grow="
                    + str(gen.curr_layer)
                )

                # Plot phase
                phase_ax.matshow(
                    phase / (phase.max() - phase.min()), cmap="plasma"
                )

                phase_ax.set_title(
                    "gen phase "
                    + str(self.__curr_save)
                    + " grow="
                    + str(gen.curr_layer)
                )

                fig.savefig(
                    join(
                        self.__output_dir,
                        f"magn_phase_{self.__curr_save}_ID{gen_idx}.png",
                    )
                )

                plt.close()

    def request_save(
        self,
        gen: Generator,
        disc: Discriminator,
        optim_gen: th.optim.Adam,
        optim_disc: th.optim.Adam,
        alpha: float,
    ) -> bool:
        self.__counter += 1

        if self.__counter % self.__save_every == 0:

            self.__save_models(gen, disc, optim_gen, optim_disc)

            self.__save_outputs(gen, alpha)

            self.__curr_save += 1

            return True

        return False

    @property
    def curr_save(self) -> int:
        # curr_save - 1 because we want last saved step
        return self.__curr_save - 1

    @property
    def save_counter(self) -> int:
        return self.__counter % self.__save_every
