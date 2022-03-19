from . import audio
from .networks import Discriminator, Generator

from os.path import join

import torch as th
from torchvision.transforms import Compose, Resize

import matplotlib.pyplot as plt

from typing import List


class Grower:
    def __init__(
            self,
            n_grow: int,
            fadein_lengths: List[int],
            train_lengths: List[int]
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

        self.__fadein_l = fadein_lengths
        self.__train_l = (
            th.tensor(train_lengths)
            .cumsum(dim=0)
            .tolist()
        )

    def grow(self, viewed_samples: int) -> bool:
        self.__sample_idx += viewed_samples
        self.__step_sample_idx += viewed_samples

        if self.__curr_grow >= self.__n_grow:
            return False

        if self.__train_l[self.__curr_grow] < self.__sample_idx:
            self.__step_sample_idx = 0
            self.__curr_grow += 1

            self.__downscale -= 1
            self.__transform = Grower.__get_transform(self.__downscale)
            return True

        return False

    @property
    def alpha(self) -> float:
        return min(
            1.,
            (1. + self.__step_sample_idx) /
            self.__fadein_l[self.__curr_grow]
        )

    @staticmethod
    def __get_transform(downscale_factor: int) -> Compose:
        size = 512

        target_size = size // 2 ** downscale_factor

        compose = Compose([
            audio.ChannelMinMaxNorm(),
            audio.ChangeRange(-1., 1.),
            Resize(target_size)
        ])

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
            optim_disc: th.optim.Adam
    ):
        # Save discriminator
        th.save(
            disc.state_dict(),
            join(self.__output_dir, f"disc_{self.__curr_save}.pt")
        )

        th.save(
            optim_disc.state_dict(),
            join(self.__output_dir, f"optim_disc_{self.__curr_save}.pt")
        )

        # save generator
        th.save(
            gen.state_dict(),
            join(self.__output_dir, f"gen_{self.__curr_save}.pt")
        )

        th.save(
            optim_gen.state_dict(),
            join(self.__output_dir, f"optim_gen_{self.__curr_save}.pt")
        )

    def __save_outputs(
            self,
            gen: Generator,
            alpha: float
    ):
        # Generate sound
        with th.no_grad():

            for gen_idx in range(self.__nb_output_images):

                z = th.randn(
                    1,
                    self.__rand_channels,
                    self.__height, self.__width,
                    device="cuda"
                )

                x_fake = gen(z, alpha)

                c_a = x_fake[0, 0, :, :].detach().cpu().numpy()

                # Plot magnitude
                fig, ax = plt.subplots()

                ax.matshow(
                    c_a / (c_a.max() - c_a.min()),
                    cmap='plasma'
                )

                plt.title(
                    "gen wavelets cA " + str(self.__curr_save) +
                    " grow=" + str(gen.curr_layer)
                )

                fig.savefig(join(
                    self.__output_dir,
                    f"cA_{self.__curr_save}_ID{gen_idx}.png"
                ))

                plt.close()


    def request_save(
            self,
            gen: Generator,
            disc: Discriminator,
            optim_gen: th.optim.Adam,
            optim_disc: th.optim.Adam,
            alpha: float
    ) -> bool:
        self.__counter += 1

        if self.__counter % self.__save_every == 0:

            self.__save_models(
                gen, disc, optim_gen, optim_disc
            )

            self.__save_outputs(
                gen, alpha
            )

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
