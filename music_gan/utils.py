from os.path import join
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch as th
from torchvision.transforms import Compose, Resize
from tqdm import tqdm

from . import audio
from .networks import Discriminator, Generator, LEAKY_RELU_SLOPE


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
            gen: Generator
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

                x_fake = gen(z)

                magn = x_fake[0, 0, :, :].detach().cpu().numpy()
                phase = x_fake[0, 1, :, :].detach().cpu().numpy()

                # create two subplots
                fig, (magn_ax, phase_ax) = plt.subplots(1, 2)

                # Plot magnitude
                magn_ax.matshow(
                    magn / (magn.max() - magn.min()),
                    cmap='plasma'
                )

                magn_ax.set_title(
                    "gen magn " + str(self.__curr_save)
                )

                # Plot phase
                phase_ax.matshow(
                    phase / (phase.max() - phase.min()),
                    cmap='plasma'
                )

                phase_ax.set_title(
                    "gen phase " + str(self.__curr_save)
                )

                fig.savefig(join(
                    self.__output_dir,
                    f"magn_phase_{self.__curr_save}_ID{gen_idx}.png"
                ))

                plt.close()

    def request_save(
            self,
            gen: Generator,
            disc: Discriminator,
            optim_gen: th.optim.Adam,
            optim_disc: th.optim.Adam
    ) -> bool:
        self.__counter += 1

        if self.__counter % self.__save_every == 0:

            self.__save_models(
                gen, disc, optim_gen, optim_disc
            )

            self.__save_outputs(
                gen
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
