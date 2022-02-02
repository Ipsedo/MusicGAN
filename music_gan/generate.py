from .networks import Generator
from . import audio
from .audio import constant

import torch as th

from os import mkdir
from os.path import exists, isdir, join

from tqdm import tqdm


def generate(
        output_dir: str,
        rand_channels: int,
        gen_dict_state: str,
        nb_vec: int,
        nb_music: int
) -> None:

    if not exists(output_dir):
        mkdir(output_dir)
    elif exists(output_dir) and not isdir(output_dir):
        raise NotADirectoryError(
            f"\"{output_dir}\" is not a directory"
        )

    print("Load model...")
    gen = Generator(
        rand_channels,
        end_layer=7
    )

    gen.load_state_dict(
        th.load(gen_dict_state)
    )

    gen.eval()

    height = 2
    width = 2

    with th.no_grad():
        print("Pass rand data to generator...")

        z = th.randn(
            nb_music,
            rand_channels,
            height,
            width * nb_vec
        )

        gen_sound = gen(z, 1.0)

        print("Saving sound...")
        for i in tqdm(range(gen_sound.size()[0])):
            out_sound_path = join(output_dir, f"sound_{i}.wav")

            audio.magn_phase_to_wav(
                gen_sound[i, None, :, :, :].detach(),
                out_sound_path,
                constant.SAMPLE_RATE
            )
