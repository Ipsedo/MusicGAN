from networks import Generator
import audio

import torch as th

from os import mkdir
from os.path import exists, isdir, join

import argparse

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser("Generate sound")

    parser.add_argument("gen_dict_state", type=str)
    parser.add_argument("rand_channels", type=int)
    parser.add_argument("style_rand_channels", type=int)
    parser.add_argument("-n", "--nb-vec", type=int)
    parser.add_argument("-m", "--nb-music", type=int)

    parser.add_argument("-o", "--out-dir", type=str, required=True)

    args = parser.parse_args()

    if not exists(args.out_dir):
        mkdir(args.out_dir)
    elif exists(args.out_dir) and not isdir(args.out_dir):
        raise NotADirectoryError(
            f"\"{args.out_dir}\" is not a directory"
        )

    print("Load model...")
    gen = Generator(
        args.rand_channels,
        args.style_rand_channels,
        end_layer=7
    )

    gen.load_state_dict(
        th.load(args.gen_dict_state)
    )

    gen.eval()

    height = 2
    width = 2

    with th.no_grad():
        print("Pass rand data to generator...")

        z = th.randn(
            args.nb_music,
            args.rand_channels,
            height,
            width * args.nb_vec
        )

        z_style = th.randn(
            args.nb_music,
            args.style_rand_channels
        )

        gen_sound = gen(z, z_style, 1.0)

        print("Saving sound...")
        for i in tqdm(range(gen_sound.size()[0])):
            out_sound_path = join(args.out_dir, f"sound_{i}.wav")

            audio.magn_phase_to_wav(
                gen_sound[i, None, :, :, :].detach(),
                out_sound_path,
                44100
            )


if __name__ == '__main__':
    main()
