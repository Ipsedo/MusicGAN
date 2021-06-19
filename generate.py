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
    parser.add_argument("nb_vec", type=int)
    parser.add_argument("nb_music", type=int)

    parser.add_argument("-o", "--out-dir", type=str, required=True)

    args = parser.parse_args()

    if not exists(args.out_dir):
        mkdir(args.out_dir)
    elif exists(args.out_dir) and not isdir(args.out_dir):
        raise NotADirectoryError(f"\"{args.out_dir}\" is not a directory")

    print("Load model...")
    gen = Generator(args.rand_channel)
    gen.load_state_dict(th.load(args.gen_dict_state))

    with th.no_grad():
        print("Pass rand data to generator...")
        gen_sound = gen(args.nb_music, args.nb_vec)

        print("Saving sound...")
        for i in tqdm(range(gen_sound.size()[0])):
            out_sound_path = join(args.out_dir, f"gen_{i}.wav")

            audio.magn_phase_to_wav(
                gen_sound[i, None, :, :, :].detach(),
                out_sound_path,
                44100
            )


if __name__ == '__main__':
    main()
