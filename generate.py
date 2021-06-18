from networks import Generator
from create_dataset import stft_to_wav

import torch as th

from os import mkdir
from os.path import exists, isdir, join

import argparse

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser("Generate sound")

    parser.add_argument("gen_dict_state", type=str)
    parser.add_argument("nb_vec", type=int)
    parser.add_argument("nb_music", type=int)

    parser.add_argument("-o", "--out-dir", type=str, required=True)

    args = parser.parse_args()

    rand_channel = 32

    nb_height = 4
    nb_width = args.nb_vec

    if not exists(args.out_dir):
        mkdir(args.out_dir)
    elif exists(args.out_dir) and not isdir(args.out_dir):
        raise NotADirectoryError(f"\"{args.out_dir}\" is not a directory")

    print("Load model...")
    gen = Generator(rand_channel, 2)
    gen.load_state_dict(th.load(args.gen_dict_state))

    rand_data = th.randn(args.nb_music, rand_channel, nb_width, nb_height)

    with th.no_grad():
        print("Pass rand data to generator...")
        gen_sound = gen(rand_data)

        print("Saving sound...")
        for i in tqdm(range(gen_sound.size()[0])):
            out_sound_path = join(args.out_dir, f"gen_{i}.wav")

            stft_to_wav(gen_sound[i, None, :, :, :].detach(), out_sound_path, 44100)


if __name__ == '__main__':
    main()
