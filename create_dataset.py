import torch as th

from tqdm import tqdm

import glob

from os.path import exists, join, isdir
from os import mkdir

import audio

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create dataset")

    parser.add_argument("audio_path", type=str, help="can be /path/to/*.wav")
    parser.add_argument("-sr", "--sample-rate", type=int, default=44100)
    parser.add_argument("-o", "--output-dir", type=str, required=True)

    args = parser.parse_args()

    w_p = glob.glob(args.audio_path)

    out_path = args.output_dir
    if not exists(out_path):
        mkdir(out_path)
    elif exists(out_path) and not isdir(out_path):
        raise NotADirectoryError(f"\"{out_path}\" is not a directory")

    nperseg = 1022
    stride = 256

    nb_vec = 512

    n_fft = 512

    idx = 0

    for wav_p in tqdm(w_p):
        complex_values = audio.wav_to_stft(
            wav_p,
            nperseg=nperseg,
            stride=stride
        )

        if complex_values.size()[0] < nb_vec:
             continue

        magn, phase = audio.stft_to_phase_magn(
            complex_values,
            nb_vec=nb_vec
        )

        nb_sample = magn.size()[0]

        for s_idx in range(nb_sample):
            s_magn = magn[s_idx].to(th.float64)
            s_phase = phase[s_idx].to(th.float64)

            magn_phase_path = join(out_path, f"magn_phase_{idx}.pt")

            magn_phase = th.stack([s_magn, s_phase], dim=0)

            th.save(magn_phase, magn_phase_path)

            idx += 1
