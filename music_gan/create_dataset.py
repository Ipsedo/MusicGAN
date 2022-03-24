import torch as th

from tqdm import tqdm

import glob

from os.path import exists, join, isdir
from os import mkdir

from . import audio


def create_dataset(
        audio_path: str,
        dataset_output_dir: str
) -> None:

    w_p = glob.glob(audio_path)

    if not exists(dataset_output_dir):
        mkdir(dataset_output_dir)
    elif exists(dataset_output_dir) and not isdir(dataset_output_dir):
        raise NotADirectoryError(
            f"\"{dataset_output_dir}\" is not a directory"
        )

    nperseg = audio.N_FFT

    nb_vec = audio.N_VEC

    idx = 0

    for wav_p in tqdm(w_p):
        c_a = audio.wav_to_wavelets(
            wav_p,
            nperseg=nperseg
        )

        if c_a.size()[1] < nb_vec:
            continue

        c_a = audio.prepare_wavelets(c_a, nb_vec)

        nb_sample = c_a.size()[0]

        for s_idx in range(nb_sample):

            wavelets_path = join(
                dataset_output_dir,
                f"wavelets_{idx}.pt"
            )

            th.save(c_a[s_idx].to(th.float64), wavelets_path)

            idx += 1
        return
