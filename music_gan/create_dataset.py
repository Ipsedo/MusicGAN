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
    stride = audio.STFT_STRIDE

    nb_vec = audio.N_VEC

    idx = 0

    for wav_p in tqdm(w_p):
        complex_values = audio.wav_to_stft(
            wav_p,
            nperseg=nperseg,
            stride=stride
        )

        if complex_values.size()[1] < nb_vec:
            continue

        magn, phase = audio.stft_to_phase_magn(
            complex_values,
            nb_vec=nb_vec
        )

        nb_sample = magn.size()[0]

        for s_idx in range(nb_sample):
            s_magn = magn[s_idx, :, :].to(th.float64)
            s_phase = phase[s_idx, :, :].to(th.float64)

            magn_phase_path = join(
                dataset_output_dir,
                f"magn_phase_{idx}.pt"
            )

            magn_phase = th.stack([s_magn, s_phase], dim=0)

            th.save(magn_phase, magn_phase_path)

            idx += 1
