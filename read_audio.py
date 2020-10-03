from utils import WavInfo, FFTAudio, SAMPLE_RATE, N_FFT, N_SEC

import scipy.io.wavfile
import numpy as np
import torch as th

from typing import List

from tqdm import tqdm

import glob


def to_tensor(wav_paths: List[str], n_fft: int, n_sec: int) -> th.Tensor:
    assert len(wav_paths) > 0, "Empty list !"

    nb_batch = 0
    for wav_p in tqdm(wav_paths):
        sample_rate, raw_audio = scipy.io.wavfile.read(wav_p)

        assert sample_rate == SAMPLE_RATE, "Only 44100Hz is supported"

        nb_batch += raw_audio.shape[0] // n_fft // (
                    n_sec * sample_rate // n_fft)

    data = th.empty(nb_batch, 2, n_sec * SAMPLE_RATE // n_fft, n_fft)

    b_idx = 0

    for wav_p in tqdm(wav_paths):
        sample_rate, raw_audio = scipy.io.wavfile.read(wav_p)

        data_th = th.from_numpy(raw_audio / np.iinfo(raw_audio.dtype).max)
        to_keep = data_th.size(0) - data_th.size(0) % n_fft
        data_th = data_th[:to_keep, :].mean(dim=-1)
        data_th = th.stack(data_th.split(n_fft, dim=0)).unsqueeze(-1)

        fft_audio = th.rfft(data_th, signal_ndim=1).squeeze(-2)
        to_keep = fft_audio.size(0) - fft_audio.size(0) % n_fft
        fft_audio = fft_audio[:to_keep, :, :]

        # 4s * 44100Hz with n_fft=420 -> 420 FFT complex vectors
        fft_audio = th.stack(fft_audio.split(n_fft, dim=0), dim=0)

        batch_size = fft_audio.size(0)

        data[b_idx:b_idx + batch_size, :, :, :] = fft_audio.permute(0, 3, 1, 2)

    return data


def to_wav(data: th.Tensor, wav_path: str) -> None:
    data = data.permute(0, 2, 3, 1).flatten(0, 1).unsqueeze(-2)
    raw_audio = th.irfft(data, signal_ndim=1).flatten(0, -1)
    scipy.io.wavfile.write(wav_path, SAMPLE_RATE, raw_audio.numpy())


if __name__ == '__main__':
    w_p = "/home/samuel/Documents/MusicAutoEncoder/res/rammstein/01 Rammlied.mp3.wav"
    w_p = glob.glob(w_p)

    out_data = to_tensor(w_p, N_FFT, N_SEC)
    print(out_data.size())

    to_wav(out_data, "out.wav")

