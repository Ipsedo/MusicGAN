from utils import SAMPLE_RATE, N_FFT, N_SEC

import scipy.io.wavfile
import soundfile as sf
import torch as th
import torch.fft as th_fft

import matplotlib.pyplot as plt

from typing import List

from tqdm import tqdm

import glob


def to_tensor(wav_paths: List[str], n_fft: int, n_sec: float) -> th.Tensor:
    assert len(wav_paths) > 0, "Empty list !"

    fft_vec_size = n_fft
    batch_vec_nb = int(n_sec * SAMPLE_RATE) // fft_vec_size

    nb_batch = 0
    for wav_p in tqdm(wav_paths):
        # sample_rate, raw_audio = scipy.io.wavfile.read(wav_p)
        raw_audio, sample_rate = sf.read(wav_p)

        assert sample_rate == SAMPLE_RATE, \
            f"Only 44100Hz is supported, " \
            f"actual = {sample_rate}Hz"

        nb_vec = raw_audio.shape[0] // fft_vec_size
        nb_vec -= nb_vec % fft_vec_size

        nb_batch += nb_vec // batch_vec_nb

    data = th.empty(nb_batch, 2, batch_vec_nb, fft_vec_size)

    b_idx = 0

    for wav_p in tqdm(wav_paths):
        # sample_rate, raw_audio = scipy.io.wavfile.read(wav_p)
        raw_audio, sample_rate = sf.read(wav_p)

        data_th = th.from_numpy(raw_audio)

        to_keep = data_th.size(0) - data_th.size(0) % n_fft

        data_th = data_th[:to_keep, :].mean(dim=-1) \
            if len(data_th.size()) > 1 \
            else data_th[:to_keep]

        data_th = th.stack(data_th.split(n_fft, dim=0))

        fft_audio = th_fft.fft(data_th, n=n_fft, dim=-1, norm="forward")

        fft_audio = th.stack([fft_audio.real, fft_audio.imag], dim=-1)

        to_keep = fft_audio.size(0) - fft_audio.size(0) % batch_vec_nb
        fft_audio = fft_audio[:to_keep, :, :]

        # 4s * 44100Hz with n_fft=420 -> 420 FFT complex vectors
        fft_audio = th.stack(fft_audio.split(batch_vec_nb, dim=0), dim=0)

        batch_size = fft_audio.size(0)

        data[b_idx:b_idx + batch_size, :, :, :] = fft_audio.permute(0, 3, 1, 2)

    return data


def to_wav(data: th.Tensor, wav_path: str) -> None:
    data = data.permute(0, 2, 3, 1).flatten(0, 1).contiguous()
    data = th.view_as_complex(data)
    raw_audio = th_fft.ifft(data, n=N_FFT, dim=1, norm="forward").flatten(0, -1).real.numpy()
    scipy.io.wavfile.write(wav_path, SAMPLE_RATE, raw_audio)


if __name__ == '__main__':
    w_p = "/home/samuel/Documents/MusicGAN/res/rammstein/Rammstein - Sehnsucht - 06 - Bueck Dich.mp3.wav"
    w_p = glob.glob(w_p)

    print(N_SEC)

    out_data = to_tensor(w_p, N_FFT, N_SEC)
    print(out_data.size())

    print(out_data[:, 0, :, :].min())
    print(out_data[:, 0, :, :].max())
    print(out_data[:, 0, :, :].mean())
    print((out_data[:, 0, :, :] > 1).sum())
    print((out_data[:, 0, :, :] < -1).sum())
    print()
    print(out_data[:, 1, :, :].min())
    print(out_data[:, 1, :, :].max())
    print(out_data[:, 1, :, :].mean())
    print((out_data[:, 1, :, :] > 1).sum())
    print((out_data[:, 1, :, :] < -1).sum())

    plt.matshow(out_data[20, 0, :, :].numpy())
    plt.show()
    plt.matshow(out_data[20, 1, :, :].numpy())
    plt.show()

    to_wav(out_data, "out.wav")
