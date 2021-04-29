from utils import SAMPLE_RATE, N_FFT, N_SEC

import scipy.io.wavfile
import scipy.signal
import soundfile as sf
import pywt
import torch as th
import torch.nn.functional as th_f
import torchaudio as th_audio
import torchaudio.functional as th_audio_f
import torch.fft as th_fft
import numpy as np

import seaborn

import matplotlib.pyplot as plt

from typing import List

from tqdm import tqdm

import glob


##############
# WaveLet
##############

def to_tensor_old(wav_paths: List[str], n_fft: int, n_sec: float) -> th.Tensor:
    assert len(wav_paths) > 0, "Empty list !"

    fft_vec_size = n_fft
    batch_vec_nb = int(n_sec * SAMPLE_RATE) // fft_vec_size

    nb_batch = 0
    total_tick = 0
    for wav_p in tqdm(wav_paths):
        # sample_rate, raw_audio = scipy.io.wavfile.read(wav_p)
        raw_audio, sample_rate = sf.read(wav_p)

        assert sample_rate == SAMPLE_RATE, \
            f"Only 44100Hz is supported, " \
            f"actual = {sample_rate}Hz"

        nb_vec = raw_audio.shape[0] // fft_vec_size
        nb_vec -= nb_vec % fft_vec_size

        total_tick += raw_audio.shape[0]

        nb_batch += nb_vec // batch_vec_nb

    data = th.empty(nb_batch, 2, batch_vec_nb, fft_vec_size)
    print(f"{total_tick // SAMPLE_RATE // 60 // 60 // 24}d "
          f"{total_tick // SAMPLE_RATE // 60 // 60 % 24}h "
          f"{total_tick // SAMPLE_RATE // 60 % 60:02d}m "
          f"{total_tick // SAMPLE_RATE % 60:02d}s "
          f"audio")
    print(f"record : {N_SEC}s")

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


def to_tensor_wavelet(wav_paths: List[str], n_fft: int,
                      n_sec: float) -> th.Tensor:
    assert len(wav_paths) > 0, "Empty list !"

    fft_vec_size = n_fft
    batch_vec_nb = int(n_sec * SAMPLE_RATE) // fft_vec_size

    nb_batch = 0
    total_tick = 0
    for wav_p in tqdm(wav_paths):
        # sample_rate, raw_audio = scipy.io.wavfile.read(wav_p)
        raw_audio, sample_rate = sf.read(wav_p)

        assert sample_rate == SAMPLE_RATE, \
            f"Only 44100Hz is supported, " \
            f"actual = {sample_rate}Hz"

        nb_vec = raw_audio.shape[0] // fft_vec_size
        nb_vec -= nb_vec % fft_vec_size

        total_tick += raw_audio.shape[0]

        nb_batch += nb_vec // batch_vec_nb

    data = th.empty(nb_batch, 2, batch_vec_nb, fft_vec_size // 2)
    print(f"{total_tick // SAMPLE_RATE // 60 // 60 // 24}d "
          f"{total_tick // SAMPLE_RATE // 60 // 60 % 24}h "
          f"{total_tick // SAMPLE_RATE // 60 % 60:02d}m "
          f"{total_tick // SAMPLE_RATE % 60:02d}s "
          f"audio")
    print(f"record : {N_SEC}s")

    b_idx = 0

    for wav_p in tqdm(wav_paths):
        # sample_rate, raw_audio = scipy.io.wavfile.read(wav_p)
        raw_audio, sample_rate = sf.read(wav_p)

        to_keep = raw_audio.shape[0] - raw_audio.shape[0] % n_fft

        raw_audio = raw_audio[:to_keep, :].mean(axis=-1) \
            if len(raw_audio.shape) > 1 \
            else raw_audio[:to_keep]

        data_th = np.stack(
            np.split(raw_audio, raw_audio.shape[0] // n_fft, axis=-1),
            axis=0)

        wv_1, wv_2 = pywt.dwt(data_th, "db1", axis=-1)

        wv = np.stack([wv_1, wv_2], axis=-1)

        to_keep = wv.shape[0] - wv.shape[0] % batch_vec_nb
        wv = wv[:to_keep, :, :]

        wv = np.stack(np.split(wv, wv.shape[0] // batch_vec_nb, axis=0), axis=0)

        batch_size = wv.shape[0]

        wv_th = th.from_numpy(wv)

        data[b_idx:b_idx + batch_size, :, :, :] = wv_th.permute(0, 3, 1, 2)

    return data


def to_wav_old(data: th.Tensor, wav_path: str) -> None:
    data = data.permute(0, 2, 3, 1).flatten(0, 1).contiguous()
    data = th.view_as_complex(data)
    raw_audio = th_fft.ifft(data, n=N_FFT, dim=1, norm="forward").flatten(0,
                                                                          -1).real.numpy()
    scipy.io.wavfile.write(wav_path, SAMPLE_RATE, raw_audio)


def wavelet_to_wav(data: th.Tensor, wav_path: str) -> None:
    data = data.permute(0, 2, 3, 1).flatten(0, 1).contiguous().numpy()
    # raw_audio = th_fft.ifft(data, n=N_FFT, dim=1, norm="forward").flatten(0, -1).real.numpy()
    raw_audio = pywt.idwt(data[:, :, 0], data[:, :, 1], "db1",
                          axis=-1).flatten()
    scipy.io.wavfile.write(wav_path, SAMPLE_RATE, raw_audio)


###########
# Ticks
###########

def to_tensor_ticks(
        wav_paths: List[str],
        sample_rate: int,
        channels: int,
        per_batch_sample: int
) -> th.Tensor:
    nb_batch = 0

    for wav_p in tqdm(wav_paths):
        # sample_rate, raw_audio = scipy.io.wavfile.read(wav_p)
        raw_audio, curr_sample_rate = sf.read(wav_p)

        assert sample_rate == curr_sample_rate, \
            f"Needed = {sample_rate}Hz, " \
            f"actual = {curr_sample_rate}Hz"

        nb_batch += raw_audio.shape[0] // per_batch_sample

    data = th.empty(nb_batch, channels, per_batch_sample)

    actual_batch = 0

    for wav_p in tqdm(wav_paths):
        raw_audio, _ = sf.read(wav_p)

        to_keep = raw_audio.shape[0] - raw_audio.shape[0] % per_batch_sample

        raw_audio = raw_audio[:to_keep, :].mean(axis=-1)[:, None] \
            if channels == 1 else raw_audio[:to_keep, :]

        raw_audio_splitted = np.stack(
            np.split(raw_audio, raw_audio.shape[0] // per_batch_sample, axis=0),
            axis=0)

        data[actual_batch:actual_batch + raw_audio_splitted.shape[0]] = \
            th.from_numpy(raw_audio_splitted).permute(0, 2, 1)

        actual_batch += raw_audio_splitted.shape[0]

    return data


def ticks_to_wav(data: th.Tensor, wav_path: str, sample_rate: int) -> None:
    raw_audio = data.permute(0, 2, 1).flatten(0, 1).contiguous().numpy()
    scipy.io.wavfile.write(wav_path, sample_rate, raw_audio)


##########
# STFT
##########
def diff(x):
    return th_f.pad(x[1:, :] - x[:-1, :], (0, 0, 1, 0), "constant", 0)


def unwrap(phi):
    dphi = diff(phi)
    dphi_m = ((dphi + np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m == -np.pi) & (dphi > 0)] = np.pi
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < np.pi] = 0
    return phi + phi_adj.cumsum(0)


def to_tensor_stft(
        wav_paths: List[str],
        sample_rate: int,

) -> th.Tensor:
    nperseg = 1022
    stride = 256

    nb_vec = 256

    nb_batch = 0

    n_fft = 512

    for wav_p in tqdm(wav_paths):
        # sample_rate, raw_audio = scipy.io.wavfile.read(wav_p)
        raw_audio, curr_sample_rate = th_audio.load(wav_p)

        assert sample_rate == curr_sample_rate, \
            f"Needed = {sample_rate}Hz, " \
            f"actual = {curr_sample_rate}Hz"

        nb_seg = raw_audio.size()[1] // stride

        nb_batch += nb_seg // nb_vec

    data = th.empty(nb_batch, 2, nb_vec, n_fft)

    curr_batch = 0

    hann_window = th.hann_window(nperseg)

    for wav_p in tqdm(wav_paths):
        raw_audio, _ = th_audio.load(wav_p)

        raw_audio_mono = raw_audio.mean(0)

        complex_values = th_audio_f.spectrogram(
            raw_audio_mono,
            pad=0, window=hann_window,
            n_fft=nperseg, hop_length=stride, win_length=nperseg,
            power=None, normalized=True
        )

        complex_values = complex_values.permute(1, 0, 2)

        magn, phase = th_audio_f.magphase(complex_values)

        magn = th.log(magn + 1)

        phase = unwrap(phase)
        phase = phase[1:, :] - phase[:-1, :]
        magn = magn[1:, :]

        magn = magn[magn.size()[0] % nb_vec:, :]
        phase = phase[phase.size()[0] % nb_vec:, :]
        magn = th.stack(magn.split(nb_vec, dim=0), dim=0)
        phase = th.stack(phase.split(nb_vec, dim=0), dim=0)

        max_magn = magn.max()
        min_magn = magn.min()
        max_phase = phase.max()
        min_phase = phase.min()

        magn = (magn - min_magn) / (max_magn - min_magn)
        phase = (phase - min_phase) / (max_phase - min_phase)

        data[curr_batch:curr_batch + magn.size()[0], 0, :, :] = \
            magn * 2. - 1.

        data[curr_batch:curr_batch + phase.size()[0], 1, :, :] = \
            phase * 2. - 1.

        curr_batch += phase.size()[0]

    return data


def stft_to_wav(x: th.Tensor, wav_path: str, sample_rate: int):
    x = x.permute(0, 2, 3, 1)
    x = x.flatten(0, 1)

    phases = x[:, :, 1]
    for i in range(phases.shape[0] - 1):
        phases[i + 1, :] = phases[i + 1, :] + phases[i, :]

    phases = phases * np.pi % (2 * np.pi)

    magn = (x[:, :, 0] + 1) / 2
    magn = th.exp(magn) - 1

    real = magn * th.cos(phases)
    imag = magn * th.sin(phases)

    real = real.numpy()
    imag = imag.numpy()

    x = real + imag * 1j
    _, raw_audio = scipy.signal.istft(x.transpose(), nperseg=1022,
                                      noverlap=1022 - 256)
    scipy.io.wavfile.write(wav_path, sample_rate, raw_audio)


if __name__ == '__main__':
    w_p = "/home/samuel/Documents/MusicGAN/res/rammstein_16000Hz/(2) Links 234.wav"
    w_p = glob.glob(w_p)

    """print(N_SEC)

    out_data = to_tensor_wavelet(w_p, N_FFT, N_SEC * 2)
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

    plt.matshow(out_data[10, 0, :, :].numpy())
    plt.show()
    plt.matshow(out_data[10, 1, :, :].numpy())
    plt.show()

    seaborn.displot(out_data[:, 0, :, :].flatten(0,-1))
    plt.show()
    seaborn.displot(out_data[:, 1, :, :].flatten(0,-1))
    plt.show()

    wavelet_to_wav(out_data, "out.wav")"""

    """out_data = to_tensor_ticks(w_p, 16000, 1, 16000)

    print(out_data.max())
    print(out_data.min())

    ticks_to_wav(out_data, "test.wav", 16000)

    print(out_data.size())"""

    out = to_tensor_stft(w_p, 16000)

    print(out[:, 0, :, :].max(), out[:, 0, :, :].min())
    print(out[:, 1, :, :].max(), out[:, 1, :, :].min())

    print(out.size())

    stft_to_wav(out, "out.wav", 16000)

    idx = 40

    real, imag = out[idx, 0, :, :].numpy(), out[idx, 1, :, :].numpy()

    # imag = (imag + 1) / 2 - (out[idx - 1, 1, :, :].numpy() + 1) / 2

    fig, ax = plt.subplots()
    ax.matshow(real / (real.max() - real.min()), cmap='plasma')
    fig.show()

    fig, ax = plt.subplots()
    ax.matshow(imag / (imag.max() - imag.min()), cmap='plasma')
    fig.show()
