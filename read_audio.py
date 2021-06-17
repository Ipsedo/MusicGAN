import scipy.io.wavfile
import scipy.signal

import torch as th
import torch.nn.functional as th_f
import torchaudio as th_audio
import torchaudio.functional as th_audio_f

import numpy as np

import matplotlib.pyplot as plt

from typing import List

from tqdm import tqdm

import glob

from os.path import exists, join, isdir
from os import mkdir


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

    nb_vec = 512

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
            magn * 1.6 - 0.8

        data[curr_batch:curr_batch + phase.size()[0], 1, :, :] = \
            phase * 1.6 - 0.8

        curr_batch += phase.size()[0]

    return data


def stft_to_wav(x: th.Tensor, wav_path: str, sample_rate: int):
    x = x.permute(0, 2, 3, 1)
    x = x.flatten(0, 1)

    phases = (x[:, :, 1] + 0.8) / 1.6 * 2. * np.pi - np.pi
    for i in range(phases.shape[0] - 1):
        phases[i + 1, :] = phases[i + 1, :] + phases[i, :]

    phases = phases % (2 * np.pi)

    magn = (x[:, :, 0] + 0.8) / 1.6
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
    w_p = "/home/samuel/Documents/MusicGAN/out/train_stft_mozart_symphonies_batchnorm/gen_epoch_234_ID0.wav"
    w_p = "/home/samuel/Documents/MusicGAN/out/train_stylegan_mozart/gen_epoch_2_ID1.wav"
    #w_p = "/home/samuel/Documents/MusicGAN/out/train_stft_piano/gen_epoch_282_ID0.wav"
    #w_p = "/home/samuel/Documents/MusicGAN/out/train_stft_mozart_symphonies_onlylrelu/gen_epoch_2_ID1.wav"
    #w_p = "/home/samuel/Documents/MusicGAN/out/train_stft_mozart_symphonies_pixelnorm/gen_epoch_0_ID0.wav"
    #w_p = "/home/samuel/Documents/MusicGAN/out/train_stft_led_zeppelin/gen_epoch_12_ID0.wav"
    #w_p = "/home/samuel/Documents/MusicGAN/res/mozart_symphonie_flac/02 - Symphony No. 20 in D Major, K133 - II. Andante.flac"
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

    out = to_tensor_stft(w_p, 44100)

    print(out[:, 0, :, :].max(), out[:, 0, :, :].min())
    print(out[:, 1, :, :].max(), out[:, 1, :, :].min())

    print(out.size())

    out_path = join(".", "out")
    if not exists(out_path):
        mkdir(out_path)
    elif exists(out_path) and not isdir(out_path):
        raise NotADirectoryError(f"\"{out_path}\" is not a directory")

    out_wav_path = join(out_path, "read_audio_out.wav")
    stft_to_wav(out, out_wav_path, 44100)

    idx = 2

    real, imag = out[idx, 0, :, :].numpy(), out[idx, 1, :, :].numpy()

    # imag = (imag + 1) / 2 - (out[idx - 1, 1, :, :].numpy() + 1) / 2

    fig, ax = plt.subplots()
    ax.matshow(real / (real.max() - real.min()), cmap='plasma')
    fig.show()

    fig, ax = plt.subplots()
    ax.matshow(imag / (imag.max() - imag.min()), cmap='plasma')
    fig.show()
