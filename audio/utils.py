import torch as th
import torch.nn.functional as th_f
import torchaudio as th_audio
import torchaudio.functional as th_audio_f

import torch_scatter

import numpy as np

from typing import Tuple


def diff(x: th.Tensor) -> th.Tensor:
    """

    :param x:
    :type x:
    :return:
    :rtype:
    """
    return th_f.pad(x[:, 1:] - x[:, :-1], (1, 0, 0, 0), "constant", 0)


def unwrap(phi: th.Tensor) -> th.Tensor:
    """

    :param phi:
    :type phi:
    :return:
    :rtype:
    """
    dphi = diff(phi)
    dphi_m = ((dphi + np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m == -np.pi) & (dphi > 0)] = np.pi
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < np.pi] = 0
    return phi + phi_adj.cumsum(1)


def get_bark_buckets(
    nfft: int = 4096,
    required_length: int = 512
) -> th.Tensor:
    n_bins = nfft // 2

    min_hz = 0.
    max_hz = 44100 // 2

    min_bark = 6 * th.arcsinh(th.tensor(min_hz) / 600)
    max_bark = 6 * th.arcsinh(th.tensor(max_hz) / 600)

    bucket_boundaries = 600 * th.sinh(th.linspace(min_bark, max_bark, required_length) / 6)

    frequencies = th.linspace(min_hz, max_hz, n_bins)

    buckets = th.bucketize(frequencies, bucket_boundaries)

    return buckets


def bark_compress(
        magn: th.Tensor,
        phase: th.Tensor,
        nfft: int = 4096,
        required_length: int = 512
) -> Tuple[th.Tensor, th.Tensor]:

    buckets = get_bark_buckets(nfft, required_length)

    buckets_tmp = buckets[:, None].repeat(1, magn.size()[1])

    magn = torch_scatter.scatter_mean(magn, buckets_tmp, dim=0)
    phase = torch_scatter.scatter_mean(phase, buckets_tmp, dim=0)

    return magn, phase


def wav_to_stft(
        wav_p: str,
        nperseg: int = 1024,
        stride: int = 256,
) -> th.Tensor:
    raw_audio, _ = th_audio.load(wav_p)

    raw_audio_mono = raw_audio.mean(0)

    hann_window = th.hann_window(nperseg)

    complex_values = th_audio_f.spectrogram(
        raw_audio_mono,
        pad=0, window=hann_window,
        n_fft=nperseg, hop_length=stride, win_length=nperseg,
        power=None, normalized=True,
        return_complex=True
    )

    # remove Nyquist frequency
    return complex_values[:-1, :]


def stft_to_phase_magn(
        complex_values: th.Tensor,
        nb_vec: int = 512
) -> Tuple[th.Tensor, th.Tensor]:
    magn = th.abs(complex_values)
    phase = th.angle(complex_values)

    magn = th.log(magn + 1)

    phase = unwrap(phase)

    phase = phase[:, 1:] - phase[:, :-1]
    magn = magn[:, 1:]

    max_magn = magn.max()
    min_magn = magn.min()
    max_phase = phase.max()
    min_phase = phase.min()

    magn = (magn - min_magn) / (max_magn - min_magn)
    phase = (phase - min_phase) / (max_phase - min_phase)

    magn, phase = magn * 2. - 1., phase * 2. - 1.

    #magn, phase = bark_compress(magn, phase, 4096, 512)

    magn = magn[:, magn.size()[1] % nb_vec:]
    phase = phase[:, phase.size()[1] % nb_vec:]
    magn = th.stack(magn.split(nb_vec, dim=1), dim=0)
    phase = th.stack(phase.split(nb_vec, dim=1), dim=0)

    return magn, phase


def magn_phase_to_wav(magn_phase: th.Tensor, wav_path: str, sample_rate: int):

    nfft = 1024
    n_bins = nfft // 2
    stride = 256

    """bark_magn = magn_phase[0, 0, :, :]
    bark_phase = magn_phase[0, 1, :, :]

    magn = th.zeros(n_bins, bark_magn.size()[1])
    phase = th.zeros(n_bins, bark_phase.size()[1])

    buckets = get_bark_buckets(
        nfft,
        bark_magn.size()[0]
    )

    for i, b in enumerate(buckets):
        magn[i, :] = bark_magn[b, :]
        phase[i, :] = bark_phase[b, :]"""

    magn = magn_phase[0, 0, :, :]
    phase = magn_phase[0, 1, :, :]

    phase = (phase + 1.) / 2. * 2. * np.pi - np.pi
    for i in range(phase.size()[1] - 1):
        phase[:, i + 1] = phase[:, i + 1] + phase[:, i]

    phase = phase % (2 * np.pi)

    magn = (magn + 1.) / 2.
    magn = th.exp(magn) - 1

    real = magn * th.cos(phase)
    imag = magn * th.sin(phase)

    real = th.cat([real, th.zeros(1, real.size()[1])], dim=0)
    imag = th.cat([imag, th.zeros(1, imag.size()[1])], dim=0)

    z = real + imag * 1j

    hann_window = th.hann_window(nfft)

    raw_audio = th_audio_f.inverse_spectrogram(
        z, length=None,
        pad=0, window=hann_window,
        n_fft=nfft, hop_length=stride,
        win_length=nfft, normalized=True
    )

    th_audio.save(wav_path, raw_audio[None, :], sample_rate)
