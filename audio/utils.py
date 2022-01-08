import torch as th
import torch.nn.functional as th_f
import torchaudio as th_audio
import torchaudio.functional as th_audio_f

import torch_scatter

import numpy as np

from typing import Tuple

from . import constant


def diff(x: th.Tensor) -> th.Tensor:
    return th_f.pad(x[:, 1:] - x[:, :-1], (1, 0, 0, 0), "constant", 0)


def unwrap(phi: th.Tensor) -> th.Tensor:
    dphi = diff(phi)
    dphi_m = ((dphi + np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m == -np.pi) & (dphi > 0)] = np.pi
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < np.pi] = 0
    return phi + phi_adj.cumsum(1)


def get_bark_buckets(
        nfft: int ,#= constant.N_FFT,
        required_length: int #= constant.BARK_SIZE
) -> th.Tensor:
    n_bins = nfft // 2

    min_hz = 0.
    max_hz = 44100 // 2

    min_bark = 6. * th.arcsinh(th.tensor(min_hz) / 600.)
    max_bark = 6. * th.arcsinh(th.tensor(max_hz) / 600.)

    bucket_boundaries = 600. * th.sinh(th.linspace(min_bark, max_bark, required_length) / 6.)

    frequencies = th.linspace(min_hz, max_hz, n_bins)

    buckets = th.bucketize(frequencies, bucket_boundaries)

    return buckets


def bark_compress(
        complex_values: th.Tensor,
        nfft: int,# = constant.N_FFT,
        required_length: int,# = constant.BARK_SIZE
) -> th.Tensor:
    buckets = get_bark_buckets(nfft, required_length)

    buckets_tmp = buckets[:, None].repeat(1, complex_values.size()[1])

    real = torch_scatter.scatter_mean(th.real(complex_values), buckets_tmp, dim=0)
    imag = torch_scatter.scatter_mean(th.imag(complex_values), buckets_tmp, dim=0)

    return real + 1j * imag


def bark_magn_scale(magn: th.Tensor, unscale: bool = False) -> th.Tensor:
    assert len(magn.size()) == 2, f"(STFT, TIME), actual = {magn.size()}"

    min_hz = 20.
    max_hz = 44100 // 2

    scale = 6. * th.arcsinh(th.linspace(min_hz, max_hz, magn.size()[0]) / 600.)[:, None]
    scale_norm = scale / scale.norm()

    return magn / scale_norm if unscale else magn * scale_norm


def wav_to_stft(
        wav_p: str,
        nperseg: int = constant.N_FFT,
        stride: int = constant.STFT_STRIDE,
) -> th.Tensor:
    raw_audio, sr = th_audio.load(wav_p)

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
        nb_vec: int = constant.N_VEC
) -> Tuple[th.Tensor, th.Tensor]:
    magn = th.abs(complex_values)
    phase = th.angle(complex_values)

    magn = bark_magn_scale(magn, unscale=False)

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

    magn = magn[:, magn.size()[1] % nb_vec:]
    phase = phase[:, phase.size()[1] % nb_vec:]
    magn = th.stack(magn.split(nb_vec, dim=1), dim=0)
    phase = th.stack(phase.split(nb_vec, dim=1), dim=0)

    return magn, phase


def magn_phase_to_wav(magn_phase: th.Tensor, wav_path: str, sample_rate: int):
    assert len(magn_phase.size()) == 4, \
        f"(N, 2, H, W), actual = {magn_phase.size()}"

    assert magn_phase.size()[1] == 2, \
        f"Channels must be equal to 2, actual = {magn_phase.size()[1]}"

    assert magn_phase.size()[2] == constant.N_FFT // 2, \
        f"Frequency size must be equal to {constant.N_FFT // 2}, actual = {magn_phase.size()[2]}"

    magn = magn_phase.permute(1, 2, 0, 3).flatten(2, 3)[0, :]
    phase = magn_phase.permute(1, 2, 0, 3).flatten(2, 3)[1, :]

    magn = (magn + 1.) / 2.
    magn = bark_magn_scale(magn, unscale=True)
    magn = magn / (magn.max() - magn.min())

    phase = (phase + 1.) / 2. * 2. * np.pi - np.pi

    for i in range(phase.size()[1] - 1):
        phase[:, i + 1] = phase[:, i + 1] + phase[:, i]

    phase = phase % (2 * np.pi)

    #magn = th.exp(magn) - 1

    real = magn * th.cos(phase)
    imag = magn * th.sin(phase)

    # real_res = th.zeros(constant.N_FFT // 2, real.size()[1])
    # imag_res = th.zeros(constant.N_FFT // 2, imag.size()[1])

    # buckets = get_bark_buckets(
    #     constant.N_FFT,
    #     constant.BARK_SIZE
    # )

    # for i, b in enumerate(buckets):
    #     real_res[i, :] = real[b, :]
    #     imag_res[i, :] = imag[b, :]

    real_res = th.cat([real, th.zeros(1, real.size()[1])], dim=0)
    imag_res = th.cat([imag, th.zeros(1, imag.size()[1])], dim=0)

    z = real_res + imag_res * 1j

    hann_window = th.hann_window(constant.N_FFT)

    raw_audio = th_audio_f.inverse_spectrogram(
        z, length=None,
        pad=0, window=hann_window,
        n_fft=constant.N_FFT, hop_length=constant.STFT_STRIDE,
        win_length=constant.N_FFT, normalized=True
    )

    th_audio.save(wav_path, raw_audio[None, :], sample_rate)
