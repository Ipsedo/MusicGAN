from typing import Tuple

import numpy as np
import torch as th
import torch.nn.functional as th_f
import torchaudio as th_audio
import torchaudio.functional as th_audio_f

from . import constants


def diff(x: th.Tensor) -> th.Tensor:
    return th_f.pad(x[:, 1:] - x[:, :-1], (1, 0, 0, 0), "constant", 0)


def unwrap(phi: th.Tensor) -> th.Tensor:
    dphi = diff(phi)
    dphi_m = ((dphi + np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m == -np.pi) & (dphi > 0)] = np.pi
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < np.pi] = 0
    return phi + phi_adj.cumsum(1)


def bark_magn_scale(magn: th.Tensor, unscale: bool = False) -> th.Tensor:
    assert len(magn.size()) == 2, f"(STFT, TIME), actual = {magn.size()}"

    min_hz = 20.
    max_hz = 44100 // 2

    scale = 6. * th.arcsinh(th.linspace(min_hz, max_hz, magn.size()[0]) / 600.)[:, None]
    scale_norm = scale / scale.norm()

    return magn / scale_norm if unscale else magn * scale_norm


def wav_to_stft(
        wav_p: str,
        nperseg: int = constants.N_FFT,
        stride: int = constants.STFT_STRIDE,
) -> th.Tensor:
    raw_audio, sr = th_audio.load(wav_p)

    assert sr == constants.SAMPLE_RATE, \
        f"Audio sample rate must be {constants.SAMPLE_RATE}Hz, " \
        f"file \"{wav_p}\" is {sr}Hz"

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
        nb_vec: int = constants.N_VEC,
        epsilon: float = 1e-8
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

    magn = (magn - min_magn) / (max_magn - min_magn + epsilon)
    phase = (phase - min_phase) / (max_phase - min_phase + epsilon)

    magn, phase = magn * 2. - 1., phase * 2. - 1.

    magn = magn[:, magn.size()[1] % nb_vec:]
    phase = phase[:, phase.size()[1] % nb_vec:]
    magn = th.stack(magn.split(nb_vec, dim=1), dim=0)
    phase = th.stack(phase.split(nb_vec, dim=1), dim=0)

    return magn, phase


def magn_phase_to_wav(
        magn_phase: th.Tensor,
        wav_path: str,
        sample_rate: int,
        epsilon: float = 1e-8
) -> None:
    assert len(magn_phase.size()) == 4, \
        f"(N, 2, H, W), actual = {magn_phase.size()}"

    assert magn_phase.size()[1] == 2, \
        f"Channels must be equal to 2, actual = {magn_phase.size()[1]}"

    assert magn_phase.size()[2] == constants.N_FFT // 2, \
        f"Frequency size must be equal to {constants.N_FFT // 2}, " \
        f"actual = {magn_phase.size()[2]}"

    magn = magn_phase.permute(1, 2, 0, 3).flatten(2, 3)[0, :]
    phase = magn_phase.permute(1, 2, 0, 3).flatten(2, 3)[1, :]

    magn = (magn + 1.) / 2.
    magn = bark_magn_scale(magn, unscale=True)
    magn = magn / (magn.max() - magn.min() + epsilon)

    phase = (phase + 1.) / 2. * 2. * np.pi - np.pi

    for i in range(phase.size()[1] - 1):
        phase[:, i + 1] = phase[:, i] + phase[:, i + 1]

    phase = phase % (2 * np.pi)

    real = magn * th.cos(phase)
    imag = magn * th.sin(phase)

    real_res = th.cat([real, th.zeros(1, real.size()[1])], dim=0)
    imag_res = th.cat([imag, th.zeros(1, imag.size()[1])], dim=0)

    z = real_res + imag_res * 1j

    hann_window = th.hann_window(constants.N_FFT)

    raw_audio = th_audio_f.inverse_spectrogram(
        z, length=None,
        pad=0, window=hann_window,
        n_fft=constants.N_FFT, hop_length=constants.STFT_STRIDE,
        win_length=constants.N_FFT, normalized=True
    )

    th_audio.save(wav_path, raw_audio[None, :], sample_rate)
