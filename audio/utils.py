import torch as th
import torch.nn.functional as th_f
import torchaudio as th_audio
import torchaudio.functional as th_audio_f

import numpy as np

import scipy.signal
import scipy.io.wavfile

from typing import Tuple


def diff(x: th.Tensor) -> th.Tensor:
    """

    :param x:
    :type x:
    :return:
    :rtype:
    """
    return th_f.pad(x[1:, :] - x[:-1, :], (0, 0, 1, 0), "constant", 0)


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
    return phi + phi_adj.cumsum(0)


def wav_to_stft(
        wav_p: str,
        nperseg: int = 1022,
        stride: int = 256,
) -> th.Tensor:
    raw_audio, _ = th_audio.load(wav_p)

    raw_audio_mono = raw_audio.mean(0)

    hann_window = th.hann_window(nperseg)

    complex_values = th_audio_f.spectrogram(
        raw_audio_mono,
        pad=0, window=hann_window,
        n_fft=nperseg, hop_length=stride, win_length=nperseg,
        power=None, normalized=True
    )

    complex_values = complex_values.permute(1, 0, 2)

    return complex_values


def stft_to_phase_magn(
        complex_values: th.Tensor,
        nb_vec: int = 512
) -> Tuple[th.Tensor, th.Tensor]:
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

    return magn * 1.6 - 0.8, phase * 1.6 - 0.8


def magn_phase_to_wav(magn_phase: th.Tensor, wav_path: str, sample_rate: int):
    magn_phase = magn_phase.permute(0, 2, 3, 1)
    x = magn_phase.flatten(0, 1)

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