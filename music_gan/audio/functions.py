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

    min_hz = 20.0
    max_hz = 44100 // 2

    linspace: th.Tensor = th.linspace(min_hz, max_hz, magn.size()[0]) / 600.0
    scale = 6.0 * th.arcsinh(linspace)[:, None]
    scale_norm = scale / scale.norm()

    res: th.Tensor = magn / scale_norm if unscale else magn * scale_norm
    return res


def simpson(
    first_primitive: th.Tensor,
    derivative: th.Tensor,
    dim: int,
    dx: float,
) -> th.Tensor:
    sizes = derivative.size()
    n = derivative.size()[dim]

    evens = th.arange(0, n, 2)
    odds = th.arange(1, n, 2)

    even_derivative = th.index_select(derivative, dim, evens)
    odd_derivative = th.index_select(derivative, dim, odds)

    shift_odd_derivative = th_f.pad(
        odd_derivative,
        [
            p
            for d in reversed(range(len(sizes)))
            for p in [1 if d == dim else 0, 0]
        ],
        "constant",
        0,
    )

    even_primitive = first_primitive + dx / 3 * (
        (
            2 * even_derivative
            + 4
            * th.index_select(
                shift_odd_derivative,
                dim=dim,
                index=th.arange(0, even_derivative.size()[dim]),
            )
        ).cumsum(dim)
        - th.select(even_derivative, dim, 0).unsqueeze(dim)
        - th.select(even_derivative, dim, 0).unsqueeze(dim)
    )

    odd_primitive = (dx / 3) * (
        (
            2 * odd_derivative
            + 4
            * th.index_select(
                even_derivative,
                dim=dim,
                index=th.arange(0, odd_derivative.size()[dim]),
            )
        ).cumsum(dim)
        - 4 * th.select(even_derivative, dim, 0).unsqueeze(dim)
        - th.select(odd_derivative, dim, 0).unsqueeze(dim)
        - odd_derivative
    )

    odd_primitive += first_primitive + dx / 12 * (
        5 * th.select(derivative, dim, 0)
        + 8 * th.select(derivative, dim, 1)
        - th.select(derivative, dim, 2)
    ).unsqueeze(dim)

    primitive = th.zeros_like(derivative)

    view = [-1 if i == dim else 1 for i in range(len(sizes))]
    repeat = [1 if i == dim else s for i, s in enumerate(sizes)]
    evens = evens.view(*view).repeat(*repeat)
    odds = odds.view(*view).repeat(*repeat)

    primitive.scatter_(dim, evens, even_primitive)
    primitive.scatter_(dim, odds, odd_primitive)

    return primitive


def trapezoid(
    first_primitive: th.Tensor,
    derivative: th.Tensor,
    dim: int,
    dx: float,
) -> th.Tensor:
    return first_primitive + dx * (
        derivative.cumsum(dim=dim)
        - derivative / 2.0
        - th.select(derivative, dim, 0).unsqueeze(dim) / 2.0
    )


def wav_to_stft(
    wav_p: str,
    nperseg: int = constants.N_FFT,
    stride: int = constants.STFT_STRIDE,
) -> th.Tensor:
    raw_audio, sr = th_audio.load(wav_p)

    assert sr == constants.SAMPLE_RATE, (
        f"Audio sample rate must be {constants.SAMPLE_RATE}Hz, "
        f'file "{wav_p}" is {sr}Hz'
    )

    raw_audio_mono = raw_audio.mean(0)

    hann_window = th.hann_window(nperseg)

    complex_values: th.Tensor = th_audio_f.spectrogram(
        raw_audio_mono,
        pad=0,
        window=hann_window,
        n_fft=nperseg,
        hop_length=stride,
        win_length=nperseg,
        power=None,
        normalized=True,
    )

    # remove Nyquist frequency
    return complex_values[:-1, :]


def stft_to_phase_magn(
    complex_values: th.Tensor,
    nb_vec: int = constants.N_VEC,
    epsilon: float = 1e-8,
) -> Tuple[th.Tensor, th.Tensor]:
    magn = th.abs(complex_values)
    phase = th.angle(complex_values)

    magn = bark_magn_scale(magn, unscale=False)
    magn = th_f.pad(magn, (1, 0, 0, 0), "constant", 0.0)

    phase = unwrap(phase)
    phase = th_f.pad(phase, (1, 0, 0, 0), "constant", 0.0)
    phase = th.gradient(phase, dim=1, spacing=1.0, edge_order=1)[0]

    max_magn = magn.max()
    min_magn = magn.min()
    max_phase = phase.max()
    min_phase = phase.min()

    magn = (magn - min_magn) / (max_magn - min_magn + epsilon)
    phase = (phase - min_phase) / (max_phase - min_phase + epsilon)

    magn, phase = magn * 2.0 - 1.0, phase * 2.0 - 1.0

    magn = magn[:, magn.size()[1] % nb_vec :]
    phase = phase[:, phase.size()[1] % nb_vec :]
    magn = th.stack(magn.split(nb_vec, dim=1), dim=0)
    phase = th.stack(phase.split(nb_vec, dim=1), dim=0)

    return magn, phase


def magn_phase_to_wav(
    magn_phase: th.Tensor,
    wav_path: str,
    sample_rate: int,
    n_fft: int = constants.N_FFT,
    stft_stride: int = constants.STFT_STRIDE,
    epsilon: float = 1e-8,
) -> None:
    assert (
        len(magn_phase.size()) == 4
    ), f"(N, 2, H, W), actual = {magn_phase.size()}"

    assert (
        magn_phase.size()[1] == 2
    ), f"Channels must be equal to 2, actual = {magn_phase.size()[1]}"

    assert magn_phase.size()[2] == n_fft // 2, (
        f"Frequency size must be equal to {n_fft // 2}, "
        f"actual = {magn_phase.size()[2]}"
    )

    magn_phase_flattened = magn_phase.permute(1, 2, 0, 3).flatten(2, 3)
    magn = magn_phase_flattened[0, :, :]
    phase = magn_phase_flattened[1, :, :]

    magn = (magn + 1.0) / 2.0
    magn = bark_magn_scale(magn, unscale=True)
    magn = magn / (magn.max() - magn.min() + epsilon)

    phase = (phase + 1.0) / 2.0 * 2.0 * np.pi - np.pi
    phase = simpson(th.zeros(phase.size()[0], 1), phase, 1, 1.0)
    phase = phase % (2 * np.pi)

    real = magn * th.cos(phase)
    imag = magn * th.sin(phase)

    real_res = th_f.pad(real, (0, 0, 0, 1), "constant", 0)
    imag_res = th_f.pad(imag, (0, 0, 0, 1), "constant", 0)

    z = real_res + imag_res * 1j

    hann_window = th.hann_window(n_fft)

    raw_audio = th_audio_f.inverse_spectrogram(
        z,
        length=None,
        pad=0,
        window=hann_window,
        n_fft=n_fft,
        hop_length=stft_stride,
        win_length=n_fft,
        normalized=True,
    )

    th_audio.save(wav_path, raw_audio[None, :], sample_rate)
