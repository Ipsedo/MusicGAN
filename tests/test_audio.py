from os import remove
from os.path import exists, isfile

import pytest
import torch as th

from music_gan.audio import (
    bark_scale,
    magn_phase_to_wav,
    simpson,
    stft_to_phase_magn,
    trapezoid,
    wav_to_stft,
)


@pytest.mark.parametrize("start", [0.0, 2.0, 4.0])
@pytest.mark.parametrize("end", [6.0, 8.0, 10.0])
@pytest.mark.parametrize("dx", [0.01, 0.02, 0.03])
def test_simpson(start: float, end: float, dx: float) -> None:

    steps = int((end - start) / dx)

    delta = 1e-1
    dim = 1

    derivative = th.cos(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )
    primitive = th.sin(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )

    res_simpson = simpson(
        th.select(primitive, dim, 0).unsqueeze(dim), derivative, dim, dx
    )

    assert th.all(th.abs(primitive - res_simpson).mean(dim=dim) < delta)


@pytest.mark.parametrize("start", [0.0, 2.0, 4.0])
@pytest.mark.parametrize("end", [6.0, 8.0, 10.0])
@pytest.mark.parametrize("dx", [0.01, 0.02, 0.03])
def test_trapezoid(start: float, end: float, dx: float) -> None:

    steps = int((end - start) / dx)

    delta = 1e-1
    dim = 1

    derivative = th.cos(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )
    primitive = th.sin(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )

    res_simpson = trapezoid(
        th.select(primitive, dim, 0).unsqueeze(dim), derivative, dim, dx
    )

    assert th.all(th.abs(primitive - res_simpson).mean(dim=dim) < delta)


@pytest.mark.parametrize("nperseg", [256, 512, 1024])
@pytest.mark.parametrize("stride", [64, 128, 256])
def test_wav_to_stft(wav_path: str, nperseg: int, stride: int) -> None:
    stft = wav_to_stft(wav_path, nperseg, stride)

    assert len(stft.size()) == 2
    assert stft.size()[0] == nperseg // 2
    assert th.is_complex(stft)


@pytest.mark.parametrize("nfft", [128, 256, 512])
@pytest.mark.parametrize("nb_vec", [128, 256, 512])
def test_bark_scale(nfft: int, nb_vec: int) -> None:
    delta = 1e-2

    magn = th.rand(nfft, nb_vec)

    magn_scaled = bark_scale(magn, "scale")

    assert len(magn_scaled.size()) == 2
    assert magn_scaled.size()[0] == nfft
    assert magn_scaled.size()[1] == nb_vec

    magn_unscaled = bark_scale(magn_scaled, "unscale")

    assert len(magn_unscaled.size()) == 2
    assert magn_unscaled.size()[0] == nfft
    assert magn_unscaled.size()[1] == nb_vec

    assert th.all((magn - magn_unscaled) < delta)


@pytest.mark.parametrize("nfft", [128, 256, 512])
@pytest.mark.parametrize("stft_nb", [1024, 2048, 4096])
@pytest.mark.parametrize("nb_vec", [128, 256, 512])
def test_stft_to_magn_phase(nfft: int, stft_nb: int, nb_vec: int) -> None:
    size = (nfft, stft_nb)
    stft = th.randn(*size) + th.randn(*size) * 1j
    magn, phase = stft_to_phase_magn(stft, nb_vec, epsilon=1e-8)

    assert len(magn.size()) == 3
    assert magn.size()[1] == nfft
    assert magn.size()[2] == nb_vec
    assert th.all((-1.0 <= magn) & (magn <= 1.0))

    assert len(phase.size()) == 3
    assert phase.size()[1] == nfft
    assert phase.size()[2] == nb_vec
    assert th.all((-1.0 <= phase) & (phase <= 1.0))


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("nfft", [128, 256, 512])
@pytest.mark.parametrize("nb_vec", [128, 256, 512])
@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100])
def test_magn_phase_to_wav(
    batch_size: int, nfft: int, nb_vec: int, sample_rate: int
) -> None:
    wav_path = "./tmp.wav"

    try:
        magn_phase = th.randn(batch_size, 2, nfft // 2, nb_vec)

        magn_phase_to_wav(magn_phase, wav_path, sample_rate, nfft, nfft // 2)

        assert exists(wav_path)
        assert isfile(wav_path)
    finally:
        if exists(wav_path):
            remove(wav_path)
