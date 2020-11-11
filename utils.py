from typing import NamedTuple

import torch as th

SAMPLE_RATE: int = 44100
N_FFT: int = 64
N_SEC: float = 4 * N_FFT ** 2 / SAMPLE_RATE

WavInfo = NamedTuple(
    "WavInfo",
    [
        ("wav_path", str),
        ("sampling_rate", int),
        ("data", th.Tensor)
    ]
)


FFTAudio = NamedTuple(
    "FFTAudio",
    [
        ("complex_data", th.Tensor),
        ("n_fft", int)
    ]
)
