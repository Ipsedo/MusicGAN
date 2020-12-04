from typing import NamedTuple

import torch as th

SAMPLE_RATE: int = 44100
N_FFT: int = 256
N_SEC: float = 256 * N_FFT / SAMPLE_RATE

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
