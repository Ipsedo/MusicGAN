from typing import NamedTuple

import torch as th

SAMPLE_RATE: int = 44100
N_FFT: int = 210
N_SEC: int = 1

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
