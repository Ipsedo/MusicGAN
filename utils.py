from typing import NamedTuple

import numpy as np
import torch as th

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
        ("real_part", th.Tensor),
        ("imag_part", th.Tensor),
        ("n_fft", int)
    ]
)
