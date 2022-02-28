import torch as th
import torch.nn.functional as th_f
import torchaudio as th_audio
import torchaudio.functional as th_audio_f

import numpy as np

from typing import Tuple

import pywt

from . import constant


def wav_to_wavelets(
        wav_p: str,
        nperseg: int = constant.N_FFT,
        nb_vec: int = constant.N_VEC
) -> Tuple[th.Tensor, th.Tensor]:
    raw_audio, sr = th_audio.load(wav_p)

    assert sr == constant.SAMPLE_RATE, \
        f"Audio sample rate must be {constant.SAMPLE_RATE}Hz, " \
        f"file \"{wav_p}\" is {sr}Hz"

    raw_audio_mono = raw_audio.mean(0)
    raw_audio_mono = raw_audio_mono / (raw_audio_mono.max() - raw_audio.min())

    raw_audio_mono = raw_audio_mono[raw_audio_mono.size()[0] % nperseg:]
    raw_audio_mono = th.stack(raw_audio_mono.split(nperseg, dim=0), dim=0)

    c_a, c_d = pywt.dwt(
        raw_audio_mono.numpy(), "db1", mode="zero", axis=-1
    )

    c_a, c_d = th.tensor(c_a), th.tensor(c_d)
    c_a, c_d = c_a.permute(1, 0), c_d.permute(1, 0)

    # TODO regler probleme de scale
    c_a = c_a / (c_a.max() - c_a.min())
    c_d = c_d / (c_d.max() - c_d.min())

    c_a, c_d = c_a * 2., c_d * 2.

    c_a = c_a[:, c_a.size()[1] % nb_vec:]
    c_d = c_d[:, c_d.size()[1] % nb_vec:]
    c_a = th.stack(c_a.split(nb_vec, dim=1), dim=0)
    c_d = th.stack(c_d.split(nb_vec, dim=1), dim=0)

    return c_a, c_d


def wavelets_to_wav(c_a_c_d: th.Tensor, wav_path: str, sample_rate: int) -> None:
    assert len(c_a_c_d.size()) == 4, \
        f"(N, 2, H, W), actual = {c_a_c_d.size()}"

    assert c_a_c_d.size()[1] == 2, \
        f"Channels must be equal to 2, actual = {c_a_c_d.size()[1]}"

    assert c_a_c_d.size()[2] == constant.N_FFT // 2, \
        f"Frequency size must be equal to {constant.N_FFT // 2}, " \
        f"actual = {c_a_c_d.size()[2]}"

    c_a = c_a_c_d.permute(1, 2, 0, 3).flatten(2, 3)[0, :]
    c_d = c_a_c_d.permute(1, 2, 0, 3).flatten(2, 3)[1, :]

    """c_a = (c_a + 1) / 2.
    c_d = (c_d + 1) / 2."""
    c_a = c_a.permute(1, 0)
    c_d = c_d.permute(1, 0)

    raw_audio = pywt.idwt(c_a.numpy(), c_d.numpy(), "db1", mode="zero", axis=-1)
    raw_audio = raw_audio.reshape(-1)

    th_audio.save(wav_path, th.from_numpy(raw_audio)[None, :], sample_rate)
