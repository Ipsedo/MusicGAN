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
        nperseg: int = constant.N_WAVELETS
) -> th.Tensor:
    raw_audio, sr = th_audio.load(wav_p)

    assert sr == constant.SAMPLE_RATE, \
        f"Audio sample rate must be {constant.SAMPLE_RATE}Hz, " \
        f"file \"{wav_p}\" is {sr}Hz"

    raw_audio_mono = raw_audio.mean(0)
    epsilon = 1e-8
    raw_audio_mono = raw_audio_mono / (raw_audio_mono.max() - raw_audio.min() + epsilon)

    """raw_audio_mono = raw_audio_mono[raw_audio_mono.size()[0] % nperseg:]
    raw_audio_mono = th.stack(raw_audio_mono.split(nperseg, dim=0), dim=0)

    wavelet = pywt.Wavelet("db1")

    c_a, _ = pywt.dwt(
        raw_audio_mono.numpy(), wavelet, mode="zero", axis=-1
    )

    return th.tensor(c_a).permute(1, 0)"""

    wavelets, _ = pywt.cwt(
        raw_audio_mono.numpy(),
        list(range(1, nperseg + 1)), "morl",
        1. / constant.SAMPLE_RATE,
        method="conv"
    )

    return th.tensor(wavelets)


def prepare_wavelets(c_a: th.Tensor, nb_vec: int = constant.N_VEC) -> th.Tensor:

    #wavelet = pywt.Wavelet("db1")
    #wv_max = wavelet.dec_hi[1]

    # to range [-1; 1]
    #c_a = c_a / wv_max

    c_a = c_a[:, c_a.size()[1] % nb_vec:]
    c_a = th.stack(c_a.split(nb_vec, dim=1), dim=0)
    c_a = c_a.unsqueeze(1)

    return c_a


def wavelets_to_wav(c_a: th.Tensor, wav_path: str, sample_rate: int) -> None:
    assert len(c_a.size()) == 4, \
        f"(N, 2, H, W), actual = {c_a.size()}"

    assert c_a.size()[1] == 1, \
        f"Channels must be equal to 1, actual = {c_a.size()[1]}"

    assert c_a.size()[2] == constant.N_WAVELETS, \
        f"Frequency size must be equal to {constant.N_WAVELETS}, " \
        f"actual = {c_a.size()[2]}"

    # (N, C, F, T) -> (F, T)
    c_a = c_a.permute(1, 2, 0, 3).flatten(2, 3)[0, :].numpy()

    """wavelet = pywt.Wavelet("db1")

    c_a = c_a * wavelet.dec_hi[1]

    c_a = c_a.permute(1, 0).numpy()

    raw_audio = pywt.idwt(c_a, np.zeros(c_a.shape, dtype=c_a.dtype), "db1", mode="zero", axis=-1)
    raw_audio = raw_audio.reshape(-1)"""

    mwf = pywt.ContinuousWavelet('morl', dtype=np.float32).wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]

    r_sum = np.transpose(np.sum(np.transpose(c_a) / np.arange(1, c_a.shape[0] + 1) ** 0.5, axis=-1))
    raw_audio = r_sum * (1. / y_0)

    th_audio.save(wav_path, th.from_numpy(raw_audio.astype(np.float32))[None, :], sample_rate)
