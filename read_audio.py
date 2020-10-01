from utils import WavInfo, FFTAudio

import scipy.io.wavfile
import numpy as np
import torch as th


def read_wav(wav_path: str) -> WavInfo:
    sampling_rate, data = scipy.io.wavfile.read(wav_path)

    data_th = th.from_numpy(data / np.iinfo(data.dtype).max)

    return WavInfo(
        wav_path,
        sampling_rate,
        data_th
    )


def fft(wav_info: WavInfo, n_fft: int) -> FFTAudio:
    to_keep = wav_info.data.size(0) - wav_info.data.size(0) % n_fft
    raw_audio = wav_info.data[:to_keep, :].mean(dim=-1)
    splitted_raw_audio = th.stack(raw_audio.split(n_fft, dim=0)).unsqueeze(-1)
    fft_audio = th.rfft(splitted_raw_audio, signal_ndim=1).squeeze(-2)

    return FFTAudio(
        fft_audio[:, :, 0],
        fft_audio[:, :, 1],
        n_fft
    )


if __name__ == '__main__':
    w_p = "/home/samuel/Documents/MusicAutoEncoder/res/childish.wav"
    w_i = read_wav(w_p)

    print(w_i.data.size(0) // w_i.sampling_rate / 60)

    fft_i = fft(w_i.data, 525)
    print(fft_i.real_part.size())
    print(fft_i.imag_part.size())
