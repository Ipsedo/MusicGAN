import argparse

import matplotlib.pyplot as plt

import audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser("View wav")

    parser.add_argument("--input-wav", type=str, required=True)
    parser.add_argument("--sample-id", type=int, required=True)

    args = parser.parse_args()

    wav_p = args.input_wav
    idx = args.sample_id

    cplx_values = audio.wav_to_stft(wav_p)

    magn, phase = audio.stft_to_phase_magn(cplx_values)

    magn = magn[idx].numpy()
    phase = phase[idx].numpy()

    fig, ax = plt.subplots()
    ax.matshow(magn / (magn.max() - magn.min()), cmap='plasma')
    fig.show()

    fig, ax = plt.subplots()
    ax.matshow(phase / (phase.max() - phase.min()), cmap='plasma')
    fig.show()
