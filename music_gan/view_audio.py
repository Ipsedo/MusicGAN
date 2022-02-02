import matplotlib.pyplot as plt

from . import audio


def view_audio(
        audio_path: str,
        image_idx: int
) -> None:

    cplx_values = audio.wav_to_stft(audio_path)

    magn, phase = audio.stft_to_phase_magn(cplx_values)

    magn = magn[image_idx].numpy()
    phase = phase[image_idx].numpy()

    fig, ax = plt.subplots()
    fig.suptitle("magnitude")
    ax.matshow(magn / (magn.max() - magn.min()), cmap='plasma')
    fig.show()

    fig, ax = plt.subplots()
    fig.suptitle("phase")
    ax.matshow(phase / (phase.max() - phase.min()), cmap='plasma')
    fig.show()
