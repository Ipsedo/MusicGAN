import matplotlib.pyplot as plt

from . import audio


def view_audio(
        audio_path: str,
        image_idx: int
) -> None:

    c_a = audio.wav_to_wavelets(audio_path)
    c_a = audio.prepare_wavelets(c_a)

    c_a = c_a[image_idx].numpy()

    fig, ax = plt.subplots()
    fig.suptitle("approximation coefficients")
    ax.matshow(c_a / (c_a.max() - c_a.min()), cmap='plasma')
    fig.show()
