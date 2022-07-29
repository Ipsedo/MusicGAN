from typing import List
import matplotlib.pyplot as plt
from os.path import basename, join, exists, isdir
from os import mkdir

from . import audio


def view_audio(
        audio_path: str,
        image_idx: List[int],
        output_path: str
) -> None:

    if not exists(output_path):
        mkdir(output_path)
    elif exists(output_path) and not isdir(output_path):
        raise NotADirectoryError(output_path)

    cplx_values = audio.wav_to_stft(audio_path)

    magn, phase = audio.stft_to_phase_magn(cplx_values)

    music_name = basename(audio_path)

    for idx in image_idx:
        fig, (magn_ax, phase_ax) = plt.subplots(1, 2)

        magn_frame = magn[idx, :, :].numpy()
        phase_frame = phase[idx, :, :].numpy()

        magn_ax.set_title("magnitude")
        magn_ax.matshow(magn_frame / (magn_frame.max() - magn_frame.min()), cmap='plasma')

        magn_ax.set_title("phase")
        phase_ax.matshow(phase_frame / (phase_frame.max() - phase_frame.min()), cmap='plasma')

        fig.suptitle(music_name + " " + str(idx))
        fig.savefig(join(output_path, f"{music_name}_{idx}.jpg"))
