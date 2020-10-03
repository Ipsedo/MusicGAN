import networks
import read_audio
import utils

import torch as th

import sys

import random
import glob
import math

from tqdm import tqdm


def main() -> None:
    wavs_path = glob.glob(
        "/home/samuel/Documents/MusicAutoEncoder/res/rammstein/*.wav")

    data = read_audio.to_tensor(wavs_path, utils.N_FFT, utils.N_SEC)

    hidden_channel = 16

    gen = networks.Generator(hidden_channel)
    disc = networks.Discriminator(2)
    disc.cuda()
    gen.cuda()

    nb_epoch = 10
    batch_size = 16

    for i in tqdm(range(data.size(0) - 1)):
        j = i + random.randint(0, sys.maxsize) // (
                sys.maxsize // (data.size(0) - i) + 1)
        data[i, :, :, :], data[j, :, :, :] = data[j, :, :, :], data[i, :, :, :]

    nb_batch = math.ceil(data.size(0) / batch_size)

    disc_optimizer = th.optim.Adam(disc.parameters(), lr=0.)
    gen_optimizer = th.optim.Adam(gen.parameters(), lr=5e-4)

    for e in range(nb_epoch):
        disc_loss_sum = 0.
        gen_loss_sum = 0.

        tqdm_bar = tqdm(range(nb_batch))

        for b_idx in tqdm_bar:
            i_min = b_idx * batch_size
            i_max = (b_idx + 1) * batch_size
            i_max = min(data.size(0), i_max)

            x_real = data[i_min:i_max, :, :, :].cuda()

            # Train generator
            h_fake = th.randn(
                i_max - i_min, hidden_channel,
                utils.N_FFT, utils.N_FFT).cuda()
            x_fake = gen(h_fake)
            out_fake = disc(x_fake)

            gen_loss = networks.generator_loss(out_fake)

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            gen_loss_sum += gen_loss.item()

            # Train discriminator
            h_fake = th.randn(
                i_max - i_min, hidden_channel,
                utils.N_FFT, utils.N_FFT).cuda()

            x_fake = gen(h_fake)
            out_real = disc(x_real)
            out_fake = disc(x_fake)

            disc_loss = networks.discriminator_loss(out_real, out_fake)

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            disc_loss_sum += disc_loss.item()

            tqdm_bar.set_description(
                f"Epoch {e} : disc_loss = {disc_loss_sum / (b_idx + 1):.6f}, "
                f"gen_loss = {gen_loss_sum / (b_idx + 1):.6f}")


if __name__ == '__main__':
    main()
