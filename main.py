import networks
import read_audio
import utils

import torch as th
from torch.distributions.multivariate_normal import MultivariateNormal

import sys

import random
import glob
import math

from tqdm import tqdm


def main() -> None:
    wavs_path = glob.glob(
        "/home/samuel/Documents/MusicAutoEncoder/res/rammstein/*.wav")

    data = read_audio.to_tensor(wavs_path, utils.N_FFT, utils.N_SEC)

    hidden_channel = 8
    fft_vec_size = utils.N_FFT // 2
    batch_vec_nb = utils.N_SEC * utils.SAMPLE_RATE // fft_vec_size

    gen = networks.Generator(hidden_channel)
    disc = networks.Discriminator(2)
    disc.cuda()
    gen.cuda()

    nb_epoch = 30
    batch_size = 8

    for i in tqdm(range(data.size(0) - 1)):
        j = i + random.randint(0, sys.maxsize) // (
                sys.maxsize // (data.size(0) - i) + 1)
        data[i, :, :, :], data[j, :, :, :] = data[j, :, :, :], data[i, :, :, :]

    nb_batch = math.ceil(data.size(0) / batch_size)

    disc_optimizer = th.optim.Adam(disc.parameters(), lr=5e-5)
    gen_optimizer = th.optim.Adam(gen.parameters(), lr=1e-4)

    # hidden distribution
    hidden_dist = MultivariateNormal(
        th.zeros(hidden_channel),
        th.eye(hidden_channel))

    # vector distribution
    vec_dist = MultivariateNormal(
        th.zeros(fft_vec_size),
        th.eye(fft_vec_size))

    # time distribution
    time_dist = MultivariateNormal(
        th.zeros(batch_vec_nb),
        th.eye(batch_vec_nb))

    def _gen_rand(curr_batch_size: int) -> th.Tensor:
        cpx_vec = hidden_dist.sample(
            (curr_batch_size, batch_vec_nb, fft_vec_size)).cuda()

        vec_proj = vec_dist.sample(
            (curr_batch_size, batch_vec_nb)
        ).unsqueeze(-1).cuda()

        time_proj = time_dist.sample(
            (curr_batch_size,)
        ).unsqueeze(-1).unsqueeze(-1).cuda()

        return (time_proj * vec_proj * cpx_vec).permute(0, 3, 1, 2)

    for e in range(nb_epoch):
        disc_loss_sum = 0.
        gen_loss_sum = 0.

        tqdm_bar = tqdm(range(nb_batch))

        for b_idx in tqdm_bar:
            i_min = b_idx * batch_size
            i_max = (b_idx + 1) * batch_size
            i_max = min(data.size(0), i_max)

            x_real = data[i_min:i_max, :, :, :].cuda()

            # Train discriminator
            h_fake = _gen_rand(i_max - i_min)

            x_fake = gen(h_fake)
            out_real = disc(x_real)
            out_fake = disc(x_fake)

            disc_loss = networks.discriminator_loss(out_real, out_fake)

            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            disc_loss.backward()
            disc_optimizer.step()

            disc_loss_sum += disc_loss.item()

            # Train generator
            h_fake = _gen_rand(i_max - i_min)

            x_fake = gen(h_fake)
            out_fake = disc(x_fake)

            gen_loss = networks.generator_loss(out_fake)

            disc_optimizer.zero_grad()
            gen_optimizer.zero_grad()

            gen_loss.backward()
            gen_optimizer.step()

            gen_loss_sum += gen_loss.item()

            tqdm_bar.set_description(
                f"Epoch {e} : disc_loss = {disc_loss_sum / (b_idx + 1):.6f}, "
                f"gen_loss = {gen_loss_sum / (b_idx + 1):.6f}")

        read_audio.to_wav(
            gen(hidden_dist.sample(
                (batch_size, utils.N_FFT, utils.N_FFT))
                .permute(0, 3, 1, 2).cuda()).detach().cpu(),
            f"out_train_epoch_{e}.wav")


if __name__ == '__main__':
    main()
