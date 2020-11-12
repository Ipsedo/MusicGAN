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

import mlflow
import argparse

from os.path import join

from typing import Tuple


def main() -> None:
    parser = argparse.ArgumentParser("MusicGAN")

    parser.add_argument(
        "run",
        type=str, metavar="RUN_NAME"
    )

    parser.add_argument(
        "-o", "--out-path",
        dest="out_path", type=str,
        required=True
    )

    parser.add_argument(
        "-i", "--input-musics",
        dest="input_musics",
        required=True,
        type=str
    )

    args = parser.parse_args()

    exp_name = "MusicGAN"
    mlflow.set_experiment(exp_name)

    wavs_path = glob.glob(
        args.input_musics)

    mlflow.start_run(run_name=args.run)

    mlflow.log_param("input_musics", wavs_path)

    data = read_audio.to_tensor(
        wavs_path, utils.N_FFT, utils.N_SEC
    )

    rand_channel = 32
    hidden_channel = 256 + 64
    hidden_w = utils.N_SEC * utils.SAMPLE_RATE / utils.N_FFT

    gen = networks.Generator2(rand_channel, hidden_channel)
    disc = networks.Discriminator2(2)
    disc.cuda()
    gen.cuda()

    nb_epoch = 400
    batch_size = 8

    def __shuffle() -> None:
        for i in tqdm(range(data.size(0) - 1)):
            j = i + random.randint(0, sys.maxsize) // (
                    sys.maxsize // (data.size(0) - i) + 1
            )
            data[i, :, :, :], data[j, :, :, :] = \
                data[j, :, :, :], data[i, :, :, :]

    nb_batch = math.ceil(data.size(0) / batch_size)

    disc_lr = 3e-5
    gen_lr = 1e-5

    disc_optimizer = th.optim.Adam(disc.parameters(), lr=disc_lr)
    gen_optimizer = th.optim.Adam(gen.parameters(), lr=gen_lr)

    mean_vec = th.randn(rand_channel)
    rand_mat = th.randn(rand_channel, rand_channel)
    cov_mat = rand_mat.t().matmul(rand_mat)

    multi_norm = th.distributions.MultivariateNormal(mean_vec, cov_mat)

    mlflow.log_params({
        "rand_channel": rand_channel,
        "hidden_channel": hidden_channel,
        "hidden_w": hidden_w,
        "nb_epoch": nb_epoch,
        "batch_size": batch_size,
        "disc_lr": disc_lr,
        "gen_lr": gen_lr
    })

    mlflow.log_param("cov_mat", cov_mat.tolist())
    mlflow.log_param("mean_vec", mean_vec.tolist())

    def __gen_rand(
            curr_batch_size: int, nb_width: int
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return multi_norm.sample(
            (curr_batch_size, int(nb_width * hidden_w))), \
               th.randn(1, curr_batch_size, hidden_channel).cuda(), \
               th.randn(1, curr_batch_size, hidden_channel).cuda()

    with mlflow.start_run(run_name="train", nested=True):

        for e in range(nb_epoch):
            disc_loss_sum = 0.
            gen_loss_sum = 0.

            __shuffle()

            tqdm_bar = tqdm(range(nb_batch))

            error_tp = 0
            error_tn = 0

            disc.train()
            gen.train()

            for b_idx in tqdm_bar:
                i_min = b_idx * batch_size
                i_max = (b_idx + 1) * batch_size
                i_max = min(data.size(0), i_max)

                x_real = data[i_min:i_max, :, :, :].cuda()

                # Train discriminator
                rand_fake, h_first, c_first = __gen_rand(i_max - i_min, 1)

                x_fake = gen(
                    rand_fake.cuda(),
                    h_first.cuda(),
                    c_first.cuda()
                )

                out_real = disc(x_real)
                out_fake = disc(x_fake)

                error_tp += (1. - out_real).mean().item()
                error_tn += out_fake.mean().item()

                disc_loss = networks.discriminator_loss(out_real, out_fake)

                gen_optimizer.zero_grad()
                disc_optimizer.zero_grad()

                disc_loss.backward()
                disc_optimizer.step()

                disc_loss_sum += disc_loss.item()

                # Train generator
                rand_fake, h_first, c_first = __gen_rand(i_max - i_min, 1)

                x_fake = gen(
                    rand_fake.cuda(), h_first.cuda(), c_first.cuda()
                )

                out_fake = disc(x_fake)

                gen_loss = networks.generator_loss(out_fake)

                disc_optimizer.zero_grad()
                gen_optimizer.zero_grad()

                gen_loss.backward()
                gen_optimizer.step()

                gen_loss_sum += gen_loss.item()

                gen_grad_norm = th.tensor(
                    [p.grad.norm() for p in gen.parameters()]
                ).mean()

                disc_grad_norm = th.tensor(
                    [p.grad.norm() for p in disc.parameters()]
                ).mean()

                tqdm_bar.set_description(
                    f"Epoch {e} : "
                    f"disc_loss = {disc_loss_sum / (b_idx + 1):.6f}, "
                    f"gen_loss = {gen_loss_sum / (b_idx + 1):.6f}, "
                    f"e_tp = {error_tp / (b_idx + 1):.4f}, "
                    f"e_tn = {error_tn / (b_idx + 1):.4f}, "
                    f"gen_gr = {gen_grad_norm.item():.4f}, "
                    f"disc_gr = {disc_grad_norm.item():.4f}"
                )

                if b_idx % 500 == 0:
                    mlflow.log_metrics({
                        "disc_loss": disc_loss.item(),
                        "gen_loss": gen_loss.item(),
                        "batch_tp_error": (1. - out_real).mean().item(),
                        "batch_tn_error": out_fake.mean().item(),
                        "disc_grad_norm_mean": disc_grad_norm.item(),
                        "gen_grad_norm_mean": gen_grad_norm.item()
                    },
                        step=e * nb_batch + b_idx)

            with th.no_grad():
                gen.eval()
                rand_gen_sound, h_first, c_first = __gen_rand(1, 100)

                gen_sound = gen(
                    rand_gen_sound.cuda(), h_first.cuda(), c_first.cuda()
                ).cpu().detach()
                read_audio.to_wav(
                    gen_sound,
                    join(args.out_path, f"out_train_epoch_{e}.wav")
                )

            mlflow.log_artifact(f"./out/out_train_epoch_{e}.wav")
            mlflow.pytorch.log_model(disc, f"disc_model_epoch_{e}")
            mlflow.pytorch.log_model(gen, f"gen_model_epoch_{e}")

    mlflow.end_run()


if __name__ == '__main__':
    main()
