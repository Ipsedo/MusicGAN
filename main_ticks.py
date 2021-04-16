import networks
import read_audio

import torch as th

import glob
from os import mkdir
from os.path import join, exists, isdir

import mlflow

import argparse

from tqdm import tqdm

import random

import sys


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

    sample_rate = 16000
    rand_channel = 32
    sample_length = 16000
    rand_length = 125
    out_channel = 1
    gen_hidden_channel = 16
    disc_hidden_channel = 9

    output_dir = args.out_path

    if not exists(output_dir):
        mkdir(output_dir)
    elif exists(output_dir) and not isdir(output_dir):
        raise Exception(f"\"{output_dir}\" is not a directory !")

    disc_lr = 1e-4
    gen_lr = 1e-4

    data = read_audio.to_tensor_ticks(wavs_path, sample_rate, out_channel, sample_length)

    gen = networks.Generator(rand_channel, gen_hidden_channel, out_channel)
    disc = networks.Discriminator(out_channel, disc_hidden_channel)

    gen.cuda()
    disc.cuda()

    optim_disc = th.optim.SGD(disc.parameters(), lr=disc_lr)
    optim_gen = th.optim.SGD(gen.parameters(), lr=gen_lr)

    def __shuffle() -> None:
        for i in tqdm(range(data.size(0) - 1)):
            j = i + random.randint(0, sys.maxsize) // (
                    sys.maxsize // (data.size(0) - i) + 1
            )
            data[i, :, :], data[j, :, :] = \
                data[j, :, :], data[i, :, :]

    nb_epoch = 10
    batch_size = 4

    nb_batch = data.size(0) // batch_size

    mean_vec = th.randn(rand_channel)
    rand_mat = th.randn(rand_channel, rand_channel)
    cov_mat = rand_mat.t().matmul(rand_mat)
    multi_norm = th.distributions.MultivariateNormal(mean_vec, cov_mat)

    mlflow.log_params({
        "rand_channel": rand_channel,
        "out_channel": out_channel,
        "gen_hidden_channel": gen_hidden_channel,
        "nb_epoch": nb_epoch,
        "batch_size": batch_size,
        "disc_lr": disc_lr,
        "gen_lr": gen_lr,
        "sample_rate": sample_rate,
        "sample_length": sample_length,
        "rand_length": rand_length
    })

    mlflow.log_param("cov_mat", cov_mat.tolist())
    mlflow.log_param("mean_vec", mean_vec.tolist())

    with mlflow.start_run(run_name="train", nested=True):
        for e in range(nb_epoch):

            __shuffle()

            tqdm_bar = tqdm(range(nb_batch))

            error_tp = 0.
            error_tn = 0.

            disc_loss_sum = 0.
            gen_loss_sum = 0.

            for b_idx in tqdm_bar:
                i_min = b_idx * batch_size
                i_max = (b_idx + 1) * batch_size

                # train discirminator
                disc.train()
                gen.eval()

                x_real = data[i_min:i_max].cuda()

                rand_fake = multi_norm.sample((i_max - i_min, rand_length))\
                    .permute(0, 2, 1)\
                    .cuda()

                x_fake = gen(rand_fake)

                out_real = disc(x_real)
                out_fake = disc(x_fake)

                error_tp += (1. - out_real).mean().item()
                error_tn += out_fake.mean().item()

                disc_loss = networks.discriminator_loss(
                    out_real, out_fake
                )

                optim_gen.zero_grad()
                optim_disc.zero_grad()

                disc_loss.backward()
                optim_disc.step()

                disc_loss_sum += disc_loss.item()

                # train generator
                gen.train()
                disc.eval()

                rand_fake = multi_norm.sample((i_max - i_min, rand_length)) \
                    .permute(0, 2, 1) \
                    .cuda()

                x_fake = gen(rand_fake)
                out_fake = disc(x_fake)

                gen_loss = networks.generator_loss(out_fake)

                optim_gen.zero_grad()
                optim_disc.zero_grad()

                gen_loss.backward()
                optim_gen.step()

                gen_loss_sum += gen_loss.item()

                # metrics
                gen_grad_norm = th.tensor(
                    [p.grad.norm() for p in gen.parameters()]
                ).mean()

                disc_grad_norm = th.tensor(
                    [p.grad.norm() for p in disc.parameters()]
                ).mean()

                tqdm_bar.set_description(
                    f"Epoch {e}, "
                    f"disc_loss = {disc_loss_sum / (b_idx + 1):.6f}, "
                    f"gen_loss = {gen_loss_sum / (b_idx + 1):.6f}, "
                    f"e_tp = {error_tp / (b_idx + 1):.4f}, "
                    f"e_tn = {error_tn / (b_idx + 1):.4f}, "
                    f"gen_gr = {gen_grad_norm.item():.4f}, "
                    f"disc_gr = {disc_grad_norm.item():.4f}"
                )

                if b_idx % 500 == 0:
                    mlflow.log_metrics(
                        {
                            "disc_loss": disc_loss.item(),
                            "gen_loss": gen_loss.item(),
                            "batch_tp_error": (1. - out_real).mean().item(),
                            "batch_tn_error": out_fake.mean().item(),
                            "disc_grad_norm_mean": disc_grad_norm.item(),
                            "gen_grad_norm_mean": gen_grad_norm.item()
                        },
                        step=e * nb_batch + b_idx)

            with th.no_grad():

                # 10 seconds
                rand_fake = multi_norm.sample((1, rand_length * 10)) \
                    .permute(0, 2, 1) \
                    .cuda()

                x_fake = gen(rand_fake)
                read_audio.ticks_to_wav(
                    x_fake.detach().cpu(),
                    join(output_dir, f"gen_epoch_{e}.wav"),
                    sample_rate
                )

            # Save discriminator
            th.save(
                disc.state_dict(),
                join(output_dir, f"disc_epoch_{e}.pt")
            )
            th.save(
                optim_disc.state_dict(),
                join(output_dir, f"optim_disc_epoch_{e}.pt")
            )

            # save generator
            th.save(
                gen.state_dict(),
                join(output_dir, f"gen_epoch_{e}.pt")
            )
            th.save(
                optim_gen.state_dict(),
                join(output_dir, f"optim_gen_epoch_{e}.pt")
            )

            # log models & optim to mlflow
            mlflow.log_artifact(
                join(output_dir, f"gen_epoch_{e}.pt")
            )
            mlflow.log_artifact(
                join(output_dir, f"optim_gen_epoch_{e}.pt")
            )
            mlflow.log_artifact(
                join(output_dir, f"disc_epoch_{e}.pt")
            )
            mlflow.log_artifact(
                join(output_dir, f"optim_disc_epoch_{e}.pt")
            )

            # log gen sound
            mlflow.log_artifact(
                join(output_dir, f"gen_epoch_{e}.wav")
            )


if __name__ == '__main__':
    main()
