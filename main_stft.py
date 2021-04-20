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
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser("MusicGAN")

    parser.add_argument(
        "run",
        type=str,
        metavar="RUN_NAME"
    )

    parser.add_argument(
        "-o", "--out-path",
        dest="out_path",
        type=str,
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
        args.input_musics
    )

    if len(wavs_path) == 0:
        raise Exception(
            "Empty train sounds."
        )

    mlflow.start_run(run_name=args.run)

    mlflow.log_param("input_musics", wavs_path)

    sample_rate = 44100

    rand_channel = 8
    rand_width = 1
    rand_height = 2

    gen_hidden_channel = 48
    gen_residual_channel = 32

    disc_hidden_channel = 64

    disc_lr = 1e-5
    gen_lr = 1e-5

    nb_epoch = 100
    batch_size = 4

    output_dir = args.out_path

    if not exists(output_dir):
        mkdir(output_dir)
    elif exists(output_dir) and not isdir(output_dir):
        raise Exception(
            f"\"{output_dir}\" is not a directory !"
        )

    gen = networks.STFTGenerator(
        rand_channel, gen_residual_channel, gen_hidden_channel, 2
    )

    disc = networks.STFTDiscriminator(
        2, disc_hidden_channel
    )

    gen.cuda()
    disc.cuda()

    optim_gen = th.optim.Adam(
        gen.parameters(), lr=gen_lr
    )

    optim_disc = th.optim.Adam(
        disc.parameters(), lr=disc_lr
    )

    data = read_audio.to_tensor_stft(wavs_path, sample_rate)

    nb_batch = data.size()[0] // batch_size

    mean_vec = th.randn(rand_channel)
    rand_mat = th.randn(rand_channel, rand_channel)
    cov_mat = rand_mat.t().matmul(rand_mat)
    multi_norm = th.distributions.MultivariateNormal(mean_vec, cov_mat)

    mlflow.log_params({
        "rand_channel": rand_channel,
        "gen_hidden_channel": gen_hidden_channel,
        "gen_residual_channel": gen_residual_channel,
        "disc_hidden_channel": disc_hidden_channel,
        "nb_epoch": nb_epoch,
        "batch_size": batch_size,
        "disc_lr": disc_lr,
        "gen_lr": gen_lr,
        "sample_rate": sample_rate
    })

    mlflow.log_param("cov_mat", cov_mat.tolist())
    mlflow.log_param("mean_vec", mean_vec.tolist())

    with mlflow.start_run(run_name="train", nested=True):

        metric_window = 100
        error_tp = [1. for _ in range(metric_window)]
        error_tn = [1. for _ in range(metric_window)]

        disc_loss_sum = [2. for _ in range(metric_window)]
        gen_loss_sum = [1. for _ in range(metric_window)]

        for e in range(nb_epoch):
            # shuffle tensor
            batch_idx_list = list(range(nb_batch))
            random.shuffle(batch_idx_list)
            tqdm_bar = tqdm(batch_idx_list)

            for b_idx in tqdm_bar:
                i_min = b_idx * batch_size
                i_max = (b_idx + 1) * batch_size

                # train discriminator

                disc.train()
                gen.eval()

                x_real = data[i_min:i_max, :, :, :].cuda()

                rand_fake = multi_norm.sample(
                    (i_max - i_min, rand_width, rand_height)) \
                    .permute(0, 3, 1, 2) \
                    .cuda()

                x_fake = gen(rand_fake)

                out_real = disc(x_real)
                out_fake = disc(x_fake)

                disc_loss = networks.discriminator_loss(
                    out_real, out_fake
                )

                optim_gen.zero_grad()
                optim_disc.zero_grad()

                disc_loss.backward()
                optim_disc.step()

                # discriminator metrics
                error_tp.append((1. - out_real).mean().item())
                error_tn.append(out_fake.mean().item())

                del error_tn[0]
                del error_tp[0]

                # train generator

                gen.train()
                disc.eval()

                rand_fake = multi_norm.sample(
                    (i_max - i_min, rand_width, rand_height)) \
                    .permute(0, 3, 1, 2) \
                    .cuda()

                x_fake = gen(rand_fake)
                out_fake = disc(x_fake)

                gen_loss = networks.generator_loss(out_fake)

                optim_gen.zero_grad()
                optim_disc.zero_grad()

                gen_loss.backward()
                optim_gen.step()

                # metrics

                gen_grad_norm = th.tensor(
                    [p.grad.norm() for p in gen.parameters()]
                ).mean()

                disc_grad_norm = th.tensor(
                    [p.grad.norm() for p in disc.parameters()]
                ).mean()

                del disc_loss_sum[0]
                del gen_loss_sum[0]

                disc_loss_sum.append(disc_loss.item())
                gen_loss_sum.append(gen_loss.item())

                tqdm_bar.set_description(
                    f"Epoch {e}, "
                    f"disc_loss = {mean(disc_loss_sum):.6f}, "
                    f"gen_loss = {mean(gen_loss_sum):.6f}, "
                    f"e_tp = {mean(error_tp):.4f}, "
                    f"e_tn = {mean(error_tn):.4f}, "
                    f"gen_gr = {gen_grad_norm.item():.4f}, "
                    f"disc_gr = {disc_grad_norm.item():.4f}"
                )

                # log metrics
                if b_idx % 500 == 0:
                    mlflow.log_metrics({
                        "disc_loss": disc_loss.item(),
                        "gen_loss": gen_loss.item(),
                        "batch_tp_error": error_tp[-1],
                        "batch_tn_error": error_tn[-1],
                        "disc_grad_norm_mean": disc_grad_norm.item(),
                        "gen_grad_norm_mean": gen_grad_norm.item()
                    },
                        step=e * nb_batch + b_idx)

            # Generate sound
            with th.no_grad():

                # 10 seconds
                rand_fake = multi_norm.sample(
                    (1, rand_width * 10, rand_height)) \
                    .permute(0, 3, 1, 2) \
                    .cuda()

                x_fake = gen(rand_fake)

                read_audio.stft_to_wav(
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
