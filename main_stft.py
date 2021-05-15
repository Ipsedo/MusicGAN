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

    rand_channel = 48
    rand_width = 2
    rand_height = 4

    disc_lr = 1e-6
    gen_lr = 1e-6

    nb_epoch = 1000
    batch_size = 4

    output_dir = args.out_path

    if not exists(output_dir):
        mkdir(output_dir)
    elif exists(output_dir) and not isdir(output_dir):
        raise Exception(
            f"\"{output_dir}\" is not a directory !"
        )

    gen = networks.STFTGenerator(
        rand_channel, 2
    )

    disc = networks.STFTDiscriminator(2)

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

    mean_vec = th.zeros(rand_channel)  # th.randn(rand_channel)
    rand_mat = th.randn(rand_channel, rand_channel)
    cov_mat = th.eye(rand_channel,
                     rand_channel)  # rand_mat.t().matmul(rand_mat)
    multi_norm = th.distributions.MultivariateNormal(mean_vec, cov_mat)

    mlflow.log_params({
        "rand_channel": rand_channel,
        "nb_epoch": nb_epoch,
        "batch_size": batch_size,
        "disc_lr": disc_lr,
        "gen_lr": gen_lr,
        "sample_rate": sample_rate
    })

    mlflow.log_param("cov_mat", cov_mat.tolist())
    mlflow.log_param("mean_vec", mean_vec.tolist())

    with mlflow.start_run(run_name="train", nested=True):

        metric_window = 20
        error_tp = [1. for _ in range(metric_window)]
        error_tn = [1. for _ in range(metric_window)]

        disc_loss_sum = [2. for _ in range(metric_window)]
        gen_loss_sum = [1. for _ in range(metric_window)]

        for e in range(nb_epoch):

            # shuffle tensor
            batch_idx_list = list(range(nb_batch))
            random.shuffle(batch_idx_list)
            tqdm_bar = tqdm(batch_idx_list)

            iter_idx = 0

            for b_idx in tqdm_bar:
                i_min = b_idx * batch_size
                i_max = (b_idx + 1) * batch_size

                # train discriminator

                disc.train()
                gen.eval()

                # clip weight
                weight_limit = 1e-2
                for p in disc.parameters():
                    p.data.clamp_(-weight_limit, weight_limit)

                x_real = data[i_min:i_max, :, :, :].cuda()

                rand_fake = multi_norm.sample(
                    (i_max - i_min, rand_width, rand_height)) \
                    .permute(0, 3, 1, 2) \
                    .cuda()

                x_fake = gen(rand_fake)

                out_real = disc(x_real)
                out_fake = disc(x_fake)

                disc_loss = networks.wasserstein_discriminator_loss(
                    out_real, out_fake
                )

                optim_gen.zero_grad()
                optim_disc.zero_grad()

                disc_loss.backward()
                optim_disc.step()

                # discriminator metrics
                error_tp.append(out_real.mean().item())
                error_tn.append(out_fake.mean().item())
                del error_tn[0]
                del error_tp[0]

                disc_grad_norm = th.tensor(
                    [p.grad.norm() for p in disc.parameters()]
                ).mean()

                del disc_loss_sum[0]
                disc_loss_sum.append(disc_loss.item())

                # train generator
                if iter_idx % 10 == 0:
                    gen.train()
                    disc.eval()

                    rand_fake = multi_norm.sample(
                        (i_max - i_min, rand_width, rand_height)) \
                        .permute(0, 3, 1, 2) \
                        .cuda()

                    x_fake = gen(rand_fake)
                    out_fake = disc(x_fake)

                    gen_loss = networks.wasserstein_generator_loss(out_fake)

                    optim_gen.zero_grad()
                    optim_disc.zero_grad()

                    gen_loss.backward()
                    optim_gen.step()

                    # metrics
                    gen_grad_norm = th.tensor(
                        [p.grad.norm() for p in gen.parameters()]
                    ).mean()

                    del gen_loss_sum[0]
                    gen_loss_sum.append(gen_loss.item())

                tqdm_bar.set_description(
                    f"Epoch {e:02} - disc, "
                    f"disc_loss = {mean(disc_loss_sum):.6f}, "
                    f"gen_loss = {mean(gen_loss_sum):.6f}, "
                    f"e_tp = {mean(error_tp):.5f}, "
                    f"e_tn = {mean(error_tn):.5f}, "
                )

                # log metrics
                if iter_idx % 200 == 0:
                    mlflow.log_metrics({
                        "disc_loss": disc_loss.item(),
                        "gen_loss": gen_loss.item(),
                        "batch_tp_error": error_tp[-1],
                        "batch_tn_error": error_tn[-1],
                        "disc_grad_norm_mean": disc_grad_norm.item(),
                        "gen_grad_norm_mean": gen_grad_norm.item()
                    },
                        step=e)

                iter_idx += 1

            # Generate sound
            with th.no_grad():

                for gen_idx in range(3):
                    # 10 seconds
                    rand_fake = multi_norm.sample(
                        (1, rand_width * 10, rand_height)) \
                        .permute(0, 3, 1, 2) \
                        .cuda()

                    x_fake = gen(rand_fake)

                    read_audio.stft_to_wav(
                        x_fake.detach().cpu(),
                        join(output_dir, f"gen_epoch_{e}_ID{gen_idx}.wav"),
                        sample_rate
                    )

                    # log gen sound
                    mlflow.log_artifact(
                        join(output_dir, f"gen_epoch_{e}_ID{gen_idx}.wav")
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


if __name__ == '__main__':
    main()
