import audio
import networks

import torch as th
from torch.utils.data import DataLoader

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
        "-i", "--input-dataset",
        dest="input_dataset",
        required=True,
        type=str
    )

    parser.add_argument('--gen', type=str, required=False)
    parser.add_argument('--gen-optim', type=str, required=False)

    parser.add_argument('--disc', type=str, required=False)
    parser.add_argument('--disc-optim', type=str, required=False)

    args = parser.parse_args()

    exp_name = "MusicGAN"
    mlflow.set_experiment(exp_name)

    assert isdir(args.input_dataset)

    mlflow.start_run(run_name=args.run)

    mlflow.log_param("input_dataset", args.input_dataset)

    sample_rate = 44100

    rand_channel = 256

    disc_lr = 2e-4
    gen_lr = 2e-4

    nb_epoch = 1000
    batch_size = 8

    output_dir = args.out_path

    if not exists(output_dir):
        mkdir(output_dir)
    elif exists(output_dir) and not isdir(output_dir):
        raise Exception(
            f"\"{output_dir}\" is not a directory !"
        )

    gen = networks.Generator(
        rand_channel
    )

    disc = networks.Discriminator(2)

    gen.cuda()
    disc.cuda()

    betas = (0.9, 0.999)

    optim_gen = th.optim.Adam(
        gen.parameters(), lr=gen_lr,
        betas=betas
    )

    optim_disc = th.optim.Adam(
        disc.parameters(), lr=disc_lr,
        betas=betas
    )

    # Load models & optimizers
    if args.gen is not None:
        gen.load_state_dict(th.load(args.gen))

    if args.disc is not None:
        disc.load_state_dict(th.load(args.disc))

    if args.gen_optim is not None:
        optim_gen.load_state_dict(th.load(args.gen_optim))

    if args.disc_optim is not None:
        optim_disc.load_state_dict(th.load(args.disc_optim))

    # create DataSet
    audio_dataset = audio.AudioDataset(args.input_dataset)

    data_loader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True
    )

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
        "sample_rate": sample_rate,
        "adam_betas": betas
    })

    mlflow.log_param("cov_mat", cov_mat.tolist())
    mlflow.log_param("mean_vec", mean_vec.tolist())

    with mlflow.start_run(run_name="train", nested=True):

        metric_window = 20
        error_tp = [0. for _ in range(metric_window)]
        error_tn = [0. for _ in range(metric_window)]
        error_gen = [0. for _ in range(metric_window)]

        disc_loss_list = [0. for _ in range(metric_window)]
        grad_pen_list = [0. for _ in range(metric_window)]
        gen_loss_list = [0. for _ in range(metric_window)]

        for e in range(nb_epoch):

            tqdm_bar = tqdm(data_loader)

            for iter_idx, x_real in enumerate(tqdm_bar):
                # train discriminator

                # pass data to cuda
                x_real = x_real.cuda().to(th.float)

                # sample fake data
                rand_fake = multi_norm.sample(
                    (batch_size,)) \
                    .cuda()

                # gen fake data
                x_fake = gen(rand_fake)

                # pass real data and gen data to discriminator
                out_real = disc(x_real)
                out_fake = disc(x_fake)

                # compute discriminator loss
                disc_loss = networks.wasserstein_discriminator_loss(
                    out_real, out_fake
                )

                # compute gradient penalty
                grad_pen = disc.gradient_penalty(x_real, x_fake)

                # add gradient penalty
                disc_loss_gp = disc_loss + grad_pen

                # reset grad
                optim_gen.zero_grad()
                optim_disc.zero_grad()

                # backward and optim step
                disc_loss_gp.backward()
                optim_disc.step()

                # discriminator metrics
                error_tp.append(out_real.mean().item())
                error_tn.append(out_fake.mean().item())
                del error_tn[0]
                del error_tp[0]

                disc_grad_norm = th.tensor(
                    [p.grad.norm() for p in disc.parameters()]
                ).mean()

                del disc_loss_list[0]
                del grad_pen_list[0]
                disc_loss_list.append(disc_loss.item())
                grad_pen_list.append(grad_pen.item())

                # train generator
                if iter_idx % 5 == 0:
                    # sample random data
                    rand_fake = multi_norm.sample(
                        (batch_size,)) \
                        .cuda()

                    # generate fake data
                    x_fake = gen(rand_fake)

                    # pass to discriminator
                    out_fake = disc(x_fake)

                    # compute generator loss
                    gen_loss = networks.wasserstein_generator_loss(out_fake)

                    # reset gradient
                    optim_gen.zero_grad()
                    optim_disc.zero_grad()

                    # backward and optim step
                    gen_loss.backward()
                    optim_gen.step()

                    # generator metrics
                    gen_grad_norm = th.tensor(
                        [p.grad.norm() for p in gen.parameters()]
                    ).mean()

                    del error_gen[0]
                    error_gen.append(out_fake.mean().item())

                    del gen_loss_list[0]
                    gen_loss_list.append(gen_loss.item())

                # update tqdm bar
                tqdm_bar.set_description(
                    f"Epoch {e:02} - disc, "
                    f"disc_loss = {mean(disc_loss_list):.6f}, "
                    f"gen_loss = {mean(gen_loss_list):.6f}, "
                    f"disc_grad_pen = {mean(grad_pen_list):.2f}, "
                    f"e_tp = {mean(error_tp):.5f}, "
                    f"e_tn = {mean(error_tn):.5f}, "
                    f"e_gen = {mean(error_gen):.5f}"
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

            # Generate sound
            with th.no_grad():

                for gen_idx in range(3):
                    # 10 seconds
                    rand_fake = multi_norm.sample(
                        (1,)) \
                        .cuda()

                    x_fake = gen(rand_fake, 10)

                    audio.magn_phase_to_wav(
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
