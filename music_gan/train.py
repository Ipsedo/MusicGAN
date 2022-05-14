from . import audio
from . import networks
from .utils import Grower, Saver

import torch as th
from torch.utils.data import DataLoader

import higher

from os import mkdir
from os.path import exists, isdir

import copy

import mlflow

from tqdm import tqdm

from statistics import mean


def train(
        run_name: str,
        input_dataset_path: str,
        output_dir: str,
) -> None:

    exp_name = "music_gan"
    mlflow.set_experiment(exp_name)

    assert isdir(input_dataset_path), \
        f"\"{input_dataset_path}\" doesn't exist or is not a directory"

    mlflow.start_run(run_name=run_name)

    sample_rate = audio.SAMPLE_RATE

    rand_channels = 16
    height = 2
    width = 2

    disc_lr = 4e-4
    gen_lr = 4e-4
    betas = (0., 0.9)

    nb_epoch = 1000
    batch_size = 8
    train_gen_every = 2

    if not exists(output_dir):
        mkdir(output_dir)
    elif exists(output_dir) and not isdir(output_dir):
        raise NotADirectoryError(
            f"\"{output_dir}\" is not a directory !"
        )

    gen = networks.Generator(
        rand_channels,
        end_layer=0
    )

    disc = networks.Discriminator()

    gen.cuda()
    disc.cuda()

    optim_gen = th.optim.Adam(
        gen.parameters(), lr=gen_lr, betas=betas
    )

    optim_disc = th.optim.Adam(
        disc.parameters(), lr=disc_lr, betas=betas
    )

    # create DataSet
    audio_dataset = audio.AudioDataset(
        input_dataset_path
    )

    data_loader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=True,
        pin_memory=False
    )

    mlflow.log_params({
        "input_dataset": input_dataset_path,
        "nb_sample": len(audio_dataset),
        "output_dir": output_dir,
        "rand_channels": rand_channels,
        "nb_epoch": nb_epoch,
        "batch_size": batch_size,
        "train_gen_every": train_gen_every,
        "disc_lr": disc_lr,
        "gen_lr": gen_lr,
        "betas": betas,
        "sample_rate": sample_rate,
        "width": width,
        "height": height
    })

    grower = Grower(
        n_grow=7,
        fadein_lengths=[
            1, 40000, 40000, 40000, 40000, 40000, 40000, 40000,
            #1,1,1,1,1,1,1,1
        ],
        train_lengths=[
            60000, 100000, 100000, 100000, 100000, 100000, 100000,
            #1,1,1,1,1,1,1
        ]
    )

    saver = Saver(
        output_dir,
        save_every=1000,
        rand_channels=rand_channels,
        rand_height=height,
        rand_width=width
    )

    with mlflow.start_run(run_name="train", nested=True):

        metric_window = 20
        error_tp = [0. for _ in range(metric_window)]
        error_tn = [0. for _ in range(metric_window)]
        error_gen = [0. for _ in range(metric_window)]

        disc_loss_list = [0. for _ in range(metric_window)]
        gen_loss_list = [0. for _ in range(metric_window)]
        grad_pen_list = [0. for _ in range(metric_window)]

        iter_idx = 0

        for e in range(nb_epoch):

            tqdm_bar = tqdm(data_loader)

            for x_real in tqdm_bar:
                # [1] train discriminator

                # pass data to cuda
                x_real = x_real.to(th.float)
                x_real = grower.scale_transform(x_real).cuda()

                # sample random latent data
                z = th.randn(
                    batch_size,
                    rand_channels,
                    height,
                    width,
                    device="cuda"
                )

                # gen fake data
                x_fake = gen(z, grower.alpha)

                # pass real data and gen data to discriminator
                out_real = disc(x_real, grower.alpha)
                out_fake = disc(x_fake, grower.alpha)

                # compute discriminator loss
                disc_loss = networks.wasserstein_discriminator_loss(
                    out_real, out_fake
                )

                grad_pen = disc.gradient_penalty(
                    x_real, x_fake, grower.alpha
                )

                disc_loss_gp = disc_loss + grad_pen

                # reset grad
                optim_disc.zero_grad()

                # backward and optim step
                disc_loss_gp.backward()
                optim_disc.step()

                # discriminator metrics
                del error_tn[0]
                del error_tp[0]
                error_tp.append(out_real.mean().item())
                error_tn.append(out_fake.mean().item())

                del disc_loss_list[0]
                del grad_pen_list[0]
                disc_loss_list.append(disc_loss.item())
                grad_pen_list.append(grad_pen.item())

                # [2] train generator

                if iter_idx % train_gen_every == 0:

                    # sample random latent data
                    z = th.randn(
                        batch_size,
                        rand_channels,
                        height,
                        width,
                        device="cuda"
                    )

                    # reset gradient
                    optim_disc.zero_grad()
                    optim_gen.zero_grad()

                    # generate fake data
                    x_fake = gen(z, grower.alpha)

                    # use unrolled discriminators
                    out_fake = disc(x_fake, grower.alpha)

                    # compute generator loss
                    gen_loss = networks.wasserstein_generator_loss(out_fake)

                    # reset gradient
                    optim_gen.zero_grad()

                    # backward pass and weight update
                    gen_loss.backward()
                    optim_gen.step()

                    # generator metrics
                    del error_gen[0]
                    error_gen.append(out_fake.mean().item())

                    del gen_loss_list[0]
                    gen_loss_list.append(gen_loss.item())

                # update tqdm bar
                tqdm_bar.set_description(
                    f"Epoch {e:02} "
                    f"[{saver.curr_save:03}: "
                    f"{saver.save_counter:03}], "
                    f"disc_l = {mean(disc_loss_list):.4f}, "
                    f"gen_l = {mean(gen_loss_list):.3f}, "
                    f"grad_pen = {mean(grad_pen_list):.3f}, "
                    f"e_tp = {mean(error_tp):.2f}, "
                    f"e_tn = {mean(error_tn):.2f}, "
                    f"e_gen = {mean(error_gen):.2f}, "
                    f"alpha = {grower.alpha:.3f} "
                )

                # log metrics
                if iter_idx % 200 == 0:
                    mlflow.log_metrics(step=gen.curr_layer, metrics={
                        "disc_loss": disc_loss.item(),
                        "gen_loss": gen_loss.item(),
                        "grad_pen": grad_pen.item(),
                        "batch_tp_error": error_tp[-1],
                        "batch_tn_error": error_tn[-1]
                    })

                # request save model
                # each N forward/backward pass
                saver.request_save(
                    gen, disc,
                    optim_gen, optim_disc,
                    grower.alpha
                )

                iter_idx += 1

                # ProGAN : add next layer
                # IF time_to_grow AND growing
                if grower.grow() and gen.growing:
                    gen.next_layer()
                    disc.next_layer()

                    optim_gen.add_param_group({
                        "params": gen.end_block_parameters(),
                        "lr": gen_lr,
                        "betas": betas
                    })

                    optim_disc.add_param_group({
                        "params": disc.start_block_parameters(),
                        "lr": disc_lr,
                        "betas": betas
                    })

                    tqdm_bar.write(
                        "\n"
                        f"Next layer, {gen.curr_layer} / {gen.down_sample}, "
                        f"curr_save = {saver.curr_save}"
                    )
