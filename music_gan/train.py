from os import mkdir
from os.path import exists, isdir
from statistics import mean
from typing import List, NamedTuple, Tuple

import mlflow
import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import audio, networks
from .utils import Grower, Saver

TrainOptions = NamedTuple(
    "TrainOptions",
    [
        ("run_name", str),
        ("dataset_path", str),
        ("output_dir", str),
        ("rand_channels", int),
        ("disc_lr", float),
        ("gen_lr", float),
        ("disc_betas", Tuple[float, float]),
        ("gen_betas", Tuple[float, float]),
        ("nb_epoch", int),
        ("batch_size", int),
        ("train_gen_every", int),
        ("fadein_lengths", List[int]),
        ("train_lengths", List[int]),
        ("save_every", int),
    ],
)


def train(train_options: TrainOptions) -> None:
    th.backends.cudnn.benchmark = True

    exp_name = "music_gan"
    mlflow.set_experiment(exp_name)

    assert isdir(
        train_options.dataset_path
    ), f'"{train_options.dataset_path}" doesn\'t exist or is not a directory'

    mlflow.start_run(run_name=train_options.run_name)

    sample_rate = audio.SAMPLE_RATE

    height = networks.INPUT_SIZES[0]
    width = networks.INPUT_SIZES[1]

    if not exists(train_options.output_dir):
        mkdir(train_options.output_dir)
    elif not isdir(train_options.output_dir):
        raise NotADirectoryError(
            f'"{train_options.output_dir}" is not a directory !'
        )

    gen = networks.Generator(
        train_options.rand_channels,
        end_layer=0,
    )
    disc = networks.Discriminator(
        start_layer=7,
    )

    gen.cuda()
    disc.cuda()

    optim_gen = th.optim.Adam(
        gen.parameters(),
        lr=train_options.gen_lr,
        betas=train_options.gen_betas,
    )
    optim_disc = th.optim.Adam(
        disc.parameters(),
        lr=train_options.disc_lr,
        betas=train_options.disc_betas,
    )

    # create DataSet
    audio_dataset = audio.AudioDataset(train_options.dataset_path)

    data_loader = DataLoader(
        audio_dataset,
        batch_size=train_options.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=True,
        pin_memory=True,
    )

    grower = Grower(
        n_grow=gen.layer_nb,
        fadein_lengths=train_options.fadein_lengths,
        train_lengths=train_options.train_lengths,
    )

    saver = Saver(
        train_options.output_dir,
        save_every=train_options.save_every,
        rand_channels=train_options.rand_channels,
        rand_height=height,
        rand_width=width,
    )

    mlflow.log_params(
        {
            "input_dataset": train_options.dataset_path,
            "nb_sample": len(audio_dataset),
            "output_dir": train_options.output_dir,
            "rand_channels": train_options.rand_channels,
            "nb_epoch": train_options.nb_epoch,
            "batch_size": train_options.batch_size,
            "train_gen_every": train_options.train_gen_every,
            "disc_lr": train_options.disc_lr,
            "gen_lr": train_options.gen_lr,
            "disc_betas": train_options.disc_betas,
            "gen_betas": train_options.gen_betas,
            "sample_rate": sample_rate,
            "width": width,
            "height": height,
        }
    )

    metric_window = 20

    error_tp = [0.0 for _ in range(metric_window)]
    error_tn = [0.0 for _ in range(metric_window)]
    error_gen = [0.0 for _ in range(metric_window)]

    disc_error_list = [0.0 for _ in range(metric_window)]
    disc_gp_list = [0.0 for _ in range(metric_window)]

    gen_error_list = [0.0 for _ in range(metric_window)]

    iter_idx = 0

    for e in range(train_options.nb_epoch):

        tqdm_bar = tqdm(data_loader)

        for x_real in tqdm_bar:
            # [1] train discriminator

            # pass data to cuda
            x_real = x_real.to(th.float)
            x_real = grower.scale_transform(x_real).cuda()

            # sample random latent data
            z = th.randn(
                train_options.batch_size,
                train_options.rand_channels,
                height,
                width,
                device="cuda",
            )

            # gen fake data
            x_fake = gen(z, grower.alpha)

            # pass real data and gen data to discriminator
            out_real = disc(x_real, grower.alpha)
            out_fake = disc(x_fake, grower.alpha)

            # compute discriminator loss
            disc_error = networks.wasserstein_discriminator_loss(
                out_real, out_fake
            )

            # compute gradient penalty
            disc_gp = disc.gradient_penalty(
                x_real,
                x_fake,
                grower.alpha,
            )

            disc_loss = disc_error + disc_gp

            # reset grad
            optim_disc.zero_grad(set_to_none=True)

            # backward and optim step
            disc_loss.backward()
            optim_disc.step()

            # discriminator metrics
            del error_tn[0]
            del error_tp[0]
            error_tp.append(out_real.mean().item())
            error_tn.append(out_fake.mean().item())

            del disc_error_list[0]
            del disc_gp_list[0]
            disc_error_list.append(disc_error.item())
            disc_gp_list.append(disc_gp.item())

            # [2] train generator
            if iter_idx % train_options.train_gen_every == 0:
                # sample random latent data
                z = th.randn(
                    train_options.batch_size,
                    train_options.rand_channels,
                    height,
                    width,
                    device="cuda",
                )

                # reset gradient
                optim_disc.zero_grad(set_to_none=True)
                optim_gen.zero_grad(set_to_none=True)

                # generate fake data
                x_fake = gen(z, grower.alpha)

                # use unrolled discriminators
                out_fake = disc(x_fake, grower.alpha)

                # compute generator error
                gen_loss = networks.wasserstein_generator_loss(out_fake)

                # reset gradient
                optim_gen.zero_grad(set_to_none=True)

                # backward pass and weight update
                gen_loss.backward()
                optim_gen.step()

                # generator metrics
                del error_gen[0]
                error_gen.append(out_fake.mean().item())

                del gen_error_list[0]
                gen_error_list.append(gen_loss.item())

            # update tqdm bar
            tqdm_bar.set_description(
                f"Epoch {e:02} "
                f"[{saver.curr_save:03}: "
                f"{saver.save_counter:04}], "
                f"disc_l = {mean(disc_error_list):.4f}, "
                f"gen_l = {mean(gen_error_list):.3f}, "
                f"disc_gp = {mean(disc_gp_list):.3f}, "
                f"e_tp = {mean(error_tp):.2f}, "
                f"e_tn = {mean(error_tn):.2f}, "
                f"e_gen = {mean(error_gen):.2f}, "
                f"alpha = {grower.alpha:.3f} "
            )

            # log metrics
            if iter_idx % 500 == 0:
                mlflow.log_metrics(
                    step=gen.curr_layer,
                    metrics={
                        "disc_loss": mean(disc_error_list),
                        "gen_loss": mean(gen_error_list),
                        "disc_gp": mean(disc_gp_list),
                        "batch_tp_error": mean(error_tp),
                        "batch_tn_error": mean(error_tn),
                    },
                )

            # request save model
            # each N forward/backward pass
            saver.request_save(
                gen,
                disc,
                optim_gen,
                optim_disc,
                grower.alpha,
            )

            iter_idx += 1

            # ProGAN : add next layer
            # IF time_to_grow AND growing
            if grower.grow() and gen.growing:
                gen.next_layer()
                disc.next_layer()

                tqdm_bar.write(
                    "\n"
                    f"Next layer, {gen.curr_layer} / {gen.layer_nb}, "
                    f"curr_save = {saver.curr_save}"
                )
