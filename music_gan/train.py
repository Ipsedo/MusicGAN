from os import mkdir
from os.path import exists, isdir
from statistics import mean

import mlflow
import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import audio, networks
from .utils import Grower, Saver


def train(
    run_name: str,
    input_dataset_path: str,
    output_dir: str,
) -> None:
    th.backends.cudnn.benchmark = True

    exp_name = "music_gan"
    mlflow.set_experiment(exp_name)

    assert isdir(
        input_dataset_path
    ), f'"{input_dataset_path}" doesn\'t exist or is not a directory'

    mlflow.start_run(run_name=run_name)

    sample_rate = audio.SAMPLE_RATE

    rand_channels = 32
    height = networks.INPUT_SIZES[0]
    width = networks.INPUT_SIZES[1]

    disc_lr = 1e-4
    gen_lr = 1e-4
    betas_disc = (0.0, 0.9)
    betas_gen = (0.0, 0.9)

    nb_epoch = 1000
    batch_size = 4
    train_gen_every = 5

    if not exists(output_dir):
        mkdir(output_dir)
    elif exists(output_dir) and not isdir(output_dir):
        raise NotADirectoryError(f'"{output_dir}" is not a directory !')

    grower = Grower(
        n_grow=7,
        fadein_lengths=[
            1,
            32000,
            64000,
            128000,
            256000,
            512000,
            512000,
            512000,
            # 1,1,1,1,1,1,1,1
        ],
        train_lengths=[
            32000,
            64000,
            128000,
            256000,
            512000,
            1024000,
            1024000,
            # 1,1,1,1,1,1,1
        ],
    )

    saver = Saver(
        output_dir,
        save_every=2000,
        rand_channels=rand_channels,
        rand_height=height,
        rand_width=width,
    )

    gen = networks.Generator(rand_channels, end_layer=0)

    disc = networks.Discriminator(start_layer=7)

    gen.cuda()
    disc.cuda()

    optim_gen = th.optim.Adam(gen.parameters(), lr=gen_lr, betas=betas_gen)

    optim_disc = th.optim.Adam(disc.parameters(), lr=disc_lr, betas=betas_disc)

    # create DataSet
    audio_dataset = audio.AudioDataset(input_dataset_path)

    data_loader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=True,
        pin_memory=True,
    )

    mlflow.log_params(
        {
            "input_dataset": input_dataset_path,
            "nb_sample": len(audio_dataset),
            "output_dir": output_dir,
            "rand_channels": rand_channels,
            "nb_epoch": nb_epoch,
            "batch_size": batch_size,
            "train_gen_every": train_gen_every,
            "disc_lr": disc_lr,
            "gen_lr": gen_lr,
            "betas_disc": betas_disc,
            "betas_gen": betas_gen,
            "sample_rate": sample_rate,
            "width": width,
            "height": height,
        }
    )

    with mlflow.start_run(run_name="train", nested=True):

        metric_window = 20
        error_tp = [0.0 for _ in range(metric_window)]
        error_tn = [0.0 for _ in range(metric_window)]
        error_gen = [0.0 for _ in range(metric_window)]

        disc_error_list = [0.0 for _ in range(metric_window)]
        disc_gp_list = [0.0 for _ in range(metric_window)]

        gen_error_list = [0.0 for _ in range(metric_window)]

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
                disc_gp = disc.gradient_penalty(x_real, x_fake, grower.alpha)

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
                if iter_idx % train_gen_every == 0:
                    # sample random latent data
                    z = th.randn(
                        batch_size,
                        rand_channels,
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
                    gen_error = networks.wasserstein_generator_loss(out_fake)

                    # generator loss
                    gen_loss = gen_error

                    # reset gradient
                    optim_gen.zero_grad(set_to_none=True)

                    # backward pass and weight update
                    gen_loss.backward()
                    optim_gen.step()

                    # generator metrics
                    del error_gen[0]
                    error_gen.append(out_fake.mean().item())

                    del gen_error_list[0]
                    gen_error_list.append(gen_error.item())

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
                            "disc_loss": disc_loss.item(),
                            "gen_loss": gen_loss.item(),
                            "disc_gp": disc_gp.item(),
                            "batch_tp_error": error_tp[-1],
                            "batch_tn_error": error_tn[-1],
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
