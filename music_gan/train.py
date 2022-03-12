from . import audio
from . import networks
from .utils import Grower, Saver

import torch as th
from torch.utils.data import DataLoader

from os import mkdir
from os.path import exists, isdir

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

    rand_channels = 32
    height = 2
    width = 2

    disc_lr = 1e-4
    gen_lr = 1e-4
    betas = (0.0, 0.99)

    nb_epoch = 1000
    batch_size = 6

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

    disc = networks.Discriminator(
        start_layer=7
    )

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
            1, 200000, 200000, 200000, 200000, 200000, 200000, 200000,
        ],
        train_lengths=[
            200000, 400000, 400000, 400000, 400000, 400000, 400000,
        ]
    )

    saver = Saver(
        output_dir, save_every=1000,
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
        grad_pen_list = [0. for _ in range(metric_window)]
        gen_loss_list = [0. for _ in range(metric_window)]

        iter_idx = 0

        for e in range(nb_epoch):

            tqdm_bar = tqdm(data_loader)

            for x_real in tqdm_bar:
                # train discriminator

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

                # compute gradient penalty
                grad_pen = disc.gradient_penalty(x_real, x_fake, grower.alpha)

                # add gradient penalty
                disc_loss_gp = disc_loss + grad_pen

                # reset grad
                gen.zero_grad()
                disc.zero_grad()

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

                # train generator
                if iter_idx % 5 == 0:
                    # sample random latent data
                    z = th.randn(
                        batch_size,
                        rand_channels,
                        height,
                        width,
                        device="cuda"
                    )

                    # generate fake data
                    x_fake = gen(z, grower.alpha)

                    # pass to discriminator
                    out_fake = disc(x_fake, grower.alpha)

                    # compute generator loss
                    gen_loss = networks.wasserstein_generator_loss(out_fake)

                    # reset gradient
                    gen.zero_grad()
                    disc.zero_grad()

                    # backward and optim step
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
                    f"gen_l = {mean(gen_loss_list):.2f}, "
                    f"grad_p = {mean(grad_pen_list):.4f}, "
                    f"e_tp = {mean(error_tp):.2f}, "
                    f"e_tn = {mean(error_tn):.2f}, "
                    f"e_gen = {mean(error_gen):.2f}, "
                    f"alpha = {grower.alpha:.3f}"
                )

                # log metrics
                if iter_idx % 200 == 0:
                    mlflow.log_metrics(step=gen.curr_layer, metrics={
                        "disc_loss": disc_loss.item(),
                        "gen_loss": gen_loss.item(),
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
                if grower.grow(batch_size) and gen.growing:
                    gen.next_layer()
                    disc.next_layer()

                    optim_gen.add_param_group({
                        "params": gen.end_block_params(),
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
