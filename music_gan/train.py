from . import audio
from .audio import constant
from . import networks

import torch as th
from torchvision.transforms import Compose, Resize
from torch.utils.data import DataLoader

from os import mkdir
from os.path import join, exists, isdir

import mlflow

import matplotlib.pyplot as plt

from tqdm import tqdm

from statistics import mean


def get_transform(downscale_factor: int) -> Compose:
    size = 512

    target_size = size // 2 ** downscale_factor

    compose = Compose([
        audio.ChannelMinMaxNorm(),
        audio.ChangeRange(-1., 1.),
        Resize(target_size)
    ])

    return compose


def train(
        run_name: str,
        input_dataset_path: str,
        output_dir: str,
) -> None:

    exp_name = "MusicGAN"
    mlflow.set_experiment(exp_name)

    assert isdir(input_dataset_path), \
        f"\"{input_dataset_path}\" doesn't exist or is not a directory"

    mlflow.start_run(run_name=run_name)

    mlflow.log_param(
        "input_dataset",
        input_dataset_path
    )

    sample_rate = constant.SAMPLE_RATE

    rand_channels = 32
    height = 2
    width = 2

    disc_lr = 1e-3
    gen_lr = 1e-3
    betas = (0.5, 0.9)

    nb_epoch = 1000
    batch_size = 4

    if not exists(output_dir):
        mkdir(output_dir)
    elif exists(output_dir) and not isdir(output_dir):
        raise Exception(
            f"\"{output_dir}\" is not a directory !"
        )

    scale_factor = 7

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
    audio_dataset = audio.AudioDataset(input_dataset_path)

    data_loader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=True,
        pin_memory=False
    )

    mlflow.log_params({
        "rand_channels": rand_channels,
        "nb_epoch": nb_epoch,
        "batch_size": batch_size,
        "disc_lr": disc_lr,
        "gen_lr": gen_lr,
        "sample_rate": sample_rate,
        "width": width,
        "height": height
    })

    transform = get_transform(scale_factor)

    with mlflow.start_run(run_name="train", nested=True):

        metric_window = 20
        error_tp = [0. for _ in range(metric_window)]
        error_tn = [0. for _ in range(metric_window)]
        error_gen = [0. for _ in range(metric_window)]

        disc_loss_list = [0. for _ in range(metric_window)]
        grad_pen_list = [0. for _ in range(metric_window)]
        gen_loss_list = [0. for _ in range(metric_window)]

        iter_idx = 0
        save_idx = 0

        save_every = 1000
        grow_idx = 0
        grow_every = [
            8000,
            16000,
            24000,
            32000,
            40000,
            48000,
            56000
        ]
        fadein_length = [
            1,
            4000,
            6000,
            8000,
            10000,
            12000,
            14000,
            16000
        ]

        for e in range(nb_epoch):

            tqdm_bar = tqdm(data_loader)

            for x_real in tqdm_bar:
                alpha = min(
                    1., (1. + grow_idx) / fadein_length[gen.curr_layer]
                )

                # train discriminator

                # pass data to cuda
                x_real = x_real.to(th.float)
                x_real = transform(x_real).cuda()

                # sample random latent data
                z = th.randn(
                    batch_size,
                    rand_channels,
                    height,
                    width,
                    device="cuda"
                )

                # gen fake data
                x_fake = gen(z, alpha)

                # pass real data and gen data to discriminator
                out_real = disc(x_real, alpha)
                out_fake = disc(x_fake, alpha)

                # compute discriminator loss
                disc_loss = networks.wasserstein_discriminator_loss(
                    out_real, out_fake
                )

                # compute gradient penalty
                grad_pen = disc.gradient_penalty(x_real, x_fake, alpha)

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
                    x_fake = gen(z, alpha)

                    # pass to discriminator
                    out_fake = disc(x_fake, alpha)

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
                    f"[{iter_idx // save_every:03}: "
                    f"{iter_idx % save_every:04} / {save_every}], "
                    f"disc_loss = {mean(disc_loss_list):.4f}, "
                    f"gen_loss = {mean(gen_loss_list):.4f}, "
                    f"disc_grad_pen = {mean(grad_pen_list):.4f}, "
                    f"e_tp = {mean(error_tp):.4f}, "
                    f"e_tn = {mean(error_tn):.4f}, "
                    f"e_gen = {mean(error_gen):.4f}, "
                    f"alpha = {alpha:.3f}"
                )

                # log metrics
                if iter_idx % 200 == 0:
                    mlflow.log_metrics(step=e, metrics={
                        "disc_loss": disc_loss.item(),
                        "gen_loss": gen_loss.item(),
                        "batch_tp_error": error_tp[-1],
                        "batch_tn_error": error_tn[-1]
                    })

                if iter_idx % save_every == 0:

                    # Generate sound
                    with th.no_grad():

                        for gen_idx in range(6):
                            z = th.randn(
                                1, rand_channels,
                                height, width,
                                device="cuda"
                            )

                            x_fake = gen(z, alpha)

                            magn = x_fake[0, 0, :, :].detach().cpu().numpy()
                            phase = x_fake[0, 1, :, :].detach().cpu().numpy()

                            fig, ax = plt.subplots()
                            ax.matshow(magn / (magn.max() - magn.min()),
                                       cmap='plasma')
                            plt.title("gen magn " + str(save_idx) +
                                      " grow=" + str(gen.curr_layer))
                            fig.savefig(
                                join(output_dir,
                                     f"magn_{save_idx}_ID{gen_idx}.png")
                            )
                            plt.close()

                            fig, ax = plt.subplots()
                            ax.matshow(phase / (phase.max() - phase.min()),
                                       cmap='plasma')
                            plt.title("gen phase " + str(save_idx) +
                                      " grow=" + str(gen.curr_layer))
                            fig.savefig(
                                join(output_dir,
                                     f"phase_{save_idx}_ID{gen_idx}.png")
                            )
                            plt.close()

                            mlflow.log_artifact(
                                join(output_dir,
                                     f"magn_{save_idx}_ID{gen_idx}.png")
                            )

                            mlflow.log_artifact(
                                join(output_dir,
                                     f"phase_{save_idx}_ID{gen_idx}.png")
                            )

                    # Save discriminator
                    th.save(
                        disc.state_dict(),
                        join(output_dir, f"disc_{save_idx}.pt")
                    )
                    th.save(
                        optim_disc.state_dict(),
                        join(output_dir, f"optim_disc_{save_idx}.pt")
                    )

                    # save generator
                    th.save(
                        gen.state_dict(),
                        join(output_dir, f"gen_{save_idx}.pt")
                    )
                    th.save(
                        optim_gen.state_dict(),
                        join(output_dir, f"optim_gen_{save_idx}.pt")
                    )

                    # log models & optim to mlflow
                    mlflow.log_artifact(
                        join(output_dir, f"gen_{save_idx}.pt")
                    )
                    mlflow.log_artifact(
                        join(output_dir, f"optim_gen_{save_idx}.pt")
                    )
                    mlflow.log_artifact(
                        join(output_dir, f"disc_{save_idx}.pt")
                    )
                    mlflow.log_artifact(
                        join(output_dir, f"optim_disc_{save_idx}.pt")
                    )

                    save_idx += 1

                iter_idx += 1
                grow_idx += 1

                # ProGAN : add next layer
                if gen.growing and grow_idx % grow_every[gen.curr_layer] == 0:
                    scale_factor -= 1

                    transform = get_transform(scale_factor)

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

                    print("\nup_layer", gen.curr_layer, "/", gen.down_sample)

                    grow_idx = 0
