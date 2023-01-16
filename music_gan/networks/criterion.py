import torch as th
import torch.nn.functional as F


def discriminator_loss(y_real: th.Tensor, y_fake: th.Tensor) -> th.Tensor:
    return F.binary_cross_entropy(
        y_real, th.ones_like(y_real), reduction="mean"
    ) + F.binary_cross_entropy(y_fake, th.zeros_like(y_fake), reduction="mean")


def generator_loss(y_fake: th.Tensor) -> th.Tensor:
    return F.binary_cross_entropy(
        y_fake, th.ones_like(y_fake), reduction="mean"
    )


def wasserstein_discriminator_loss(
    y_real: th.Tensor, y_fake: th.Tensor
) -> th.Tensor:
    return -th.mean(y_real - y_fake)


def wasserstein_generator_loss(y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(y_fake)
