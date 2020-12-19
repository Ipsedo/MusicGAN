import torch as th


def discriminator_loss(y_real: th.Tensor, y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(th.log2(y_real) + th.log2(1. - y_fake))


def generator_loss(y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(th.log2(y_fake))


def wasserstein_discriminator_loss(y_real: th.Tensor,
                                  y_fake: th.Tensor) -> th.Tensor:
    return -(th.mean(y_real) - th.mean(y_fake))


def wasserstein_generator_loss(y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(y_fake)
