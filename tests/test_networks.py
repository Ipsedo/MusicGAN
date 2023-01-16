import pytest
import torch as th

from music_gan.networks import INPUT_SIZES, Discriminator, Generator


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("rand_channels", [1, 2, 3])
def test_grow(batch_size: int, rand_channels: int) -> None:

    width, height = INPUT_SIZES

    alpha = 0.5

    gen = Generator(rand_channels)
    disc = Discriminator(7)

    for i in range(gen.down_sample + 3):
        z = th.randn(batch_size, rand_channels, width, height)

        out = gen(z, alpha)

        expected_size = 2 ** (
            i + 1 if disc.growing and gen.growing else gen.down_sample + 1
        )

        assert width * expected_size == out.size()[2]
        assert height * expected_size == out.size()[3]

        out_disc = disc(out, alpha)

        assert gen.growing == disc.growing
        assert batch_size == out_disc.size()[0]
        assert 1 == out_disc.size()[1]

        gen.next_layer()
        disc.next_layer()
