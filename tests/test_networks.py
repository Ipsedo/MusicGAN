import pytest
import torch as th

from music_gan.networks import (
    INPUT_SIZES,
    MAX_GROW,
    OUTPUT_SIZES,
    Discriminator,
    Generator,
)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("rand_channels", [1, 2, 3])
@pytest.mark.parametrize("curr_grow", [0, 4, MAX_GROW])
def test_generator(
    batch_size: int, rand_channels: int, curr_grow: int, use_cuda: bool
) -> None:
    gen = Generator(rand_channels, curr_grow)

    assert gen.curr_layer == curr_grow
    assert gen.growing == (curr_grow != MAX_GROW)

    if use_cuda:
        gen.cuda()
        device = "cuda"
    else:
        device = "cpu"

    assert gen.layer_nb == MAX_GROW

    w, h = INPUT_SIZES

    z = th.randn(batch_size, rand_channels, w, h, device=device)

    o = gen(z, 0.5)

    expected_w = w * 2 ** (curr_grow + 1)
    expected_h = h * 2 ** (curr_grow + 1)

    assert len(o.size()) == 4
    assert o.size()[0] == batch_size
    assert o.size()[1] == 2
    assert o.size()[2] == expected_w
    assert o.size()[3] == expected_h


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("curr_grow", [0, 4, MAX_GROW])
def test_discriminator(
    batch_size: int, curr_grow: int, use_cuda: bool
) -> None:
    disc = Discriminator(curr_grow)

    assert disc.curr_layer == curr_grow
    assert disc.growing == (curr_grow != 0)

    if use_cuda:
        disc.cuda()
        device = "cuda"
    else:
        device = "cpu"

    w, h = OUTPUT_SIZES

    expected_size = 2**curr_grow
    w = int(w / expected_size)
    h = int(h / expected_size)

    x = th.randn(batch_size, 2, w, h, device=device)

    o = disc(x, 0.5)

    assert len(o.size()) == 2
    assert o.size()[0] == batch_size
    assert o.size()[1] == 1

    grad_pen = disc.gradient_penalty(x, x.clone(), 0.5)

    assert len(grad_pen.size()) == 0


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("rand_channels", [1, 2, 3])
@pytest.mark.parametrize("curr_grow", list(range(MAX_GROW)))
def test_grow(
    batch_size: int, rand_channels: int, curr_grow: int, use_cuda: bool
) -> None:

    width, height = INPUT_SIZES

    alpha = 0.5

    gen = Generator(rand_channels, curr_grow)
    disc = Discriminator(MAX_GROW - curr_grow)

    if use_cuda:
        gen.cuda()
        disc.cuda()
        device = "cuda"
    else:
        device = "cpu"

    assert gen.growing
    assert disc.growing

    for i in range(MAX_GROW + 1):
        z = th.randn(batch_size, rand_channels, width, height, device=device)

        out = gen(z, alpha)

        expected_size = 2 ** (gen.curr_layer + 1)

        assert len(out.size()) == 4
        assert out.size()[0] == batch_size
        assert out.size()[1] == 2
        assert width * expected_size == out.size()[2]
        assert height * expected_size == out.size()[3]

        out_disc = disc(out, alpha)
        grad_pen = disc.gradient_penalty(out, out.clone(), alpha)

        is_growing = gen.curr_layer != gen.layer_nb
        assert gen.growing == is_growing
        assert disc.growing == is_growing

        assert batch_size == out_disc.size()[0]
        assert 1 == out_disc.size()[1]

        assert len(grad_pen.size()) == 0

        gen.next_layer()
        disc.next_layer()
