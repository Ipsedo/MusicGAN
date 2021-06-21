import torch as th
from networks import Generator, Discriminator

if __name__ == '__main__':
    batch_size = 5
    rand_channels = 64
    style_channels = 256

    gen = Generator(rand_channels, style_channels)
    disc = Discriminator(2)

    print(gen)
    print(disc)

    rand = th.randn(5, rand_channels, 2, 2)
    style_rand = th.randn(5, style_channels)

    out = gen(rand, style_rand)

    print(out.size())

    out = disc(out)

    print(out.size())
