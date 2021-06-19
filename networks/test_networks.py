import torch as th
from networks import Generator, Discriminator

if __name__ == '__main__':
    batch_size = 5
    rand_channels = 64

    gen = Generator(rand_channels)
    disc = Discriminator(2)

    print(gen)
    print(disc)

    rand = th.randn(5, rand_channels, 1, 1)

    out = gen(rand)

    print(out.size())

    out = disc(out)

    print(out.size())
