import torch as th
from networks import Generator, Discriminator

if __name__ == '__main__':
    batch_size = 5
    style_channels = 256

    gen = Generator(style_channels)
    disc = Discriminator(2)

    z = th.randn(batch_size, style_channels)

    print(gen)
    print(disc)

    out = gen(z)

    print(out.size())

    out = disc(out)

    print(out.size())
