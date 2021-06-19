import torch as th
from networks import Generator, Discriminator

if __name__ == '__main__':
    batch_size = 5
    style_channels = 32

    gen = Generator(style_channels)
    disc = Discriminator(2)

    print(gen)
    print(disc)

    out = gen(3, 1)

    print(out.size())

    out = disc(out)

    print(out.size())
