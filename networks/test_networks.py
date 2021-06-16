import torch as th
from networks import Generator, Discriminator

if __name__ == '__main__':
    batch_size = 5
    style_channels = 256

    gen = Generator(style_channels)

    z = th.randn(batch_size, style_channels)

    print(gen)

    out = gen(z)

    print(out.size())

    out_2 = gen(z, 10)

    print(out_2.size())
