import torch as th
from networks import Generator, Discriminator

if __name__ == '__main__':
    rand_data = th.rand(5, 8, 1, 1)

    # rs = ResidualTransConv(8, 24, 16, 3, 4, 2)
    """gu = GatedActUnit(
        8, 10, 16, 3, 2
    )

    o = gu(rand_data)

    print(o.size())"""

    gen = Generator(8, 2)

    print(gen)

    o = gen(rand_data)

    print("A ", o.size())

    disc = Discriminator(2)

    print(disc)

    o = disc(o)

    print(o.size())
