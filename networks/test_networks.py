import torch as th
from networks import Generator, Discriminator

if __name__ == '__main__':
    batch_size = 5
    rand_channels = 8
    rand_style_channels = 32
    style_channels = 128

    alpha = 0.5

    gen = Generator(rand_channels, rand_style_channels, style_channels, 0)
    disc = Discriminator(7)

    for i in range(gen.down_sample + 3):
        rand = th.randn(5, rand_channels, 2, 2)
        style_rand = th.randn(5, rand_style_channels)

        print("input_size :", rand.size())

        out = gen(rand, style_rand, alpha)

        print("A", out.size())

        out_disc = disc(out, alpha)

        print("B", out_disc.size())

        print(gen.growing)
        print(disc.growing)

        gen.next_layer()
        disc.next_layer()

    # print(out.size())

    # out = disc(out)

    # print(out.size())
    print(gen)
    print(disc)
