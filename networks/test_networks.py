import torch as th
from networks import Generator, Discriminator

if __name__ == '__main__':
    batch_size = 5
    rand_channels = 8
    style_rand_channels = 16

    alpha = 0.5

    gen = Generator(rand_channels, style_rand_channels)
    disc = Discriminator(7)

    for i in range(gen.down_sample + 3):
        z = th.randn(5, rand_channels, 2, 2)
        z_style = th.randn(5, style_rand_channels)

        print("input_size :", z.size())

        out = gen(z, z_style, alpha)

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
