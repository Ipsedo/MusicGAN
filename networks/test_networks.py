import torch as th
from networks import Generator, Discriminator

if __name__ == '__main__':
    batch_size = 5
    rand_channels = 64
    style_channels = 256

    gen = Generator(rand_channels, style_channels, 0).cuda()
    disc = Discriminator(2).cuda()

    print(gen)
    print(disc)

    for i in range(gen.nb_layer+3):
        rand = th.randn(5, rand_channels, 2, 2).cuda()

        print("input_size :", rand.size())
        style_rand = th.randn(5, style_channels).cuda()

        out = gen(rand, style_rand)

        print("A", out.size())

        out_disc = disc(out)

        print("B", out_disc.size())

        gen.next_layer()
        disc.next_layer()

    # print(out.size())

    # out = disc(out)

    # print(out.size())
