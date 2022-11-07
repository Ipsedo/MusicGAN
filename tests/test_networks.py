import unittest

import torch as th
from music_gan.networks import Generator, Discriminator, INPUT_SIZES


class TestNetworks(unittest.TestCase):
    def test_grow(self):
        batch_size = 5
        rand_channels = 8

        width, height = INPUT_SIZES

        alpha = 0.5

        gen = Generator(rand_channels)
        disc = Discriminator(7)

        for i in range(gen.down_sample + 3):
            z = th.randn(
                batch_size,
                rand_channels,
                width,
                height
            )

            out = gen(z, alpha)

            expected_size = 2 ** (
                i + 1 if disc.growing and gen.growing
                else gen.down_sample + 1
            )

            self.assertEqual(width * expected_size, out.size()[2])
            self.assertEqual(height * expected_size, out.size()[3])

            out_disc = disc(out, alpha)

            self.assertEqual(gen.growing, disc.growing)
            self.assertEqual(batch_size, out_disc.size()[0])
            self.assertEqual(1, out_disc.size()[1])

            gen.next_layer()
            disc.next_layer()
