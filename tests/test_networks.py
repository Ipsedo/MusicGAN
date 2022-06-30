import unittest

import torch as th
from music_gan.networks import Generator, Discriminator


class TestNetworks(unittest.TestCase):
    def test_grow(self):
        try:
            batch_size = 5
            rand_channels = 8

            input_size = 2

            alpha = 0.5

            gen = Generator(rand_channels)
            disc = Discriminator(7)

            for i in range(gen.down_sample + 3):
                z = th.randn(
                    batch_size,
                    rand_channels,
                    input_size,
                    input_size
                )

                out = gen(z, alpha)

                expected_size = input_size * 2 ** (
                    i + 1 if disc.growing and gen.growing
                    else gen.down_sample + 1
                )

                self.assertEqual(out.size()[2], expected_size)
                self.assertEqual(out.size()[3], expected_size)

                out_disc = disc(out, alpha)

                self.assertEqual(gen.growing, disc.growing)
                self.assertEqual(out_disc.size()[0], batch_size)

                gen.next_layer()
                disc.next_layer()

        except Exception as e:
            self.fail(str(e))
