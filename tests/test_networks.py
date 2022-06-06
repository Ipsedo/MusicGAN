import unittest

import torch as th
from music_gan.networks import Generator, Discriminator


class TestNetworks(unittest.TestCase):
    def test_grow(self):
        try:
            batch_size = 5
            rand_channels = 8

            alpha = 0.5
            slope = 0.5

            gen = Generator(rand_channels)
            disc = Discriminator(7)

            for i in range(gen.down_sample + 3):
                z = th.randn(batch_size, rand_channels, 2, 2)

                out = gen(z, slope, alpha)

                out_disc = disc(out, slope, alpha)

                self.assertEqual(gen.growing, disc.growing)
                self.assertEqual(out_disc.size()[0], batch_size)

                gen.next_layer()
                disc.next_layer()

        except Exception as e:
            print(e.with_traceback())
            self.fail(str(e))
