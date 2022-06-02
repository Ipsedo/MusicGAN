import unittest

import torch as th

from music_gan.networks import matrix_multiple


class TestMatrixMultiple(unittest.TestCase):
    def setUp(self):
        self.__decimal = 2

        self.__input_dim_max = 64
        self.__input_dim_min = 2

        self.__output_dim = 2

        self.__intermediate_dim_max = 56
        self.__intermediate_dim_min = 8

    @staticmethod
    def __round(t: th.Tensor, decimals: int = 0) -> th.Tensor:
        return (t * 10 ** decimals).round() / (10 ** decimals)

    def __get_random_input_dim(self) -> int:
        return (
            th.randint(self.__input_dim_max - self.__input_dim_min, [1])[0].item() +
            self.__input_dim_min
        )

    def __get_random_intermediate_dim(self) -> int:
        return (
            th.randint(self.__intermediate_dim_max - self.__intermediate_dim_min, [1])[0].item() +
            self.__intermediate_dim_min
        )

    def test_dim_equality(self):
        input_dim = self.__get_random_input_dim()
        intermediate_dim = self.__get_random_intermediate_dim()

        a = th.randn(input_dim, self.__output_dim)
        b, c = matrix_multiple(a, intermediate_dim)

        self.assertEqual(a.size()[0], b.size()[0])
        self.assertEqual(a.size()[1], c.size()[1])

        self.assertEqual(b.size()[1], intermediate_dim)
        self.assertEqual(c.size()[0], intermediate_dim)

    def test_equality(self):
        input_dim = self.__get_random_input_dim()
        intermediate_dim = self.__get_random_intermediate_dim()

        a = th.randn(input_dim, self.__output_dim)
        b, c = matrix_multiple(a, intermediate_dim)

        self.assertTrue(
            th.all(
                self.__round(b @ c, decimals=self.__decimal) ==
                self.__round(a, decimals=self.__decimal)
            ).item()
        )


