import torch as th


class ChannelMinMaxNorm:
    def __call__(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 4
        assert x.size()[1] == 2

        tmp_x = (
            x
            .permute(1, 0, 2, 3)
            .reshape(2, -1)
        )

        x_max = (
            tmp_x
            .max(dim=-1)[0]
            .view(1, 2, 1, 1)
        )

        x_min = (
            tmp_x
            .min(dim=-1)[0]
            .view(1, 2, 1, 1)
        )

        return (x - x_min) / (x_max - x_min)


class ChangeRange:
    def __init__(self, lower_bond: float, upper_bound: float):
        self.__range = upper_bound - lower_bond
        self.__start = lower_bond

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return x * self.__range + self.__start
