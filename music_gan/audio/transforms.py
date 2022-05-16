import torch as th


class ChannelMinMaxNorm:
    def __init__(self, epsilon: float = 1e-8):
        self.__epsilon = epsilon

    def __call__(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 4
        assert x.size()[1] == 2

        batch_size = x.size()[0]

        tmp_x = x.view(batch_size, 2, -1)

        x_max = (
            tmp_x
            .max(dim=-1)[0]
            .view(batch_size, 2, 1, 1)
        )

        x_min = (
            tmp_x
            .min(dim=-1)[0]
            .view(batch_size, 2, 1, 1)
        )

        return (x - x_min) / (x_max - x_min + self.__epsilon)


class ChangeRange:
    def __init__(self, lower_bond: float, upper_bound: float):
        self.__range = upper_bound - lower_bond
        self.__start = lower_bond

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return x * self.__range + self.__start
