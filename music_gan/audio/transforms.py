import torch as th


class ChannelMinMaxNorm:
    def __init__(self, epsilon: float = 1e-8):
        self.__epsilon = epsilon

    def __call__(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 4
        assert x.size()[1] == 2

        x_max = x.amax(dim=(2, 3), keepdim=True)
        x_min = x.amin(dim=(2, 3), keepdim=True)

        return (x - x_min) / (x_max - x_min + self.__epsilon)


class ChangeRange:
    def __init__(self, lower_bond: float, upper_bound: float):
        self.__range = upper_bound - lower_bond
        self.__start = lower_bond

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return x * self.__range + self.__start
