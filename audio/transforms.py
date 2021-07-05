import torch as th


class ChannelMinMaxNorm:
    def __call__(self, x: th.Tensor) -> th.Tensor:
        x_max = x.view(2, -1).max(dim=-1)[0].view(2, 1, 1)
        x_min = x.view(2, -1).min(dim=-1)[0].view(2, 1, 1)
        return (x - x_min) / (x_max - x_min)


class ChangeRange:
    def __init__(self, lower_bond: float, upper_bound: float):
        self.__range = upper_bound - lower_bond
        self.__start = lower_bond

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return x * self.__range + self.__start
