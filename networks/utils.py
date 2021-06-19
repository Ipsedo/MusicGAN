import torch as th


# https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
def tile(a: th.Tensor, dim: int, n_tile: int) -> th.Tensor:
    init_dim = a.size(dim)

    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile

    a = a.repeat(*repeat_idx)

    order_index = th.cat([
        init_dim * th.arange(n_tile, device=a.device) + i
        for i in range(init_dim)]
    ).to(th.long)

    return th.index_select(a, dim, order_index)
