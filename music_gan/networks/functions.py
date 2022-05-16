import torch as th

from typing import Tuple


def __fill_diag(m: th.Tensor, v: th.Tensor) -> th.Tensor:
    device = "cuda" if m.is_cuda else "cpu"

    def __fill(values: th.Tensor) -> th.Tensor:
        to_fill = th.zeros(*m.size(), device=device)
        end_dim_1, end_dim_2 = \
            min(m.size()[0], values.size()[0]), \
            min(m.size()[1], values.size()[1])
        to_fill[:end_dim_1, :end_dim_2] = values[:end_dim_1, :end_dim_2]

        return to_fill

    mask = th.diag(th.ones_like(v))
    full_mask = __fill(mask)

    diag = th.diag(v)
    full_diag = __fill(diag)

    return full_mask * full_diag + (1. - full_mask) * m


def matrix_multiple(m: th.Tensor, p: int) -> Tuple[th.Tensor, th.Tensor]:
    """
    We want `a = b @ c` :
    1. a = u @ d @ v -> SVD
    2. a = u @ d1 @ d2 @ v
    3. a = u @ d1 @ w @ transpose(w) @ d2 @ v
    4. b = u @ d1 @ w
    5. c = transpose(w) @ d2 @ v

    :param m: A matrix as a 2-D Tensor
    :type m: torch.Tensor
    :param p: The intermediate dimension
    :type p: int
    :return:
    """
    assert len(m.size()) == 2, "Need 2-D Tensor (projection matrix)"

    device = "cuda" if m.is_cuda else "cpu"

    n, q = m.size()[-2:]

    # SVD decomposition
    u, d, v = th.linalg.svd(m, full_matrices=True)

    # diagonal of d matrix vectorization
    d_diag = th.zeros(p, device=device)
    diag_end = min(n, p, q)
    d_diag[:diag_end] = th.sqrt(d[:diag_end])

    # such as : d1 @ d2 = d
    d1 = th.zeros(n, p, device=device)
    d1 = __fill_diag(d1, d_diag)

    d2 = th.zeros(p, q, device=device)
    d2 = __fill_diag(d2, d_diag)

    # get W matrix such as : W @ W^(-1) = W @ transpose(W) = I
    g = th.randn(p, p, device=device)
    g = g + g.transpose(1, 0)
    _, w = th.linalg.eigh(g)

    # compute b and c
    b = u @ d1 @ w
    c = w.transpose(1, 0) @ d2 @ v

    return b, c
