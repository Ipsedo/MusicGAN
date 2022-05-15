import torch as th

from typing import Tuple


def __fill_diag(m: th.Tensor, v: th.Tensor) -> th.Tensor:
    device = "cuda" if m.is_cuda else "cpu"

    def __fill(values: th.Tensor) -> th.Tensor:
        to_fill = th.zeros(*m.size(), device=device)
        start, end = \
            min(m.size()[0], values.size()[0]), \
            min(m.size()[1], values.size()[1])
        to_fill[:start, :end] = values[:start, :end]

        return to_fill

    mask = th.diag(th.ones_like(v))
    full_mask = __fill(mask)

    diag = th.diag(v)
    full_diag = __fill(diag)

    return full_mask * full_diag + (1. - full_mask) * m


def decomposition(m: th.Tensor, p: int) -> Tuple[th.Tensor, th.Tensor]:
    assert len(m.size()) == 2, "Need 2-D Tensor (projection matrix)"

    device = "cuda" if m.is_cuda else "cpu"

    n, q = m.size()[-2:]

    u, d, v = th.linalg.svd(m, full_matrices=True)

    d_diag = th.zeros(p, device=device)
    d_diag[:min(n, p, q)] = th.sqrt(d[:min(n, p, q)])

    d1 = th.zeros(n, p, device=device)
    d1 = __fill_diag(d1, d_diag)

    d2 = th.zeros(p, q, device=device)
    d2 = __fill_diag(d2, d_diag)

    g = th.randn(p, p, device=device)
    g = g + g.transpose(1, 0)
    e, w = th.linalg.eigh(g)

    b = u @ (d1 @ w)
    c = w.transpose(1, 0) @ (d2 @ v)

    return b, c
