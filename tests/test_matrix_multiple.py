from typing import cast

import pytest
import torch as th

from music_gan.networks import matrix_multiple


def __round(t: th.Tensor, decimals: int = 0) -> th.Tensor:
    return cast(th.Tensor, (t * 10**decimals).round() / (10**decimals))


@pytest.mark.parametrize("input_dim", [32, 64])
@pytest.mark.parametrize("intermediate_dim", [8, 16])
@pytest.mark.parametrize("output_dim", [2, 4])
def test_dim_equality(
    input_dim: int, intermediate_dim: int, output_dim: int
) -> None:
    a = th.randn(input_dim, output_dim)
    b, c = matrix_multiple(a, intermediate_dim)

    assert a.size()[0] == b.size()[0]
    assert a.size()[1] == c.size()[1]

    assert b.size()[1] == intermediate_dim
    assert c.size()[0] == intermediate_dim


@pytest.mark.parametrize("input_dim", [32, 64])
@pytest.mark.parametrize("intermediate_dim", [8, 16])
@pytest.mark.parametrize("output_dim", [2, 4])
@pytest.mark.parametrize("decimal", [0, 1])
def test_equality(
    input_dim: int, intermediate_dim: int, output_dim: int, decimal: int
) -> None:
    a = th.randn(input_dim, output_dim)
    b, c = matrix_multiple(a, intermediate_dim)

    assert th.all(
        th.eq(__round(b @ c, decimals=decimal), __round(a, decimals=decimal))
    ).item()
