import pytest


@pytest.fixture(name="use_cuda", scope="session")
def use_cuda() -> bool:
    return True
