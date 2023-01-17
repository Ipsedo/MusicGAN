from os.path import dirname, join

import pytest


@pytest.fixture(name="use_cuda", scope="session")
def use_cuda() -> bool:
    return True


@pytest.fixture(name="wav_path", scope="session")
def wav_path() -> str:
    return join(dirname(__file__), "resources", "example.wav")
