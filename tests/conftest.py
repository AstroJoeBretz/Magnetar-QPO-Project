import mpmath as mp
import pytest


@pytest.fixture(autouse=True)
def _default_precision():
    """Every test starts at 60 digits regardless of what other tests set."""
    with mp.workdps(60):
        yield
