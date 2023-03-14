import pytest

from pylixir.core.base import Enchanter


@pytest.fixture(name="clean_enchanter")
def fixture_clean_enchanter() -> Enchanter:
    return Enchanter()
