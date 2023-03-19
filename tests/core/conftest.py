import pytest

from pylixir.application.enchant import EnchantCommand
from pylixir.core.base import Enchanter


@pytest.fixture(name="clean_enchanter")
def fixture_clean_enchanter() -> Enchanter:
    return Enchanter()


@pytest.fixture(name="enchant_command")
def fixture_enchant_command() -> EnchantCommand:
    return EnchantCommand()
