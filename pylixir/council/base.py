import abc
import enum

import pydantic

from pylixir.core.base import GameState

"""
class CouncilLogicType(enum.Enum):
  | "mutateProb" // 1
  | "mutateLuckyRatio" // 2
  | "increaseTargetWithRatio" // 3
  | "increaseTargetRanged" // 4
  | "decreaseTurnLeft" // 5
  | "shuffleAll" // 6
  | "setEnchantTargetAndAmount" // 7
  | "unlockAndLockOther" // 8
  | "changeEffect" // 9
  | "lockTarget" // 10
  | "increaseReroll" // 11
  | "decreasePrice" // 12
  | "restart" // 13
  | "setEnchantIncreaseAmount" // 14
  | "setEnchantEffectCount" // 15
  | "setValueRanged" // 16
  | "redistributeAll" // 17
  | "redistributeSelectedToOthers" // 18
  | "shiftAll" // 19
  | "swapValues" // 20
  | "swapMinMax" // 23
  | "exhaust" // 24
  | "increaseMaxAndDecreaseTarget" // 25
  | "increaseMinAndDecreaseTarget" // 26
  | "redistributeMinToOthers" // 27
  | "redistributeMaxToOthers" // 28
  | "decreaseMaxAndSwapMinMax" // 29
  | "decreaseFirstTargetAndSwap"; // 30

"""


class CouncilTargetType(enum.Enum):
    none = "none"
    random = "random"
    proposed = "proposed"
    maxValue = "maxValue"
    minValue = "minValue"
    userSelect = "userSelect"
    lteValue = "lteValue"
    oneThreeFive = "oneThreeFive"
    twoFour = "twoFour"


class ElixirLogic(pydantic.BaseModel, metaclass=abc.ABCMeta):
    js_alias: str

    ratio: int
    value: tuple[int, int]
    remain_turn: int

    @abc.abstractmethod
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        ...


class Council(pydantic.BaseModel):
    id: str


class CouncilRepository:
    def get_council(self, council_id: str) -> Council:
        ...

    def sample(self, state) -> tuple[Council, Council, Council]:
        ...
