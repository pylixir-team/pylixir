import pydantic

from pylixir.core.base import Randomness
from pylixir.core.state import GameState


class EnchantCommand(pydantic.BaseModel):
    size: int = 5

    def enchant(self, state: GameState, randomness: Randomness) -> list[int]:
        locked = state.board.locked_indices()

        enchant_prob = state.enchanter.query_enchant_prob(locked)
        lucky_ratio = state.enchanter.query_lucky_ratio()
        enchant_effect_count = state.enchanter.get_enchant_effect_count()
        enchant_amount = state.enchanter.get_enchant_amount()

        result = self.get_enchant_result(
            enchant_prob,
            lucky_ratio,
            enchant_effect_count,
            enchant_amount,
            randomness,
        )

        return result

    def get_enchant_result(
        self,
        prob: list[float],
        lucky_ratio: list[float],
        count: int,
        amount: int,
        randomness: Randomness,
    ) -> list[int]:
        masked_prob = list(prob)
        result = [0 for _ in range(self.size)]

        for _ in range(count):
            if sum(masked_prob) == 0: ## 2-enchant given, but only one available
                break

            target_index = randomness.weighted_sampling(masked_prob)

            # add result as amount
            result[target_index] += amount
            if randomness.binomial(lucky_ratio[target_index]):
                result[target_index] += 1

            # pick and prevent duplicated sampling
            masked_prob[target_index] = 0

        return result
