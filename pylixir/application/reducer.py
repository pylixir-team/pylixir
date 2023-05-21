import pydantic

from pylixir.application.enchant import EnchantCommand
from pylixir.application.service import CouncilPool
from pylixir.core.base import Randomness
from pylixir.core.state import GameState


class Action(pydantic.BaseModel):
    ...


class PickCouncilAndEnchantAndRerollAction(Action):
    effect_index: int
    sage_index: int


def pick_council(
    action: PickCouncilAndEnchantAndRerollAction,
    state: GameState,
    randomness: Randomness,
    council_pool: CouncilPool,
) -> GameState:
    council_query = state.suggestions[action.sage_index]
    council = council_pool.get_council(council_query)

    try:
        for logic in council.logics:
            state = logic.apply(
                state,
                action.effect_index,
                randomness,
            )
    except Exception as e:
        # print(council.descriptions[action.sage_index])
        raise e

    state.committee.pick(action.sage_index)

    enchanted_result = EnchantCommand().enchant(state, randomness)

    for idx, amount in enumerate(enchanted_result):
        state.board.modify_effect_count(idx, amount)

    state.enchanter.get_enchant_effect_count()
    state.progress.spent_turn(1)
    state.enchanter.elapse_turn()

    state.suggestions = council_pool.get_council_queries(
        state,
        randomness=randomness,
        is_reroll=False,
    )

    return state


def reroll(
    state: GameState,
    randomness: Randomness,
    council_pool: CouncilPool,
) -> GameState:
    state.suggestions = council_pool.get_council_queries(
        state,
        randomness=randomness,
        is_reroll=True,
    )
    state.progress.spent_reroll()

    return state
