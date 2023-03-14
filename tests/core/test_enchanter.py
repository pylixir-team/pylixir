from pylixir.core.base import Enchanter


def test_enchant_amount(clean_enchanter: Enchanter) -> None:
    assert clean_enchanter.get_enchant_amount() == 1


def test_enchant_amount_with_irrelevant_mutation(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_prob(3, 0.05, 1)
    clean_enchanter.mutate_prob(1, -0.05, 1)
    clean_enchanter.mutate_lucky_ratio(0, 0.1, 1)

    assert clean_enchanter.get_enchant_amount() == 1


def test_enchant_amount_with_relevant_mutation(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_prob(3, 0.05, 1)
    clean_enchanter.mutate_prob(1, -0.05, 1)
    clean_enchanter.mutate_lucky_ratio(0, 0.1, 1)
    clean_enchanter.increase_enchant_amount(2)

    assert clean_enchanter.get_enchant_amount() == 2


def test_enchant_effect_count(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_prob(3, 0.05, 1)
    clean_enchanter.mutate_prob(1, -0.05, 1)
    clean_enchanter.mutate_lucky_ratio(0, 0.1, 1)

    assert clean_enchanter.get_enchant_effect_count() == 1


def test_enchant_effect_count_with_relevant_mutation(
    clean_enchanter: Enchanter,
) -> None:
    clean_enchanter.mutate_prob(3, 0.05, 1)
    clean_enchanter.change_enchant_effect_count(2)
    clean_enchanter.mutate_lucky_ratio(0, 0.1, 1)
    clean_enchanter.increase_enchant_amount(2)

    assert clean_enchanter.get_enchant_effect_count() == 2
