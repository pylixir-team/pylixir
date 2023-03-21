from typing import Type

from pylixir.application.council import ElixirOperation, TargetSelector


class ElixirOperationLoader:
    def __init__(self, operations: list[Type[ElixirOperation]]):
        self._operation_mappings: dict[str, Type[ElixirOperation]] = {
            cls.get_type(): cls for cls in operations
        }

    def get_operation(
        self, operation_type: str, ratio: int, value: tuple[int, int], remain_turn: int
    ) -> ElixirOperation:
        target_operation = self._get_operation_class(operation_type)
        return target_operation(ratio=ratio, value=value, remain_turn=remain_turn)

    def _get_operation_class(self, operation_type: str) -> Type[ElixirOperation]:
        return self._operation_mappings[operation_type]


class ElixirTargetSelectorLoader:
    def __init__(self, selector_mappings: dict[str, Type[TargetSelector]]):
        self._selector_mappings = selector_mappings

    def get_selector(
        self, selector_type: str, condition: int, count: int
    ) -> TargetSelector:
        target_operation = self._get_selector_class(selector_type)
        return target_operation(target_condition=condition, count=count)

    def _get_selector_class(self, selector_type: str) -> Type[TargetSelector]:
        return self._selector_mappings[selector_type]
