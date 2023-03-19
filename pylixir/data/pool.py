from __future__ import annotations

import json
import os
from pathlib import Path

import pydantic

from pylixir.application.council import Council, CouncilPool, Logic
from pylixir.data.council.operation import get_operation_classes
from pylixir.data.council.target import get_target_classes
from pylixir.data.loader import ElixirOperationLoader, ElixirTargetSelectorLoader


class LogicQuery(pydantic.BaseModel):
    type: str
    targetType: str
    targetCondition: int
    targetCount: int
    ratio: int
    value: tuple[int, int]
    remainTurn: int

    class Config:
        extra = "forbid"


class CouncilQuery(pydantic.BaseModel):
    id: str
    pickupRatio: int
    range: tuple[int, int]
    descriptions: tuple[str, str, str]
    slotType: int
    type: str
    applyLimit: int
    logics: list[LogicQuery]
    applyImmediately: bool

    class Config:
        extra = "forbid"


class CouncilLoader:
    def __init__(
        self,
        operation_loader: ElixirOperationLoader,
        selector_loader: ElixirTargetSelectorLoader,
    ):
        self._operation_loader = operation_loader
        self._selector_loader = selector_loader

    def get_council(self, query: CouncilQuery) -> Council:
        return Council(
            id=query.id,
            logics=[self._get_logic(logic_query) for logic_query in query.logics],
            pickup_ratio=query.pickupRatio,
            turn_range=query.range,
            slot_type=query.slotType,
            descriptions=query.descriptions,
            type=query.type,
        )

    def _get_logic(self, logic_query: LogicQuery) -> Logic:
        operation = self._operation_loader.get_operation(
            logic_query.type,
            logic_query.ratio,
            logic_query.value,
            logic_query.remainTurn,
        )
        target_selector = self._selector_loader.get_selector(
            logic_query.targetType,
            logic_query.targetCondition,
            logic_query.targetCount,
        )
        return Logic(
            operation=operation,
            target_selector=target_selector,
        )


def _get_pool_from_file_and_loader(
    resource_file_path: str, council_loader: CouncilLoader, skip: bool
) -> CouncilPool:
    with open(resource_file_path, encoding="utf-8") as f:
        raws = json.load(f)

    councils = []
    for raw in raws:
        query = CouncilQuery.parse_obj(raw)
        try:
            councils.append(council_loader.get_council(query))
        except KeyError as e:
            if skip:
                continue

            raise e

    return CouncilPool(councils)


def get_ingame_resource_path() -> str:
    return str(Path(os.path.dirname(__file__)) / "resource" / "council.json")


def get_ingame_council_pool(skip: bool = False) -> CouncilPool:
    return _get_pool_from_file_and_loader(
        get_ingame_resource_path(),
        CouncilLoader(
            ElixirOperationLoader(get_operation_classes()),
            ElixirTargetSelectorLoader(get_target_classes()),
        ),
        skip=skip,
    )
