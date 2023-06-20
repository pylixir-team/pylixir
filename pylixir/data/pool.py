from __future__ import annotations

import json
import os
from pathlib import Path

import pydantic

from pylixir.application.council import Council, CouncilType, Logic
from pylixir.data.council.operation import get_operation_classes
from pylixir.data.council.target import get_target_classes
from pylixir.data.council_pool import ConcreteCouncilPool
from pylixir.data.loader import ElixirOperationLoader, ElixirTargetSelectorLoader


class LogicMeta(pydantic.BaseModel):
    type: str
    targetType: str
    targetCondition: int
    targetCount: int
    ratio: int
    value: tuple[int, int]
    remainTurn: int

    class Config:
        extra = "forbid"


class CouncilMeta(pydantic.BaseModel):
    id: str
    pickupRatio: int
    range: tuple[int, int]
    descriptions: tuple[str, str, str]
    slotType: int
    type: str
    applyLimit: int
    logics: list[LogicMeta]
    applyImmediately: bool

    class Config:
        extra = "forbid"


def get_metadatas(resource_file_path) -> dict[str, CouncilMeta]:
    metas: dict[str, CouncilMeta] = {}

    with open(resource_file_path, encoding="utf-8") as f:
        raws = json.load(f)

    for raw in raws:
        meta = CouncilMeta.parse_obj(raw)
        metas[meta.id] = meta

    return metas


class CouncilLoader:
    def __init__(
        self,
        operation_loader: ElixirOperationLoader,
        selector_loader: ElixirTargetSelectorLoader,
    ):
        self._operation_loader = operation_loader
        self._selector_loader = selector_loader

    def get_council(self, meta: CouncilMeta) -> Council:
        return Council(
            id=meta.id,
            logics=[self._get_logic(logic_meta) for logic_meta in meta.logics],
            pickup_ratio=meta.pickupRatio,
            turn_range=meta.range,
            slot_type=meta.slotType,
            descriptions=list(meta.descriptions),
            type=CouncilType(meta.type),
        )

    def _get_logic(self, meta: LogicMeta) -> Logic:
        operation = self._operation_loader.get_operation(
            meta.type,
            meta.ratio,
            meta.value,
            meta.remainTurn,
        )
        target_selector = self._selector_loader.get_selector(
            meta.targetType,
            meta.targetCondition,
            meta.targetCount,
        )
        return Logic(
            operation=operation,
            target_selector=target_selector,
        )


def _get_pool_from_file_and_loader(
    resource_file_path: str, council_loader: CouncilLoader, skip: bool
) -> ConcreteCouncilPool:
    with open(resource_file_path, encoding="utf-8") as f:
        raws = json.load(f)

    councils = []
    for raw in raws:
        meta = CouncilMeta.parse_obj(raw)
        try:
            councils.append(council_loader.get_council(meta))
        except KeyError as e:
            if skip:
                continue

            raise e

    return ConcreteCouncilPool(councils)


def get_ingame_resource_path() -> str:
    return str(Path(os.path.dirname(__file__)) / "resource" / "council.json")


def get_ingame_council_pool(skip: bool = False) -> ConcreteCouncilPool:
    return _get_pool_from_file_and_loader(
        get_ingame_resource_path(),
        CouncilLoader(
            ElixirOperationLoader(get_operation_classes()),
            ElixirTargetSelectorLoader(get_target_classes()),
        ),
        skip=skip,
    )
