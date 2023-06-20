from typing import Any, Generator

from pydantic import BaseModel

from pylixir.data.pool import CouncilMeta, LogicMeta


def _as_index_map(value_set: set) -> dict[str, int]:
    ordered_list = sorted(value_set)
    return {k: idx + 1 for idx, k in enumerate(ordered_list)}


def _all_logics(
    metadata_map: dict[str, CouncilMeta]
) -> Generator[LogicMeta, None, None]:
    for meta in metadata_map.values():
        for logic in meta.logics:
            yield logic


class CouncilFeatureBuilder(BaseModel):
    council_id: dict[str, int]
    council_pickupRatio: dict[str, int]
    council_range: dict[str, int]
    council_slotType: dict[str, int]
    council_type: dict[str, int]

    logic_type: dict[str, int]
    logic_targetType: dict[str, int]
    logic_targetCondition: dict[int, int]
    logic_targetCount: dict[int, int]
    logic_ratio: dict[int, int]
    logic_remainTurn: dict[int, int]

    def get_indexed_vector(self, meta: CouncilMeta) -> tuple[list[int], list[float]]:
        council_features = {
            "id": self.council_id[meta.id],
            "pickupRatio": self.council_pickupRatio[meta.pickupRatio],
            "range0": self.council_range[meta.range[0]],
            "range1": self.council_range[meta.range[1]],
            "slotType": self.council_range[meta.slotType],
            "type": self.council_range[meta.type],
            "applyLimit": self.council_range[meta.applyLimit],
            "applyImmediately": int(meta.applyImmediately),
        }

        for idx in range(2):
            if idx < len(meta.logics):
                logic_feature = self._get_logic_input(meta.logics[idx], idx)
            else:
                logic_feature = self._get_empty_logic_input(idx)

            council_features.update(logic_feature)

        return council_features

    def _get_logic_input(self, logic_meta: LogicMeta, logic_index: int) -> dict:
        return {
            f"logic_{logic_index}_type": self.logic_type[logic_meta.type],
            f"logic_{logic_index}_targetType": self.logic_type[logic_meta.targetType],
            f"logic_{logic_index}_targetCondition": self.logic_type[
                logic_meta.targetCondition
            ],
            f"logic_{logic_index}_targetCount": self.logic_type[logic_meta.targetCount],
            f"logic_{logic_index}_ratio": self.logic_type[logic_meta.ratio],
            f"logic_{logic_index}_remainTurn": self.logic_type[logic_meta.remainTurn],
        }

    def _get_empty_logic_input(self, logic_index: int) -> dict:
        return {
            f"logic_{logic_index}_type": 0,
            f"logic_{logic_index}_targetType": 0,
            f"logic_{logic_index}_targetCondition": 0,
            f"logic_{logic_index}_targetCount": 0,
            f"logic_{logic_index}_ratio": 0,
            f"logic_{logic_index}_remainTurn": 0,
        }


def _get_metadata_index_maps(metadata_map: dict[str, CouncilMeta]):
    return CouncilFeatureBuilder(
        council_id=_as_index_map(set([meta.id for meta in metadata_map.values()])),
        council_pickupRatio=_as_index_map(
            set([meta.pickupRatio for meta in metadata_map.values()])
        ),
        council_range=_as_index_map(
            set(
                [meta.range[0] for meta in metadata_map.values()]
                + [meta.range[1] for meta in metadata_map.values()]
            )
        ),
        council_slotType=_as_index_map(
            set([meta.slotType for meta in metadata_map.values()])
        ),
        council_type=_as_index_map(set([meta.type for meta in metadata_map.values()])),
        logic_type_index=_as_index_map(
            set([logic.type for logic in _all_logics(metadata_map)])
        ),
        logic_target_type_index=_as_index_map(
            set([logic.targetType for logic in _all_logics(metadata_map)])
        ),
    )
