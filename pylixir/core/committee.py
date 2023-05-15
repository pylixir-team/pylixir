import enum

import pydantic


class SageType(enum.Enum):
    none = "none"
    lawful = "lawful"
    chaos = "chaos"


MAX_LAWFUL = 3
MAX_CHAOS = -6


class Sage(pydantic.BaseModel):
    power: int
    is_removed: bool
    slot: int

    def selected(self) -> None:
        if self.power < 0 or self.power == MAX_LAWFUL:
            self.power = 0

        self.power += 1

    def discarded(self) -> None:
        if self.power > 0 or self.power == MAX_CHAOS:
            self.power = 0

        self.power -= 1

    def is_lawful_max(self) -> bool:
        return self.power == MAX_LAWFUL

    def is_chaos_max(self) -> bool:
        return self.power == MAX_CHAOS


class SageCommittee(pydantic.BaseModel):
    sages: tuple[Sage, Sage, Sage]

    def set_exhaust(self, slot: int) -> None:
        self.sages[slot].is_removed = True

    def get_valid_slots(self) -> list[int]:
        return [idx for idx, sage in enumerate(self.sages) if not sage.is_removed]

    def pick(self, picked_slot: int) -> None:
        for sage in self.sages:
            if sage.slot == picked_slot:
                sage.selected()
            else:
                sage.discarded()
