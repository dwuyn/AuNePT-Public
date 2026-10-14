"""``G`` — pending evidence-acquisition subgoals (Figure 1, block E/D).

The figure routes ``unresolved prerequisites`` and unknown product identity
into a subgoal store instead of discarding the candidate. The store is
serializable so it can live in ``PentestState`` next to the candidate queue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class SubgoalKind(str, Enum):
    IDENTIFY_PRODUCT = "IdentifyProduct"
    OBTAIN_VERSION = "ObtainVersion"
    ACQUIRE_SESSION = "AcquireSession"
    NETWORK_PREREQ = "NetworkPrereq"
    GENERIC = "Generic"


@dataclass(frozen=True)
class Subgoal:
    kind: SubgoalKind
    key: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "key": self.key, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Subgoal":
        raw = str(data.get("kind") or SubgoalKind.GENERIC.value)
        try:
            kind = SubgoalKind(raw)
        except ValueError:
            kind = SubgoalKind.GENERIC
        return cls(kind=kind, key=str(data.get("key") or ""), detail=str(data.get("detail") or ""))


class SubgoalQueue:
    """Deduplicating FIFO of pending subgoals (``G`` in Figure 1)."""

    def __init__(self, items: Iterable[Subgoal] = ()) -> None:
        self._items: list[Subgoal] = []
        self._seen: set[tuple[SubgoalKind, str]] = set()
        for item in items:
            self.push(item)

    def push(self, subgoal: Subgoal) -> bool:
        token = (subgoal.kind, subgoal.key)
        if token in self._seen:
            return False
        self._seen.add(token)
        self._items.append(subgoal)
        return True

    def pop(self) -> Subgoal | None:
        if not self._items:
            return None
        item = self._items.pop(0)
        self._seen.discard((item.kind, item.key))
        return item

    def pending(self) -> list[Subgoal]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def to_list(self) -> list[dict[str, str]]:
        return [item.to_dict() for item in self._items]

    @classmethod
    def from_list(cls, data: Iterable[Mapping[str, Any]] | None) -> "SubgoalQueue":
        return cls(Subgoal.from_dict(item) for item in (data or []))


def subgoals_from_preconditions(
    missing: Sequence[str] | None,
    *,
    service_key: str = "",
) -> list[Subgoal]:
    """Translate missing-precondition labels into figure-style subgoals."""
    out: list[Subgoal] = []
    for raw in missing or ():
        text = str(raw or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if "version" in lowered:
            out.append(Subgoal(SubgoalKind.OBTAIN_VERSION, service_key, text))
        elif "product" in lowered or "fingerprint" in lowered:
            out.append(Subgoal(SubgoalKind.IDENTIFY_PRODUCT, service_key, text))
        elif "auth" in lowered or "session" in lowered or "credential" in lowered:
            out.append(Subgoal(SubgoalKind.ACQUIRE_SESSION, service_key, text))
        elif "network" in lowered or "route" in lowered:
            out.append(Subgoal(SubgoalKind.NETWORK_PREREQ, service_key, text))
        else:
            out.append(Subgoal(SubgoalKind.GENERIC, f"{service_key}:{text}", text))
    return out
