# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate and seal strategy-blind P10.3 human scores."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from tools.attention_coupling_benchmark.manifest_types import JsonObject

FAILURE_CLASSES = frozenset(
    {
        "subject_fusion",
        "identity_leakage",
        "attribute_swap",
        "boundary_seam",
        "incompatible_geometry",
        "missing_subject",
        "duplicate_subject",
        "region_nonadherence",
        "prompt_weight_instability",
        "long_prompt_instability",
        "tile_discontinuity",
        "other",
    }
)
SEVERE_FAILURE_CLASSES = frozenset(
    {
        "subject_fusion",
        "incompatible_geometry",
        "missing_subject",
        "duplicate_subject",
    }
)
RUBRIC: JsonObject = {
    "subject_count": "0 wrong/unusable; 4 exact intended subject count.",
    "identity_separation": "0 fused/indistinguishable; 4 distinct intended identities.",
    "attribute_isolation": "0 reversed/pervasive leakage; 4 attributes stay assigned.",
    "anatomical_integrity": "0 fused/invalid anatomy; 4 coherent subjects and anatomy.",
    "pose_continuity": "0 broken/incompatible pose; 4 continuous intended pose.",
    "boundary_integrity": "0 destructive seam; 4 natural boundary without artifacts.",
    "global_composition": "0 unusable layout; 4 coherent intended composition.",
    "style_consistency": "0 contradictory/collapsed style; 4 coherent prompt style.",
    "failure_severity": "0 no material failure; 4 wholly unusable severe failure.",
}


@dataclass(frozen=True, slots=True)
class VisualScore:
    """Hold one complete blind review under the fixed nine-category rubric."""

    subject_count: int
    identity_separation: int
    attribute_isolation: int
    anatomical_integrity: int
    pose_continuity: int
    boundary_integrity: int
    global_composition: int
    style_consistency: int
    failure_severity: int
    failure_classes: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        """Fail closed on incomplete ranges, duplicate classes, or unknown labels."""

        for field, value in asdict(self).items():
            if field in {"failure_classes", "notes"}:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 4
            ):
                raise ValueError(f"P10.3 score {field} must be an integer from 0 to 4.")
        if len(set(self.failure_classes)) != len(self.failure_classes):
            raise ValueError("P10.3 failure classes must be unique.")
        unknown = set(self.failure_classes) - FAILURE_CLASSES
        if unknown:
            raise ValueError(f"P10.3 unknown failure classes: {sorted(unknown)}")

    @property
    def severe_failure(self) -> bool:
        """Apply the fixed P10.3 severe-failure definition."""

        return (
            self.failure_severity >= 3
            or self.subject_count == 0
            or self.identity_separation == 0
            or self.anatomical_integrity == 0
            or bool(set(self.failure_classes) & SEVERE_FAILURE_CLASSES)
        )


class BlindScoreLedger:
    """Persist one score per opaque packet identity and seal only completeness."""

    def __init__(self, packet_path: Path) -> None:
        """Load one strategy-blind packet without access to its private mapping."""

        self._packet_path = packet_path.resolve()
        self._root = self._packet_path.parent
        self._journal = self._root / "scores.inprogress.json"
        self._sealed = self._root / "scores.sealed.json"
        packet = _object(
            json.loads(self._packet_path.read_text(encoding="utf-8")), "blind packet"
        )
        if packet.get("strategy_blind") is not True:
            raise ValueError("P10.3 scoring requires a strategy-blind packet.")
        self._benchmark_id = _text(packet.get("benchmark_id"), "benchmark_id")
        self._opaque_ids = tuple(
            _text(_object(value, "packet entry").get("opaque_id"), "opaque_id")
            for value in _array(packet.get("entries"), "packet entries")
        )
        if len(set(self._opaque_ids)) != len(self._opaque_ids):
            raise ValueError("P10.3 blind packet contains duplicate opaque IDs.")
        self._scores: dict[str, JsonObject] = {}
        if self._journal.exists():
            self._resume()

    @property
    def remaining_count(self) -> int:
        """Return the number of packet identities still requiring a score."""

        return len(set(self._opaque_ids) - set(self._scores))

    @property
    def next_opaque_id(self) -> str | None:
        """Return the next unscored identity in blind presentation order."""

        return next(
            (
                opaque_id
                for opaque_id in self._opaque_ids
                if opaque_id not in self._scores
            ),
            None,
        )

    def record(self, opaque_id: str, score: VisualScore) -> None:
        """Atomically record one complete score without replacing prior judgment."""

        if opaque_id not in self._opaque_ids:
            raise KeyError(f"P10.3 opaque identity is not in the packet: {opaque_id}")
        if opaque_id in self._scores:
            raise ValueError(f"P10.3 opaque identity was already scored: {opaque_id}")
        record: JsonObject = {
            "opaque_id": opaque_id,
            **asdict(score),
            "failure_classes": list(score.failure_classes),
            "severe_failure": score.severe_failure,
        }
        self._scores[opaque_id] = record
        self._persist(self._journal, status="scoring")

    def seal(self) -> Path:
        """Seal scores only after every opaque identity has one judgment."""

        if set(self._scores) != set(self._opaque_ids):
            raise ValueError(
                f"P10.3 scores are incomplete: remaining={self.remaining_count}."
            )
        self._persist(self._sealed, status="sealed")
        self._journal.unlink(missing_ok=True)
        return self._sealed

    def _resume(self) -> None:
        """Restore scores only from this exact packet identity."""

        payload = _object(
            json.loads(self._journal.read_text(encoding="utf-8")), "score journal"
        )
        if payload.get("packet_sha256") != _sha256(self._packet_path.read_bytes()):
            raise ValueError("P10.3 score journal belongs to a different packet.")
        for value in _array(payload.get("scores"), "score records"):
            record = _object(value, "score record")
            self._scores[_text(record.get("opaque_id"), "opaque_id")] = record

    def _persist(self, path: Path, *, status: str) -> None:
        """Atomically persist scores in packet presentation order."""

        payload: JsonObject = {
            "schema_version": 1,
            "benchmark_id": self._benchmark_id,
            "status": status,
            "packet_sha256": _sha256(self._packet_path.read_bytes()),
            "rubric": RUBRIC,
            "score_count": len(self._scores),
            "scores": [
                self._scores[opaque_id]
                for opaque_id in self._opaque_ids
                if opaque_id in self._scores
            ],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _sha256(value: bytes) -> str:
    """Return one lowercase byte identity."""

    return hashlib.sha256(value).hexdigest()


def _object(value: object, label: str) -> JsonObject:
    """Narrow one JSON object."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"P10.3 {label} must be an object.")
    return cast(JsonObject, value)


def _array(value: object, label: str) -> list[object]:
    """Narrow one JSON array."""

    if not isinstance(value, list):
        raise TypeError(f"P10.3 {label} must be an array.")
    return value


def _text(value: object, label: str) -> str:
    """Narrow one required nonempty string."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"P10.3 {label} must be a nonempty string.")
    return value
