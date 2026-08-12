"""Stage a strategy-blind P10.3 image packet with protected mapping."""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from random import SystemRandom
from typing import cast

from tools.attention_coupling_benchmark.manifest_types import (
    BenchmarkManifest,
    JsonObject,
)

from .matrix import visual_positions


@dataclass(frozen=True, slots=True)
class BlindPacketArtifacts:
    """Expose public packet and private mapping paths without mapping contents."""

    packet_path: Path
    mapping_path: Path
    mapping_sha256: str


class BlindCorpusStager:
    """Assign opaque image identities and keep source association private."""

    def __init__(
        self,
        manifest: BenchmarkManifest,
        *,
        identity_factory: Callable[[], str] | None = None,
    ) -> None:
        """Retain the frozen case owner and a collision-resistant ID factory."""

        self._manifest = manifest
        self._identity_factory = identity_factory or (lambda: secrets.token_hex(16))

    def stage(self, corpus_path: Path) -> BlindPacketArtifacts:
        """Copy every corpus image under a random opaque identity and order."""

        corpus_path = corpus_path.resolve()
        root = corpus_path.parent
        blind_root = root / "blind"
        private_root = root / "private"
        if blind_root.exists():
            raise FileExistsError(f"P10.3 blind packet already exists: {blind_root}")
        blind_images = blind_root / "images"
        blind_images.mkdir(parents=True)
        private_root.mkdir(parents=True, exist_ok=True)
        mapping_path = private_root / "blind-mapping.json"
        if mapping_path.exists():
            raise FileExistsError(
                f"P10.3 private mapping already exists: {mapping_path}"
            )

        corpus = _object(
            json.loads(corpus_path.read_text(encoding="utf-8")), "corpus result"
        )
        if corpus.get("status") != "completed":
            raise ValueError("P10.3 can blind only a completed corpus.")
        observations = {
            _text(record.get("artifact_id"), "artifact_id"): record
            for record in (
                _object(value, "corpus observation")
                for value in _array(corpus.get("observations"), "corpus observations")
            )
        }
        expected = visual_positions(self._manifest)
        if set(observations) != {position.artifact_id for position in expected}:
            raise ValueError("P10.3 corpus identities do not match the frozen matrix.")

        identities = self._unique_identities(len(expected))
        assignments = list(zip(identities, expected, strict=True))
        SystemRandom().shuffle(assignments)
        mapping: list[JsonObject] = []
        packet_entries: list[JsonObject] = []
        cases = {case.case_id: case for case in self._manifest.cases}
        for opaque_id, position in assignments:
            observation = observations[position.artifact_id]
            source_path = root / _text(observation.get("image_file"), "image_file")
            expected_sha = _text(observation.get("image_sha256"), "image_sha256")
            if (
                not source_path.is_file()
                or _sha256(source_path.read_bytes()) != expected_sha
            ):
                raise ValueError(
                    f"P10.3 corpus image is missing or changed: {position.artifact_id}"
                )
            blind_relative = Path("images") / f"{opaque_id}.png"
            shutil.copyfile(source_path, blind_root / blind_relative)
            mapping.append(
                {"opaque_id": opaque_id, "artifact_id": position.artifact_id}
            )
            case = cases[position.case_id]
            packet_entries.append(
                {
                    "opaque_id": opaque_id,
                    "image_file": blind_relative.as_posix(),
                    "image_sha256": expected_sha,
                    "case_id": position.case_id,
                    "seed": position.seed,
                    "spatial_profile": position.spatial_profile.value,
                    "global_prompt": case.global_prompt,
                    "regional_prompts": list(case.regional_prompts),
                    "masks": [asdict(mask) for mask in case.masks],
                }
            )

        mapping_bytes = _json_bytes(
            {
                "schema_version": 1,
                "benchmark_id": self._manifest.benchmark_id,
                "assignments": mapping,
            }
        )
        mapping_path.write_bytes(mapping_bytes)
        mapping_sha256 = _sha256(mapping_bytes)
        packet_path = blind_root / "packet.json"
        packet_path.write_bytes(
            _json_bytes(
                {
                    "schema_version": 1,
                    "benchmark_id": self._manifest.benchmark_id,
                    "strategy_blind": True,
                    "mapping_sha256": mapping_sha256,
                    "entry_count": len(packet_entries),
                    "entries": packet_entries,
                }
            )
        )
        return BlindPacketArtifacts(packet_path, mapping_path, mapping_sha256)

    def _unique_identities(self, count: int) -> tuple[str, ...]:
        """Generate nonempty unique opaque IDs without strategy-derived content."""

        identities: list[str] = []
        observed: set[str] = set()
        while len(identities) < count:
            value = self._identity_factory()
            if not value or not value.isascii() or not value.isalnum():
                raise ValueError(
                    "P10.3 opaque identities must be nonempty ASCII alnum."
                )
            if value not in observed:
                identities.append(value)
                observed.add(value)
        return tuple(identities)


def _json_bytes(value: object) -> bytes:
    """Return stable UTF-8 JSON bytes for hashing and atomic identity."""

    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


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
