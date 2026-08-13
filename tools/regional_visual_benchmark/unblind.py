# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Join a sealed P10.3 score ledger to protected corpus identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from tools.attention_coupling_benchmark.manifest_types import (
    BenchmarkManifest,
    JsonObject,
)

from .matrix import visual_positions


def unblind_scores(
    manifest: BenchmarkManifest,
    *,
    corpus_path: Path,
    packet_path: Path,
    mapping_path: Path,
    sealed_scores_path: Path,
) -> Path:
    """Unblind only a complete sealed ledger with intact packet and mapping."""

    corpus_path = corpus_path.resolve()
    packet_path = packet_path.resolve()
    mapping_path = mapping_path.resolve()
    sealed_scores_path = sealed_scores_path.resolve()
    packet = _read(packet_path, "blind packet")
    mapping_bytes = mapping_path.read_bytes()
    if packet.get("mapping_sha256") != _sha256(mapping_bytes):
        raise ValueError("P10.3 private mapping hash does not match the packet.")
    scores = _read(sealed_scores_path, "sealed score ledger")
    if scores.get("status") != "sealed":
        raise ValueError("P10.3 scores must be sealed before unblinding.")
    if scores.get("packet_sha256") != _sha256(packet_path.read_bytes()):
        raise ValueError("P10.3 sealed scores belong to a different blind packet.")
    mapping = _read_bytes(mapping_bytes, "private mapping")
    corpus = _read(corpus_path, "corpus result")

    assignments = {
        _text(item.get("opaque_id"), "opaque_id"): _text(
            item.get("artifact_id"), "artifact_id"
        )
        for item in (
            _object(value, "mapping assignment")
            for value in _array(mapping.get("assignments"), "mapping assignments")
        )
    }
    score_records = {
        _text(item.get("opaque_id"), "opaque_id"): item
        for item in (
            _object(value, "score record")
            for value in _array(scores.get("scores"), "score records")
        )
    }
    observations = {
        _text(item.get("artifact_id"), "artifact_id"): item
        for item in (
            _object(value, "corpus observation")
            for value in _array(corpus.get("observations"), "corpus observations")
        )
    }
    expected_artifacts = {
        position.artifact_id for position in visual_positions(manifest)
    }
    if set(assignments) != set(score_records):
        raise ValueError("P10.3 mapping and sealed score identities differ.")
    if (
        set(assignments.values()) != expected_artifacts
        or set(observations) != expected_artifacts
    ):
        raise ValueError("P10.3 unblinding inputs do not cover the frozen matrix.")

    unblinded: list[JsonObject] = []
    for opaque_id, artifact_id in assignments.items():
        observation = observations[artifact_id]
        unblinded.append(
            {
                "opaque_id": opaque_id,
                "artifact_id": artifact_id,
                "case_id": observation["case_id"],
                "seed": observation["seed"],
                "strategy": observation["strategy"],
                "spatial_profile": observation["spatial_profile"],
                "image_file": observation["image_file"],
                "image_sha256": observation["image_sha256"],
                "score": score_records[opaque_id],
            }
        )
    output = corpus_path.parent / "p10.3-unblinded-scores.json"
    _write(
        output,
        {
            "schema_version": 1,
            "benchmark_id": manifest.benchmark_id,
            "corpus_sha256": _sha256(corpus_path.read_bytes()),
            "packet_sha256": _sha256(packet_path.read_bytes()),
            "mapping_sha256": _sha256(mapping_bytes),
            "sealed_scores_sha256": _sha256(sealed_scores_path.read_bytes()),
            "observation_count": len(unblinded),
            "observations": unblinded,
        },
    )
    return output


def _read(path: Path, label: str) -> JsonObject:
    """Read and narrow one JSON object file."""

    return _read_bytes(path.read_bytes(), label)


def _read_bytes(value: bytes, label: str) -> JsonObject:
    """Decode and narrow one UTF-8 JSON object."""

    return _object(json.loads(value.decode("utf-8")), label)


def _write(path: Path, value: object) -> None:
    """Atomically write one stable unblinded result."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
