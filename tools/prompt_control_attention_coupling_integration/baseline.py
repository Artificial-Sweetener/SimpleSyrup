# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load and revalidate the authoritative P0.8 Prompt Control baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.comfy_integration.default_paths import (
    default_benchmark_artifact_root,
)
from tools.prompt_control_characterization.cases import PromptControlCase, cases
from tools.prompt_control_characterization.evidence_validation import validate_evidence
from tools.prompt_control_characterization.history_outputs import PromptControlOutputs
from tools.prompt_control_characterization.source_identity import (
    PINNED_TRACKED_PATH_COUNT,
    PINNED_TRACKED_TREE_SHA256,
    PINNED_VERSION,
)

DEFAULT_BASELINE_PATH = default_benchmark_artifact_root(
    "anima-regional-prompting-v1/p0.8/result.json"
)


@dataclass(frozen=True, slots=True)
class PromptControlBaselineObservation:
    """Retain one already-revalidated P0.8 observation."""

    case: PromptControlCase
    expansion: JsonObject
    snapshot: JsonObject
    runtime: JsonObject


@dataclass(frozen=True, slots=True)
class PromptControlBaseline:
    """Expose immutable validated observations and artifact identity."""

    path: Path
    sha256: str
    observations: tuple[PromptControlBaselineObservation, ...]

    def observation(self, case_id: str) -> PromptControlBaselineObservation:
        """Return the sole validated observation for one case identity."""

        matches = tuple(
            observation
            for observation in self.observations
            if observation.case.case_id == case_id
        )
        if len(matches) != 1:
            raise ValueError(f"P0.8 baseline case is not unique: {case_id!r}.")
        return matches[0]


def load_baseline(path: Path = DEFAULT_BASELINE_PATH) -> PromptControlBaseline:
    """Read, identify, and fully revalidate the complete P0.8 artifact."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"P0.8 baseline result is missing: {resolved}.")
    data = resolved.read_bytes()
    payload = _object(json.loads(data), "P0.8 result")
    if payload.get("benchmark_id") != "prompt-control-characterization-v1":
        raise ValueError("P0.8 baseline benchmark identity changed.")
    source = _object(payload.get("prompt_control_source"), "prompt_control_source")
    if source != {
        "version": PINNED_VERSION,
        "tracked_path_count": PINNED_TRACKED_PATH_COUNT,
        "tracked_tree_sha256": PINNED_TRACKED_TREE_SHA256,
    }:
        raise ValueError("P0.8 baseline Prompt Control source identity changed.")
    raw_observations = _array(payload.get("observations"), "observations")
    definitions = cases()
    if len(raw_observations) != len(definitions):
        raise ValueError("P0.8 baseline matrix cardinality changed.")
    observations: list[PromptControlBaselineObservation] = []
    for case, raw in zip(definitions, raw_observations, strict=True):
        observation = _object(raw, f"observation {case.case_id}")
        if (
            observation.get("case_id") != case.case_id
            or observation.get("status") != "completed"
            or observation.get("error") is not None
        ):
            raise ValueError("P0.8 baseline case order or status changed.")
        outputs = PromptControlOutputs(
            _object(observation.get("expansion"), "baseline expansion"),
            _object(observation.get("snapshot"), "baseline snapshot"),
            _object(observation.get("runtime"), "baseline runtime"),
        )
        validate_evidence(case, outputs)
        observations.append(
            PromptControlBaselineObservation(
                case,
                outputs.expansion,
                outputs.snapshot,
                outputs.runtime,
            )
        )
    return PromptControlBaseline(
        resolved,
        hashlib.sha256(data).hexdigest(),
        tuple(observations),
    )


def _array(value: object, field: str) -> list[object]:
    """Narrow one JSON array."""

    if not isinstance(value, list):
        raise TypeError(f"P0.8 baseline {field} must be an array.")
    return value


def _object(value: object, field: str) -> JsonObject:
    """Narrow one JSON object with string keys."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"P0.8 baseline {field} must be an object.")
    return value
