# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load and semantically validate the Attention Coupling benchmark manifest."""

from __future__ import annotations

import json
from collections.abc import Hashable, Iterable
from pathlib import Path

from .manifest_decode import decode_manifest, json_object
from .manifest_types import BenchmarkManifest, JsonObject

FIXTURE_DIRECTORY = Path(__file__).with_name("fixtures")
DEFAULT_MANIFEST_PATH = FIXTURE_DIRECTORY / "manifest.json"
DEFAULT_RESULT_SCHEMA_PATH = FIXTURE_DIRECTORY / "result.schema.json"
REQUIRED_SCENARIOS = frozenset(
    {
        "hard_50_50",
        "horizontal_split",
        "asymmetric_regions",
        "boundary_crossing",
        "identical_prompts",
        "no_region",
        "short_prompts",
        "long_prompts",
        "high_anima_weights",
        "overlap",
        "uncovered_pixels",
        "narrow_region",
    }
)
SUPPORTED_SPATIAL_MODES = frozenset(
    {"full", "multidiffusion", "mixture_of_diffusers", "contextual"}
)


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> BenchmarkManifest:
    """Load the versioned manifest and reject incomplete or ambiguous inputs."""

    manifest = decode_manifest(_load_json(path, "benchmark manifest"))
    _validate_manifest(manifest)
    return manifest


def load_result_schema(path: Path = DEFAULT_RESULT_SCHEMA_PATH) -> JsonObject:
    """Load the tracked result schema for benchmark writers and reviewers."""

    return json_object(_load_json(path, "benchmark result schema"), "result schema")


def _load_json(path: Path, description: str) -> object:
    """Read one JSON artifact with actionable failure context."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load {description} '{path}': {error}") from error


def _validate_manifest(manifest: BenchmarkManifest) -> None:
    """Enforce completeness and deterministic expansion invariants."""

    if manifest.schema_version != 1:
        raise ValueError("Only benchmark manifest schema_version 1 is supported.")
    _unique((model.artifact_id for model in manifest.models), "model IDs")
    _unique((case.case_id for case in manifest.cases), "case IDs")
    _unique((item.execution_id for item in manifest.executions), "execution IDs")
    _unique(manifest.sampling.seeds, "seeds")
    if len(manifest.sampling.seeds) < 2:
        raise ValueError("The benchmark requires multiple fixed seeds.")
    if manifest.sampling.width % 8 or manifest.sampling.height % 8:
        raise ValueError("Benchmark dimensions must be divisible by 8 pixels.")
    if not 0.0 <= manifest.sampling.denoise <= 1.0:
        raise ValueError("sampling.denoise must be between 0 and 1.")
    if any(
        item.spatial_mode not in SUPPORTED_SPATIAL_MODES for item in manifest.executions
    ):
        raise ValueError("An execution uses an unsupported spatial mode.")
    covered = frozenset(tag for case in manifest.cases for tag in case.scenario_tags)
    missing = sorted(REQUIRED_SCENARIOS - covered)
    if missing:
        raise ValueError(f"Benchmark cases are missing required scenarios: {missing}.")
    for case in manifest.cases:
        if not 0.0 <= case.regional_prompt_weight <= 1.0:
            raise ValueError(f"Case '{case.case_id}' has an invalid regional weight.")
        if "no_region" in case.scenario_tags and (case.regional_prompts or case.masks):
            raise ValueError("The no_region case must not contain regional inputs.")
    _validate_artifact_template(manifest)
    _unique((run.artifact_id for run in manifest.runs()), "expanded artifact IDs")


def _validate_artifact_template(manifest: BenchmarkManifest) -> None:
    """Prove the artifact template uses all identities and is Windows-safe."""

    required_fields = {"benchmark_id", "case_id", "execution_id", "seed"}
    try:
        rendered = manifest.artifact_name_template.format(
            benchmark_id="benchmark",
            case_id="case",
            execution_id="execution",
            seed=1,
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            "artifact_name_template contains an unsupported field."
        ) from error
    if any(
        f"{{{field}}}" not in manifest.artifact_name_template
        for field in required_fields
    ):
        raise ValueError("artifact_name_template must include every identity field.")
    if not rendered or any(character in rendered for character in '<>:"/\\|?*'):
        raise ValueError("artifact_name_template produces an unsafe Windows filename.")


def _unique(values: Iterable[Hashable], label: str) -> None:
    """Reject duplicate values from one finite iterable."""

    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{label} must be unique.")
