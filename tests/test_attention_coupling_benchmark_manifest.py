# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the fixed Anima regional benchmark manifest and scoring contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.attention_coupling_benchmark.manifest import (
    DEFAULT_MANIFEST_PATH,
    REQUIRED_SCENARIOS,
    load_manifest,
    load_result_schema,
)


def test_manifest_pins_complete_anima_stack_and_generation_profile() -> None:
    """Pin the complete model stack and primary Anima generation settings."""

    manifest = load_manifest()

    assert manifest.schema_version == 1
    assert manifest.benchmark_id == "anima-regional-prompting-v1"
    assert manifest.simple_syrup_commit == "a5e53bd0ef3b089cfcccf54655520d03602836a6"
    assert manifest.comfyui_commit == "a1c421994cdcc5044dbce2bb7628e89386311cc5"
    assert {
        model.role: (model.filename, model.size_bytes, model.sha256)
        for model in manifest.models
    } == {
        "diffusion_model": (
            "diffusion-model.safetensors",
            4182218328,
            "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e",
        ),
        "text_encoder": (
            "qwen_3_06b_base.safetensors",
            1192135096,
            "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
        ),
        "vae": (
            "qwen_image_vae.safetensors",
            253806246,
            "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
        ),
    }
    assert (
        manifest.sampling.width,
        manifest.sampling.height,
        manifest.sampling.steps,
        manifest.sampling.cfg,
        manifest.sampling.sampler,
        manifest.sampling.scheduler,
        manifest.sampling.denoise,
    ) == (1024, 1024, 30, 4.0, "er_sde", "simple", 1.0)
    assert manifest.sampling.seeds == (1029384756, 3141592653, 2718281828)


def test_manifest_covers_every_required_scenario_with_fixed_geometry() -> None:
    """Keep every P0.5 scenario and its distinguishing geometry present."""

    manifest = load_manifest()
    cases_by_tag = {tag: case for case in manifest.cases for tag in case.scenario_tags}

    assert set(cases_by_tag) == REQUIRED_SCENARIOS
    assert len(manifest.cases) == 12
    assert cases_by_tag["no_region"].regional_prompts == ()
    assert cases_by_tag["horizontal_split"].masks[0].y1 == 0.5
    overlap = cases_by_tag["overlap"].masks
    assert overlap[0].x1 > overlap[1].x0
    uncovered = cases_by_tag["uncovered_pixels"].masks
    assert uncovered[0].x1 < uncovered[1].x0
    narrow = cases_by_tag["narrow_region"].masks[0]
    assert narrow.x1 - narrow.x0 == pytest.approx(0.1)
    assert "(black hair:3.0)" in cases_by_tag["high_anima_weights"].regional_prompts[0]
    assert len(cases_by_tag["long_prompts"].regional_prompts[0]) > 200


def test_manifest_expands_every_case_seed_and_execution_once() -> None:
    """Prevent seed cherry-picking or silent omission during run expansion."""

    manifest = load_manifest()
    runs = manifest.runs()

    assert len(manifest.executions) == 5
    assert {execution.spatial_mode for execution in manifest.executions} == {
        "full",
        "multidiffusion",
        "mixture_of_diffusers",
        "contextual",
    }
    executions = {
        execution.execution_id: execution for execution in manifest.executions
    }
    assert executions["regional-conditioning-full"].controls == {}
    assert executions["regional-conditioning-multidiffusion"].controls == {
        "tile_width": 64,
        "tile_height": 64,
        "overlap": 16,
        "tile_batch_size": 4,
    }
    assert executions["regional-conditioning-mod"].controls == {
        "tile_width": 64,
        "tile_height": 64,
        "overlap": 16,
        "tile_batch_size": 4,
    }
    assert executions["regional-conditioning-contextual-md"].controls == {
        "diffusion_mode": "multidiffusion",
        "latent_context_size": 64,
        "latent_context_overlap": 16,
        "latent_context_batch_size": 4,
        "global_weight": 1.0,
        "global_steps": 10,
        "global_decay": 0.5,
    }
    assert executions["regional-conditioning-contextual-mod"].controls == {
        "diffusion_mode": "mixture_of_diffusers",
        "latent_context_size": 64,
        "latent_context_overlap": 16,
        "latent_context_batch_size": 4,
        "global_weight": 1.0,
        "global_steps": 10,
        "global_decay": 0.5,
    }
    assert len(runs) == 12 * 3 * 5
    assert len({run.artifact_id for run in runs}) == len(runs)
    assert all(run.case_id in run.artifact_id for run in runs)
    assert all(run.execution_id in run.artifact_id for run in runs)
    assert all(f"seed-{run.seed}" in run.artifact_id for run in runs)


def test_result_schema_requires_measurements_scores_and_failure_classes() -> None:
    """Keep runtime evidence and visual failure scoring machine-readable."""

    schema = load_result_schema()
    definitions = _mapping(schema["$defs"])
    observation = _mapping(definitions["observation"])
    observation_required = set(_strings(observation["required"]))
    scores = _mapping(definitions["visual_scores"])
    score_required = set(_strings(scores["required"]))
    properties = _mapping(observation["properties"])
    failure_classes = _mapping(properties["failure_classes"])
    failure_items = _mapping(failure_classes["items"])

    assert observation_required >= {
        "runtime_ms",
        "peak_vram_bytes",
        "model_call_count",
        "cross_attention_branch_count",
        "scores",
        "failure_classes",
    }
    assert score_required == {
        "composition_integrity",
        "region_assignment",
        "boundary_coherence",
        "subject_integrity",
        "prompt_adherence",
    }
    assert set(_strings(failure_items["enum"])) >= {
        "subject_fusion",
        "identity_leakage",
        "attribute_swap",
        "boundary_seam",
        "incompatible_geometry",
        "region_nonadherence",
        "prompt_weight_instability",
        "long_prompt_instability",
        "tile_discontinuity",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"unexpected": True}), "fields differ"),
        (
            lambda payload: payload["sampling"].update(
                {"seeds": [1029384756, 1029384756]}
            ),
            "seeds must be unique",
        ),
        (
            lambda payload: payload["cases"].pop(),
            "missing required scenarios",
        ),
        (
            lambda payload: payload["cases"][0]["masks"][0].update({"x1": 1.1}),
            "coordinates must not exceed",
        ),
    ],
)
def test_manifest_rejects_incomplete_or_ambiguous_inputs(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    """Fail closed when tracked benchmark inputs lose determinism or coverage."""

    payload = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_manifest(path)


def _mapping(value: object) -> dict[str, object]:
    """Narrow a result-schema object for contract assertions."""

    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return value


def _strings(value: object) -> list[str]:
    """Narrow a result-schema string array for contract assertions."""

    assert isinstance(value, list)
    assert all(isinstance(item, str) for item in value)
    return value
