# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure the native Anima model with the established P5.7 tensor fixture."""

from __future__ import annotations

import json
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

import comfy.model_management
import comfy.sampler_helpers

from .artifacts import require_artifact, verify_artifact
from .environment import performance_environment
from .execution_fixture import build_attention_fixture, build_input_fixture
from .manifest import PerformanceManifest, PerformanceProfile
from .measurement import activate_measurement_model, execute_call_sequence
from .model_fixture import load_model_fixture
from .runtime_profile import PerformanceRuntimeProfile


def run_plain_anima_comparison(
    manifest: PerformanceManifest,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return native-model median evidence and the installed environment."""

    for artifact in manifest.artifacts:
        verify_artifact(artifact)
    model_artifact = require_artifact(manifest, "anima_base")
    lora_artifact = require_artifact(manifest, "regional_lora")
    fixture = load_model_fixture(model_artifact, lora_artifact)
    source: Any = fixture.source
    device = source.load_device
    try:
        activate_measurement_model(source)
        attention = build_attention_fixture(manifest, device=device)
        latent, context, sample_sigmas = build_input_fixture(
            manifest,
            attention,
            device=device,
        )
        model_options = deepcopy(source.model_options)
        comfy.sampler_helpers.prepare_model_patcher(source, {}, model_options)
        profile = PerformanceRuntimeProfile(
            PerformanceProfile("plain-anima", 0, 0.0),
            source,
            model_options,
            (),
            (),
            None,
        )
        activate_measurement_model(source)
        execute_call_sequence(
            profile,
            latent=latent,
            context=context,
            sample_sigmas=sample_sigmas,
            call_count=manifest.warmup_calls,
            measured=False,
        )
        measurements = tuple(
            execute_call_sequence(
                profile,
                latent=latent,
                context=context,
                sample_sigmas=sample_sigmas,
                call_count=manifest.denoiser_calls,
                measured=True,
            )
            for _repeat_index in range(manifest.repeats)
        )
        hashes = {measurement.output_sha256 for measurement in measurements}
        if len(hashes) != 1:
            raise ValueError("Plain Anima repeated outputs are not deterministic.")
        result: dict[str, object] = {
            "profile_id": "plain-anima",
            "denoiser_calls": manifest.denoiser_calls,
            "warmup_calls": manifest.warmup_calls,
            "repeats": manifest.repeats,
            "median_runtime_ms": statistics.median(
                measurement.runtime_ms for measurement in measurements
            ),
            "median_peak_vram_bytes": int(
                statistics.median(
                    measurement.peak_vram_bytes for measurement in measurements
                )
            ),
            "output_sha256": hashes.pop(),
            "observations": [
                {
                    "runtime_ms": measurement.runtime_ms,
                    "peak_vram_bytes": measurement.peak_vram_bytes,
                    "output_sha256": measurement.output_sha256,
                }
                for measurement in measurements
            ],
        }
        return result, performance_environment(device)
    finally:
        comfy.model_management.unload_all_models()


def write_plain_anima_result(
    path: Path,
    *,
    result: dict[str, object],
    environment: dict[str, object],
) -> None:
    """Atomically persist one complete native-model comparison."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"status": "completed", "result": result, "environment": environment},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
