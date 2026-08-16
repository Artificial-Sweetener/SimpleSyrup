# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure and persist SDXL no-LoRA comparison image evidence."""

from __future__ import annotations

import hashlib
import time
from io import BytesIO

from PIL import Image

from tools.comfy_api import JsonObject, LoopbackComfyClient
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.history_output import extract_saved_image
from tools.sdxl_attention_coupling_integration.evidence_validation import (
    sdxl_image_evidence,
    validate_sdxl_diagnostics,
    validate_sdxl_metrics,
)
from tools.sdxl_attention_coupling_integration.visual_history import (
    decode_sdxl_visual_history,
)
from tools.sdxl_attention_coupling_integration.visual_workflow import (
    BuiltSdxlVisualWorkflow,
)
from tools.sdxl_full_strength_lora_fidelity.plain_workflow import (
    BuiltPlainKSamplerWorkflow,
)


class SdxlNoLoraMeasurementRecorder:
    """Own measured execution, image validation, and artifact persistence."""

    def __init__(self, artifacts: IntegrationArtifacts) -> None:
        """Retain the managed artifact directory authority."""

        if not isinstance(artifacts, IntegrationArtifacts):
            raise TypeError("No-LoRA measurement requires integration artifacts.")
        self._artifacts = artifacts

    def measure_plain(
        self,
        client: LoopbackComfyClient,
        *,
        workflow: BuiltPlainKSamplerWorkflow,
        prompt_timeout: float,
    ) -> JsonObject:
        """Measure and persist the ordinary KSampler artifact."""

        if not isinstance(workflow, BuiltPlainKSamplerWorkflow):
            raise TypeError("Plain no-LoRA measurement requires its workflow type.")
        started = time.perf_counter()
        prompt_id = client.submit(workflow.prompt)
        history = client.wait_for_history(prompt_id, timeout=prompt_timeout)
        wall_ms = (time.perf_counter() - started) * 1000.0
        image_bytes = client.download_image(
            extract_saved_image(history, workflow.save_node_id)
        )
        metrics = validate_sdxl_metrics(
            _single_output(history, workflow.metrics_node_id, "benchmark_metrics"),
            "plain-no-lora",
        )
        return self._save(
            name="plain-no-lora.png",
            label="Plain KSampler — combined pink/black prompt — no LoRA",
            prompt_id=prompt_id,
            image_bytes=image_bytes,
            metrics=metrics,
            wall_runtime_ms=wall_ms,
        )

    def measure_regional(
        self,
        client: LoopbackComfyClient,
        *,
        workflow: BuiltSdxlVisualWorkflow,
        prompt_timeout: float,
    ) -> JsonObject:
        """Measure and persist the SimpleSyrup regional artifact."""

        if not isinstance(workflow, BuiltSdxlVisualWorkflow):
            raise TypeError("Regional no-LoRA measurement requires its workflow type.")
        started = time.perf_counter()
        prompt_id = client.submit(workflow.prompt)
        history = client.wait_for_history(prompt_id, timeout=prompt_timeout)
        wall_ms = (time.perf_counter() - started) * 1000.0
        evidence = decode_sdxl_visual_history(history, workflow)
        if len(workflow.outputs) != 1 or len(evidence) != 1:
            raise ValueError("Regional no-LoRA measurement requires one image.")
        output = workflow.outputs[0]
        terminal = evidence[output.artifact_id]
        metrics = validate_sdxl_metrics(terminal.metrics, "regional-no-lora")
        validate_sdxl_diagnostics(
            terminal.diagnostics,
            label="regional-no-lora",
            expected_spatial_modes=frozenset({"full"}),
        )
        image_bytes = client.download_image(terminal.image_reference)
        return self._save(
            name="regional-no-lora.png",
            label="SimpleSyrup regional — pink left / black right — no LoRA",
            prompt_id=prompt_id,
            image_bytes=image_bytes,
            metrics=metrics,
            wall_runtime_ms=wall_ms,
        )

    def _save(
        self,
        *,
        name: str,
        label: str,
        prompt_id: str,
        image_bytes: bytes,
        metrics: JsonObject,
        wall_runtime_ms: float,
    ) -> JsonObject:
        """Persist one exact image and its synchronized performance evidence."""

        dimensions, dynamic_range = sdxl_image_evidence(image_bytes)
        if dimensions != (1024, 1024) or dynamic_range == 0:
            raise ValueError("No-LoRA comparison image evidence is invalid.")
        image_path = self._artifacts.root / name
        image_path.write_bytes(image_bytes)
        return {
            "label": label,
            "prompt_id": prompt_id,
            "image_file": name,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "rgb_sha256": _rgb_sha256(image_bytes),
            "image_width": dimensions[0],
            "image_height": dimensions[1],
            "image_dynamic_range": dynamic_range,
            "model_runtime_ms": metrics["runtime_ms"],
            "wall_runtime_ms": wall_runtime_ms,
            "metrics": metrics,
        }


def _single_output(history: JsonObject, node_id: str, field: str) -> JsonObject:
    """Return one exact dictionary-valued UI output."""

    outputs = history.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get(node_id), dict):
        raise ValueError("No-LoRA comparison history is missing evidence.")
    values = outputs[node_id].get(field)
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise ValueError("No-LoRA comparison evidence is invalid.")
    return values[0]


def _rgb_sha256(image_bytes: bytes) -> str:
    """Hash decoded RGB pixels independently from PNG metadata."""

    with Image.open(BytesIO(image_bytes)) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
