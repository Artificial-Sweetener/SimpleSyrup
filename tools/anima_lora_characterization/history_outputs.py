"""Decode P0.7 LoRA probe and image outputs from Comfy history."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import ImageReference, JsonObject

from .json_contract import (
    array_value,
    boolean_value,
    integer_value,
    number_value,
    object_value,
    text_value,
)


@dataclass(frozen=True)
class LoraCompletedOutputs:
    """Hold one validated probe record and saved Comfy image reference."""

    metrics: JsonObject
    image: ImageReference


def parse_completed_outputs(
    history: JsonObject,
    *,
    metrics_node_id: str,
    save_node_id: str,
) -> LoraCompletedOutputs:
    """Narrow one successful Comfy history record into P0.7 evidence."""

    status = object_value(history.get("status"), "history.status")
    if status.get("status_str") != "success":
        raise RuntimeError(f"ComfyUI execution failed: {status.get('messages')!r}.")
    outputs = object_value(history.get("outputs"), "history.outputs")
    metric_output = object_value(outputs.get(metrics_node_id), "LoRA probe output")
    entries = array_value(
        metric_output.get("lora_benchmark_metrics"), "lora_benchmark_metrics"
    )
    if len(entries) != 1:
        raise ValueError("LoRA probe must return exactly one metric record.")
    metrics = object_value(entries[0], "LoRA metric record")
    _validate_metric_shape(metrics)
    save_output = object_value(outputs.get(save_node_id), "save output")
    images = array_value(save_output.get("images"), "saved images")
    if len(images) != 1:
        raise ValueError("LoRA workflow must save exactly one image.")
    image = object_value(images[0], "saved image")
    return LoraCompletedOutputs(
        metrics=metrics,
        image=ImageReference(
            text_value(image.get("filename"), "saved image filename"),
            text_value(image.get("subfolder"), "saved image subfolder", empty=True),
            text_value(image.get("type"), "saved image type"),
        ),
    )


def _validate_metric_shape(metrics: JsonObject) -> None:
    """Validate top-level probe values before preserving dynamic evidence."""

    text_value(metrics.get("run_id"), "run_id")
    boolean_value(metrics.get("capture_outputs"), "capture_outputs")
    number_value(metrics.get("runtime_ms"), "runtime_ms")
    integer_value(metrics.get("peak_vram_bytes"), "peak_vram_bytes")
    integer_value(metrics.get("model_call_count"), "model_call_count")
    object_value(metrics.get("static_patches"), "static_patches")
    array_value(metrics.get("hook_patches"), "hook_patches")
    array_value(metrics.get("schedule_observations"), "schedule_observations")
    array_value(metrics.get("denoiser_outputs"), "denoiser_outputs")
