# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Enforce complete ADAPTER_A target, order, schedule, and denoiser evidence."""

from __future__ import annotations

import math

from tools.comfy_api import JsonObject

from .artifact_inventory import AdapterInventory
from .history_outputs import LoraCompletedOutputs
from .json_contract import (
    array_value,
    boolean_value,
    integer_value,
    number_value,
    object_value,
    string_list,
    text_value,
)
from .matrix import LoraRun


def validate_run_evidence(
    run: LoraRun,
    outputs: LoraCompletedOutputs,
    inventory: AdapterInventory,
) -> None:
    """Require complete targets, order, schedules, calls, and digest coverage."""

    metrics = outputs.metrics
    if text_value(metrics.get("run_id"), "run_id") != run.artifact_id:
        raise ValueError("LoRA probe run ID does not match the scheduled run.")
    if (
        boolean_value(metrics.get("capture_outputs"), "capture_outputs")
        != run.capture_outputs
    ):
        raise ValueError("LoRA probe capture mode does not match the scheduled run.")
    if integer_value(metrics.get("model_call_count"), "model_call_count") != 30:
        raise ValueError("Pinned global LoRA run must execute exactly 30 model calls.")
    expected_targets = [f"{target}.weight" for target in inventory.target_keys]
    static = object_value(metrics.get("static_patches"), "static_patches")
    hooks = [
        object_value(item, "hook patch")
        for item in array_value(metrics.get("hook_patches"), "hook_patches")
    ]
    schedules = [
        object_value(item, "schedule observation")
        for item in array_value(
            metrics.get("schedule_observations"), "schedule_observations"
        )
    ]
    denoiser = [
        object_value(item, "denoiser output")
        for item in array_value(metrics.get("denoiser_outputs"), "denoiser_outputs")
    ]
    if run.profile.mode == "static":
        _validate_static(run, static, hooks, schedules, expected_targets)
    else:
        _validate_scheduled(run, static, hooks, schedules, expected_targets)
    _validate_denoiser_outputs(run, denoiser)


def _validate_static(
    run: LoraRun,
    static: JsonObject,
    hooks: list[JsonObject],
    schedules: list[JsonObject],
    expected: list[str],
) -> None:
    """Require ordinary static loader targets and ordered patch strengths."""

    count = len(run.profile.adapters)
    expected_count = len(expected) if count else 0
    if (
        integer_value(static.get("target_count"), "static target count")
        != expected_count
    ):
        raise ValueError("Static LoRA target count is incomplete.")
    expected_loaded = expected if count else []
    if string_list(static.get("target_keys"), "static target keys") != expected_loaded:
        raise ValueError("Static LoRA loaded targets do not match the pinned surface.")
    if string_list(static.get("inconsistent_order_targets"), "inconsistent targets"):
        raise ValueError("Static LoRA patch order differs across targets.")
    order = [
        object_value(item, "static patch order")
        for item in array_value(
            static.get("canonical_patch_order"), "canonical patch order"
        )
    ]
    _validate_identity_strength_order(run, order, "strength_patch")
    if hooks or schedules:
        raise ValueError("Static LoRA run must not report scheduled hooks.")


def _validate_scheduled(
    run: LoraRun,
    static: JsonObject,
    hooks: list[JsonObject],
    schedules: list[JsonObject],
    expected: list[str],
) -> None:
    """Require complete independently ordered hooks and schedule intervals."""

    if integer_value(static.get("target_count"), "static target count") != 0:
        raise ValueError("Scheduled LoRA run must not contain static patches.")
    _validate_identity_strength_order(run, hooks, "base_strength_model")
    for hook in hooks:
        if integer_value(hook.get("target_count"), "hook target count") != len(
            expected
        ):
            raise ValueError("Scheduled LoRA hook target count is incomplete.")
        if string_list(hook.get("target_keys"), "hook target keys") != expected:
            raise ValueError("Scheduled hook targets do not match the pinned surface.")
    expected_strengths = [
        [adapter.strength * multiplier for adapter in run.profile.adapters]
        for multiplier in (0.0, 1.0, 0.5, 0.0)
    ]
    actual_strengths = [
        [
            number_value(value, "effective strength")
            for value in array_value(
                item.get("effective_strengths"), "effective strengths"
            )
        ]
        for item in schedules
    ]
    if len(actual_strengths) != len(expected_strengths) or any(
        not all(
            math.isclose(actual, wanted) for actual, wanted in zip(a, b, strict=True)
        )
        for a, b in zip(actual_strengths, expected_strengths, strict=True)
    ):
        raise ValueError("Scheduled LoRA transitions do not match fixed boundaries.")


def _validate_identity_strength_order(
    run: LoraRun, values: list[JsonObject], strength_field: str
) -> None:
    """Match exact declared identity, order, and base strength."""

    if len(values) != len(run.profile.adapters):
        raise ValueError("Observed adapter count does not match the profile.")
    for index, (value, adapter) in enumerate(
        zip(values, run.profile.adapters, strict=True)
    ):
        identity_matches = (
            text_value(value.get("identity"), "adapter identity") == adapter.identity
        )
        order_matches = integer_value(value.get("order"), "adapter order") == index
        strength_matches = math.isclose(
            number_value(value.get(strength_field), strength_field), adapter.strength
        )
        if not identity_matches or not order_matches or not strength_matches:
            raise ValueError(
                "Observed adapter identity, order, or strength differs "
                "from the profile."
            )


def _validate_denoiser_outputs(run: LoraRun, denoiser: list[JsonObject]) -> None:
    """Require exact call coverage only on isolated digest-capture runs."""

    expected_count = 30 if run.capture_outputs else 0
    if len(denoiser) != expected_count:
        raise ValueError("Denoiser digest count does not match the capture profile.")
    for index, digest in enumerate(denoiser, start=1):
        if integer_value(digest.get("call_index"), "denoiser call index") != index:
            raise ValueError("Denoiser output call indices must be contiguous.")
        value = text_value(digest.get("float32_sha256"), "denoiser SHA-256")
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("Denoiser SHA-256 must contain 64 lowercase hex digits.")
