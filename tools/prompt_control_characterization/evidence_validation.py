# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Enforce exact Prompt Control conditioning, hook, UUID, and runtime evidence."""

from __future__ import annotations

import math
import uuid

from tools.comfy_api import JsonObject

from .cases import ExpectedAdapter, PromptControlCase
from .history_outputs import PromptControlOutputs


def validate_evidence(case: PromptControlCase, outputs: PromptControlOutputs) -> None:
    """Reject incomplete or semantically changed P0.8 observations."""

    _validate_expansion(case, outputs.expansion)
    snapshot = outputs.snapshot
    runtime = outputs.runtime
    if snapshot.get("case_id") != case.case_id or runtime.get("run_id") != case.case_id:
        raise ValueError(
            "Prompt Control evidence run identity does not match its case."
        )
    positive = _array(snapshot.get("positive"), "snapshot.positive")
    negative = _array(snapshot.get("negative"), "snapshot.negative")
    if len(positive) != len(case.expected_positive):
        raise ValueError("Prompt Control positive conditioning entry count changed.")
    for index, (entry, expected) in enumerate(
        zip(positive, case.expected_positive, strict=True)
    ):
        metadata = _entry_metadata(entry, f"positive[{index}]")
        _same_number(metadata.get("start_percent"), expected.start, "start_percent")
        _same_number(metadata.get("end_percent"), expected.end, "end_percent")
        if expected.strength is None:
            if "strength" in metadata:
                raise ValueError("Unexpected Prompt Control conditioning strength.")
        else:
            _same_number(metadata.get("strength"), expected.strength, "strength")
    if len(negative) != len(case.expected_negative):
        raise ValueError("Prompt Control negative conditioning entry count changed.")
    for index, (entry, expected) in enumerate(
        zip(negative, case.expected_negative, strict=True)
    ):
        negative_metadata = _entry_metadata(entry, f"negative[{index}]")
        _same_number(
            negative_metadata.get("start_percent"), expected.start, "negative start"
        )
        _same_number(negative_metadata.get("end_percent"), expected.end, "negative end")

    hooks = _array(snapshot.get("hooks"), "snapshot.hooks")
    _validate_hooks(hooks, case.expected_adapters)
    for entry in (*positive, *negative):
        metadata = _entry_metadata(entry, "conditioning entry")
        attached = metadata.get("hooks")
        if case.expected_adapters:
            attached_object = _object(attached, "conditioning hooks")
            attached_hooks = _array(
                attached_object.get("hook_group"), "conditioning hook_group"
            )
            _validate_hooks(attached_hooks, case.expected_adapters)
        elif attached is not None:
            raise ValueError("Conditioning unexpectedly contains Prompt Control hooks.")

    registered = _array(runtime.get("registered_hooks"), "registered_hooks")
    _validate_hooks(registered, case.expected_adapters)
    static_target_count = _integer(
        runtime.get("static_patch_target_count"), "static_patch_target_count"
    )
    histogram = _object(
        runtime.get("static_patch_entry_count_histogram"),
        "static_patch_entry_count_histogram",
    )
    if case.expect_static_model_lora:
        if static_target_count <= 0 or histogram != {"1": static_target_count}:
            raise ValueError(
                "Static Prompt Control LoRA expansion must apply every loaded target."
            )
        if (
            case.expected_static_target_count is not None
            and static_target_count != case.expected_static_target_count
        ):
            raise ValueError("Static Prompt Control target count changed.")
    elif static_target_count != 0 or histogram:
        raise ValueError("Scheduled or text-only case unexpectedly has static patches.")

    calls = _array(runtime.get("calls"), "runtime.calls")
    call_count = _integer(runtime.get("model_call_count"), "model_call_count")
    if call_count != len(calls) or call_count != case.expected_model_call_count:
        raise ValueError("Prompt Control runtime call count changed.")
    observed_uuids: set[str] = set()
    for call in calls:
        call_object = _object(call, "runtime call")
        uuids = _array(call_object.get("uuids"), "runtime call UUIDs")
        layout = _array(
            call_object.get("cond_or_uncond"), "runtime call cond_or_uncond"
        )
        if len(uuids) != len(layout) or len(set(uuids)) != len(uuids):
            raise ValueError("Prompt Control call UUID/layout alignment changed.")
        for value in uuids:
            if not isinstance(value, str) or uuid.UUID(value).version != 4:
                raise ValueError("Comfy conditioning UUID must be a UUIDv4 string.")
            observed_uuids.add(value)
    expected_uuid_count = len(case.expected_positive) + len(case.expected_negative)
    if len(observed_uuids) != expected_uuid_count:
        raise ValueError("Sampler-assigned conditioning UUID cardinality changed.")
    _validate_transition_vectors(runtime, case)


def _validate_expansion(case: PromptControlCase, expansion: JsonObject) -> None:
    """Verify canonical lazy graph shape while preserving every exact input."""

    if expansion.get("case_id") != case.case_id:
        raise ValueError("Prompt Control expansion identity changed.")
    text_graph = _object(expansion.get("text_expansion"), "text_expansion")
    lora_graph = _object(expansion.get("lora_expansion"), "lora_expansion")
    text_counts = _class_counts(text_graph)
    if case.text_construction == "adjacent":
        expected_text = {
            "PCTextEncode": 2,
            "ConditioningSetTimestepRange": 2,
            "ConditioningCombine": 1,
        }
    elif case.text_construction == "lazy":
        expected_text = {"PCTextEncode": 1, "ConditioningSetTimestepRange": 1}
    else:
        expected_text = {}
    if text_counts != expected_text:
        raise ValueError("Prompt Control text lazy expansion graph changed.")
    expected_lora: dict[str, int]
    if case.case_id == "lora-single-static":
        expected_lora = {"LoraLoader": 1}
    elif case.case_id == "lora-single-scheduled":
        expected_lora = {
            "CreateHookLora": 1,
            "CreateHookKeyframe": 3,
            "SetHookKeyframes": 1,
            "SetClipHooks": 1,
        }
    elif case.case_id == "lora-adjacent":
        expected_lora = {
            "CreateHookLora": 2,
            "CreateHookKeyframe": 4,
            "SetHookKeyframes": 2,
            "CombineHooks2": 1,
            "SetClipHooks": 1,
        }
    elif case.case_id == "lora-stacked-overlap":
        expected_lora = {
            "CreateHookLora": 2,
            "CreateHookKeyframe": 8,
            "SetHookKeyframes": 2,
            "CombineHooks2": 1,
            "SetClipHooks": 1,
        }
    else:
        expected_lora = {}
    if _class_counts(lora_graph) != expected_lora:
        raise ValueError("Prompt Control LoRA lazy expansion graph changed.")


def _class_counts(graph: JsonObject) -> dict[str, int]:
    """Count exact expanded class types without interpreting schedule inputs."""

    counts: dict[str, int] = {}
    for raw_node in graph.values():
        node = _object(raw_node, "expansion node")
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or not class_type:
            raise TypeError("Prompt Control expansion class_type must be text.")
        counts[class_type] = counts.get(class_type, 0) + 1
    return counts


def _validate_hooks(hooks: list[object], expected: tuple[ExpectedAdapter, ...]) -> None:
    """Match ordered native WeightHooks and every preserved keyframe."""

    if len(hooks) != len(expected):
        raise ValueError("Prompt Control HookGroup size changed.")
    for index, (raw, adapter) in enumerate(zip(hooks, expected, strict=True)):
        hook = _object(raw, f"hook[{index}]")
        identity = hook.get("identity")
        if not isinstance(identity, str) or not identity or hook.get("order") != index:
            raise ValueError("Prompt Control adapter identity or order is invalid.")
        if (
            hook.get("hook_type") != "WeightHook"
            or hook.get("hook_scope") != "hooked_only"
        ):
            raise ValueError("Prompt Control hook class or scope changed.")
        _same_number(
            hook.get("base_strength_model"), adapter.strength_model, "model strength"
        )
        _same_number(
            hook.get("base_strength_clip"), adapter.strength_clip, "CLIP strength"
        )
        keyframes = _array(hook.get("keyframes"), "hook keyframes")
        actual = []
        for raw_keyframe in keyframes:
            keyframe = _object(raw_keyframe, "hook keyframe")
            actual.append(
                (
                    _number(keyframe.get("start_percent"), "keyframe start"),
                    _number(keyframe.get("strength"), "keyframe strength"),
                )
            )
            if keyframe.get("guarantee_steps") != 1:
                raise ValueError("Prompt Control keyframe guarantee_steps changed.")
        if tuple(actual) != adapter.keyframes:
            raise ValueError("Prompt Control native keyframe chain changed.")


def _validate_transition_vectors(runtime: JsonObject, case: PromptControlCase) -> None:
    """Match the exact pinned call-index and adapter-strength transition sequence."""

    transitions = _array(runtime.get("schedule_transitions"), "schedule_transitions")
    transitions_actual: list[tuple[int, tuple[float, ...]]] = []
    for raw in transitions:
        transition = _object(raw, "schedule transition")
        values = _array(transition.get("effective_strengths"), "effective strengths")
        transitions_actual.append(
            (
                _integer(transition.get("call_index"), "transition call_index"),
                tuple(_number(value, "effective strength") for value in values),
            )
        )
    if tuple(transitions_actual) != case.expected_runtime_transitions:
        raise ValueError("Prompt Control runtime schedule transitions changed.")


def _entry_metadata(value: object, field: str) -> JsonObject:
    """Read one normalized conditioning entry's metadata."""

    entry = _object(value, field)
    return _object(entry.get("metadata"), f"{field}.metadata")


def _same_number(value: object, expected: float, field: str) -> None:
    """Compare one JSON number without accepting booleans."""

    if not math.isclose(_number(value, field), expected, abs_tol=1e-9):
        raise ValueError(f"Prompt Control {field} changed.")


def _number(value: object, field: str) -> float:
    """Narrow one JSON number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric.")
    return float(value)


def _integer(value: object, field: str) -> int:
    """Narrow one JSON integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer.")
    return value


def _array(value: object, field: str) -> list[object]:
    """Narrow one JSON array."""

    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array.")
    return value


def _object(value: object, field: str) -> JsonObject:
    """Narrow one JSON object with string keys."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be an object.")
    return value
