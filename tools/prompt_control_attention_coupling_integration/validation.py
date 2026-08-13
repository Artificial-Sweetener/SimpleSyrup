# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate P9.1 conditioning, selection, LoRA, and performance evidence."""

from __future__ import annotations

import math
import uuid
from copy import deepcopy
from dataclasses import dataclass

from tools.anima_attention_coupling_workflow import (
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import JsonObject

from .baseline import PromptControlBaselineObservation
from .history import PromptControlAttentionOutputs
from .matrix import STEPS, PromptControlAttentionCase


@dataclass(frozen=True, slots=True)
class ValidatedPromptControlAttentionEvidence:
    """Retain normalized accepted values for durable result persistence."""

    sampling_sigmas: tuple[float, ...]
    conditioning_uuid_count: int
    diagnostic_record_count: int
    adapter_tokens: tuple[str, ...]


def validate_case_evidence(
    case: PromptControlAttentionCase,
    workflow: BuiltAnimaAttentionCouplingWorkflow,
    outputs: PromptControlAttentionOutputs,
    baseline: PromptControlBaselineObservation,
) -> ValidatedPromptControlAttentionEvidence:
    """Reject any divergence from pinned conditioning or regional execution."""

    if outputs.expansion != baseline.expansion:
        raise ValueError("P9.1 Prompt Control expansion diverged from P0.8.")
    _validate_snapshot(case, outputs.snapshot, baseline.snapshot)
    _validate_metrics(workflow, outputs.metrics)
    return _validate_diagnostics(case, workflow, outputs.diagnostics, baseline)


def _validate_snapshot(
    case: PromptControlAttentionCase,
    observed: JsonObject,
    expected: JsonObject,
) -> None:
    """Require exact P0.8 conditioning and supported regional hook metadata."""

    if observed.get("case_id") != case.case_id:
        raise ValueError("P9.1 conditioning snapshot identity changed.")
    if case.characterization.expect_static_model_lora:
        if _conditioning_without_hooks(observed) != _conditioning_without_hooks(
            expected
        ):
            raise ValueError("P9.1 static regional LoRA changed encoded conditioning.")
        hooks = _array(observed.get("hooks"), "snapshot hooks")
        _validate_static_hook(hooks)
        for side in ("positive", "negative"):
            for entry in _array(observed.get(side), f"snapshot {side}"):
                metadata = _object(entry, "conditioning entry").get("metadata")
                attached = _object(metadata, "conditioning metadata").get("hooks")
                attached_group = _object(attached, "conditioning hooks").get(
                    "hook_group"
                )
                _validate_static_hook(_array(attached_group, "hook group"))
        return
    if case.has_regional_hooks:
        _require_expected_subset(
            _without_hook_refs(observed),
            _without_hook_refs(expected),
            path="snapshot",
        )
        return
    if observed != expected:
        raise ValueError("P9.1 conditioning snapshot diverged from P0.8.")


def _without_hook_refs(value: object) -> object:
    """Remove only non-semantic native-versus-string hook references."""

    if isinstance(value, dict):
        return {
            key: _without_hook_refs(item)
            for key, item in value.items()
            if key != "hook_ref"
        }
    if isinstance(value, list):
        return [_without_hook_refs(item) for item in value]
    return value


def _require_expected_subset(observed: object, expected: object, *, path: str) -> None:
    """Require every baseline field while admitting additive metadata keys."""

    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise ValueError(f"P9.1 {path} changed type from the P0.8 baseline.")
        for key, expected_value in expected.items():
            if key not in observed:
                raise ValueError(f"P9.1 {path}.{key} is missing from the baseline.")
            _require_expected_subset(
                observed[key],
                expected_value,
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"P9.1 {path} list cardinality changed from P0.8.")
        for index, (observed_item, expected_item) in enumerate(
            zip(observed, expected, strict=True)
        ):
            _require_expected_subset(
                observed_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if observed != expected:
        raise ValueError(f"P9.1 {path} diverged from the P0.8 baseline.")


def _conditioning_without_hooks(snapshot: JsonObject) -> JsonObject:
    """Return static-case conditioning evidence without expected new model hooks."""

    result = deepcopy(snapshot)
    result["hooks"] = []
    for side in ("positive", "negative"):
        entries = _array(result.get(side), f"snapshot {side}")
        for entry in entries:
            metadata = _object(
                _object(entry, "conditioning entry").get("metadata"),
                "conditioning metadata",
            )
            metadata.pop("hooks", None)
    return result


def _validate_static_hook(hooks: list[object]) -> None:
    """Require one full-quality static ADAPTER_A WeightHook."""

    if len(hooks) != 1:
        raise ValueError("P9.1 static regional ADAPTER_A must expose one WeightHook.")
    hook = _object(hooks[0], "static hook")
    if (
        hook.get("identity") != "adapter_a-static-0.75"
        or hook.get("order") != 0
        or hook.get("hook_type") != "WeightHook"
        or hook.get("hook_scope") != "hooked_only"
        or hook.get("keyframes")
        != [{"start_percent": 0.0, "strength": 1.0, "guarantee_steps": 1}]
    ):
        raise ValueError("P9.1 static regional ADAPTER_A hook identity changed.")
    _same_number(hook.get("base_strength_model"), 0.75, "static model strength")
    _same_number(hook.get("base_strength_clip"), 0.25, "static CLIP strength")


def _validate_metrics(
    workflow: BuiltAnimaAttentionCouplingWorkflow,
    metrics: JsonObject,
) -> None:
    """Require one exact eight-call trajectory with measured runtime and VRAM."""

    if metrics.get("run_id") != workflow.metrics_run_id:
        raise ValueError("P9.1 metrics identity changed.")
    if metrics.get("model_call_count") != STEPS:
        raise ValueError("P9.1 must use one complete denoiser call per step.")
    if _number(metrics.get("runtime_ms"), "runtime_ms") <= 0.0:
        raise ValueError("P9.1 instrumented runtime must be positive.")
    peak = metrics.get("peak_vram_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError("P9.1 peak VRAM must be a non-negative integer.")


def _validate_diagnostics(
    case: PromptControlAttentionCase,
    workflow: BuiltAnimaAttentionCouplingWorkflow,
    diagnostics: JsonObject,
    baseline: PromptControlBaselineObservation,
) -> ValidatedPromptControlAttentionEvidence:
    """Match every P9.1 model-call snapshot to one P0.8 timestep."""

    if diagnostics.get("run_id") != workflow.diagnostics_run_id:
        raise ValueError("P9.1 diagnostic identity changed.")
    snapshots = _array(diagnostics.get("snapshots"), "diagnostic snapshots")
    if diagnostics.get("record_count") != len(snapshots) or len(snapshots) != STEPS:
        raise ValueError("P9.1 diagnostics must contain one record per denoiser call.")
    expected_calls = _baseline_calls(baseline.runtime)
    observed_by_sigma: dict[float, JsonObject] = {}
    all_uuids: set[str] = set()
    tokens: tuple[str, ...] | None = None
    for raw in snapshots:
        snapshot = _object(raw, "diagnostic snapshot")
        model_call = _object(snapshot.get("model_call"), "diagnostic model_call")
        sigma = _number(model_call.get("sampling_sigma"), "sampling sigma")
        matched_sigma = _matching_sigma(sigma, tuple(expected_calls))
        if matched_sigma in observed_by_sigma:
            raise ValueError("P9.1 emitted duplicate diagnostics for one timestep.")
        observed_by_sigma[matched_sigma] = snapshot
        uuids = _uuid_strings(model_call.get("conditioning_uuids"))
        chunks = _array(
            _object(snapshot.get("batch_layout"), "batch_layout").get("chunks"),
            "diagnostic chunks",
        )
        if len(uuids) != len(chunks):
            raise ValueError("P9.1 diagnostic UUID/chunk alignment changed.")
        all_uuids.update(uuids)
        _validate_regional_entries(case, snapshot, expected_calls[matched_sigma])
        observed_tokens = _validate_adapter_uses(
            case,
            snapshot,
            expected_calls[matched_sigma]["strengths"],
        )
        if tokens is None:
            tokens = observed_tokens
        elif tokens != observed_tokens:
            raise ValueError("P9.1 adapter tokens changed across timesteps.")
        work = _object(snapshot.get("estimated_work"), "estimated_work")
        _same_number(work.get("denoiser_call_multiplier"), 1.0, "denoiser work")
    if set(observed_by_sigma) != set(expected_calls):
        raise ValueError("P9.1 diagnostic timestep coverage changed.")
    return ValidatedPromptControlAttentionEvidence(
        tuple(observed_by_sigma),
        len(all_uuids),
        len(snapshots),
        () if tokens is None else tokens,
    )


def _baseline_calls(runtime: JsonObject) -> dict[float, JsonObject]:
    """Normalize P0.8 calls by timestep without interpreting schedules."""

    grouped: dict[float, JsonObject] = {}
    for raw in _array(runtime.get("calls"), "baseline runtime calls"):
        call = _object(raw, "baseline runtime call")
        timestep = _number(call.get("timestep"), "baseline timestep")
        selectors = _array(call.get("cond_or_uncond"), "baseline selectors")
        strengths = [
            _number(value, "baseline adapter strength")
            for value in _array(
                call.get("effective_strengths"),
                "baseline effective strengths",
            )
        ]
        current = grouped.setdefault(
            timestep,
            {"positive_count": 0, "negative_count": 0, "strengths": strengths},
        )
        if current["strengths"] != strengths:
            raise ValueError("P0.8 adapter strengths disagree within one timestep.")
        for selector in selectors:
            if selector == 0:
                current["positive_count"] = (
                    _integer(current["positive_count"], "positive_count") + 1
                )
            elif selector == 1:
                current["negative_count"] = (
                    _integer(current["negative_count"], "negative_count") + 1
                )
            else:
                raise ValueError("P0.8 baseline contains an invalid CFG selector.")
    if len(grouped) != STEPS:
        raise ValueError("P0.8 baseline must expose eight unique timesteps.")
    return grouped


def _validate_regional_entries(
    case: PromptControlAttentionCase,
    snapshot: JsonObject,
    expected_call: JsonObject,
) -> None:
    """Match selected region-zero entry support to P0.8 UUID cardinality."""

    entries = tuple(
        _object(raw, "regional entry")
        for raw in _array(
            snapshot.get("regional_conditioning_entries"),
            "regional_conditioning_entries",
        )
        if _object(raw, "regional entry").get("region_index") == 0
    )
    chunks = tuple(
        _object(raw, "diagnostic chunk")
        for raw in _array(
            _object(snapshot.get("batch_layout"), "batch_layout").get("chunks"),
            "diagnostic chunks",
        )
    )
    allowed_positive = {
        1.0 if expected.strength is None else expected.strength
        for expected in case.characterization.expected_positive
    }
    for branch, expected_field in (
        ("positive", "positive_count"),
        ("negative", "negative_count"),
    ):
        branch_slices = tuple(
            (
                _integer(chunk.get("batch_start"), "chunk batch_start"),
                _integer(chunk.get("batch_stop"), "chunk batch_stop"),
            )
            for chunk in chunks
            if chunk.get("branch") == branch
        )
        active_count = 0
        for entry in entries:
            strengths = tuple(
                _number(value, "regional entry strength")
                for value in _array(entry.get("strengths"), "entry strengths")
            )
            active_values = tuple(
                value
                for start, stop in branch_slices
                for value in strengths[start:stop]
                if value != 0.0
            )
            if active_values:
                active_count += 1
                allowed = allowed_positive if branch == "positive" else {1.0}
                if any(
                    not any(math.isclose(value, item, abs_tol=1e-9) for item in allowed)
                    for value in active_values
                ):
                    raise ValueError("P9.1 selected conditioning strength changed.")
        if active_count != _integer(expected_call[expected_field], expected_field):
            raise ValueError(
                "P9.1 active regional text-entry count diverged from P0.8."
            )


def _validate_adapter_uses(
    case: PromptControlAttentionCase,
    snapshot: JsonObject,
    baseline_strengths: object,
) -> tuple[str, ...]:
    """Require exact full-target regional adapter strengths on both CFG branches."""

    uses = tuple(
        _object(raw, "adapter use")
        for raw in _array(snapshot.get("adapter_uses", []), "adapter_uses")
    )
    if not case.has_regional_hooks:
        if uses:
            raise ValueError("P9.1 hook-free case unexpectedly executed adapters.")
        return ()
    expected = (
        (0.75,)
        if case.characterization.expect_static_model_lora
        else tuple(
            _number(value, "baseline strength")
            for value in _array(baseline_strengths, "baseline strengths")
        )
    )
    if len(uses) != 2 * len(expected):
        raise ValueError("P9.1 adapter-use cardinality changed.")
    if tuple(use.get("composition_index") for use in uses) != tuple(range(len(uses))):
        raise ValueError("P9.1 adapter composition order changed.")
    tokens: list[str] = []
    for branch in ("positive", "negative"):
        branch_uses = tuple(use for use in uses if use.get("branch") == branch)
        if len(branch_uses) != len(expected):
            raise ValueError("P9.1 adapter CFG ownership changed.")
        for index, (use, strength) in enumerate(
            zip(branch_uses, expected, strict=True)
        ):
            if use.get("region_index") != 0 or use.get("target_count") != 448:
                raise ValueError("P9.1 regional adapter target surface changed.")
            _same_number(
                use.get("effective_strength"), strength, "adapter effective strength"
            )
            if use.get("active") is not (strength != 0.0):
                raise ValueError("P9.1 adapter active state changed.")
            token = use.get("adapter_token")
            if not isinstance(token, str) or len(token) != 16:
                raise ValueError("P9.1 adapter token is invalid.")
            if branch == "positive":
                tokens.append(token)
            elif token != tokens[index]:
                raise ValueError("P9.1 adapter identity differs across CFG branches.")
    return tuple(tokens)


def _matching_sigma(value: float, candidates: tuple[float, ...]) -> float:
    """Return the sole numerically equal pinned timestep."""

    matches = tuple(
        candidate
        for candidate in candidates
        if math.isclose(value, candidate, abs_tol=1e-6)
    )
    if len(matches) != 1:
        raise ValueError(f"P9.1 sigma does not match one P0.8 timestep: {value}.")
    return matches[0]


def _uuid_strings(value: object) -> tuple[str, ...]:
    """Validate diagnostic UUIDv4 strings in exact order."""

    values = _array(value, "conditioning UUIDs")
    result: list[str] = []
    for item in values:
        if not isinstance(item, str) or uuid.UUID(item).version != 4:
            raise ValueError("P9.1 conditioning identities must be UUIDv4 strings.")
        result.append(item)
    if len(set(result)) != len(result):
        raise ValueError("P9.1 conditioning identities must be call-unique.")
    return tuple(result)


def _same_number(value: object, expected: float, field: str) -> None:
    """Compare one finite number without accepting booleans."""

    if not math.isclose(_number(value, field), expected, abs_tol=1e-6):
        raise ValueError(f"P9.1 {field} changed.")


def _number(value: object, field: str) -> float:
    """Narrow one finite JSON number."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"P9.1 {field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"P9.1 {field} must be finite.")
    return result


def _integer(value: object, field: str) -> int:
    """Narrow one JSON integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"P9.1 {field} must be an integer.")
    return value


def _array(value: object, field: str) -> list[object]:
    """Narrow one JSON array."""

    if not isinstance(value, list):
        raise TypeError(f"P9.1 {field} must be an array.")
    return value


def _object(value: object, field: str) -> JsonObject:
    """Narrow one JSON object with string keys."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"P9.1 {field} must be an object.")
    return value
