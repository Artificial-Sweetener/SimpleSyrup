"""Validate exact P9.7 modifier, execution, and rejection evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tools.comfy_api import JsonObject

from .history import (
    RegionalPatchInteropError,
    RegionalPatchInteropHistory,
    RegionalPatchInteropSuccess,
)
from .matrix import (
    STEPS,
    PatchInteropModelFamily,
    PatchInteropModifier,
    PatchInteropSpatialMode,
    RegionalPatchInteropCase,
)
from .workflow import BuiltRegionalPatchInteropWorkflow


@dataclass(frozen=True, slots=True)
class ValidatedRegionalPatchInterop:
    """Retain normalized evidence for durable result publication."""

    status: str
    model_call_count: int
    diagnostic_record_count: int
    spatial_modes: tuple[str, ...]
    exception_type: str | None = None
    exception_message: str | None = None


def validate_case(
    case: RegionalPatchInteropCase,
    workflow: BuiltRegionalPatchInteropWorkflow,
    observed: RegionalPatchInteropHistory,
) -> ValidatedRegionalPatchInterop:
    """Require the case's exact modifier state and terminal outcome."""

    _validate_modifier_snapshot(case, workflow, observed.modifier_snapshot)
    if case.expect_success:
        if not isinstance(observed, RegionalPatchInteropSuccess):
            raise ValueError("P9.7 accepted case did not complete sampling.")
        return _validate_success(case, workflow, observed)
    if not isinstance(observed, RegionalPatchInteropError):
        raise ValueError("P9.7 rejected case unexpectedly completed sampling.")
    return _validate_rejection(case, workflow, observed)


def _validate_modifier_snapshot(
    case: RegionalPatchInteropCase,
    workflow: BuiltRegionalPatchInteropWorkflow,
    snapshot: JsonObject,
) -> None:
    """Require exact upstream public modifier state before sampler derivation."""

    expected_run_id = f"{workflow.metrics_run_id}:modifier"
    if snapshot.get("run_id") != expected_run_id:
        raise ValueError("P9.7 MODEL modifier snapshot identity changed.")
    marker = snapshot.get("ppm_negpip")
    if marker is not (case.modifier is PatchInteropModifier.NEGPIP):
        raise ValueError("P9.7 NegPiP marker state changed.")
    cache_type = snapshot.get("cache_holder_type")
    expected_cache = {
        PatchInteropModifier.EASYCACHE: "EasyCacheHolder",
        PatchInteropModifier.LAZYCACHE: "LazyCacheHolder",
    }.get(case.modifier)
    if cache_type != expected_cache:
        raise ValueError("P9.7 cache holder identity changed.")
    optimized = snapshot.get("optimized_attention_override")
    if (optimized is not None) is not (
        case.modifier is PatchInteropModifier.OPTIMIZED_ATTENTION
    ):
        raise ValueError("P9.7 optimized-attention override state changed.")
    wrappers = _array(snapshot.get("wrappers"), "modifier wrappers")
    wrapper_keys = {
        (
            _string(_object(item, "modifier wrapper").get("wrapper_type"), "type"),
            _string(_object(item, "modifier wrapper").get("key"), "key"),
        )
        for item in wrappers
    }
    expected_wrapper_keys = {
        PatchInteropModifier.NONE: set(),
        PatchInteropModifier.EASYCACHE: {
            ("outer_sample", "easycache"),
            ("calc_cond_batch", "easycache"),
            ("diffusion_model", "easycache"),
        },
        PatchInteropModifier.LAZYCACHE: {
            ("outer_sample", "lazycache"),
            ("predict_noise", "lazycache"),
        },
        PatchInteropModifier.OPTIMIZED_ATTENTION: set(),
        PatchInteropModifier.NEGPIP: (
            {("diffusion_model", "ppm_negpip_anima")}
            if case.model_family is PatchInteropModelFamily.ANIMA
            else set()
        ),
    }[case.modifier]
    if wrapper_keys != expected_wrapper_keys:
        raise ValueError("P9.7 upstream wrapper keys changed.")
    patch_counts = _object(
        snapshot.get("transformer_patch_counts"),
        "transformer patch counts",
    )
    expected_patches = (
        {"attn2_patch": 1} if case.modifier is PatchInteropModifier.NEGPIP else {}
    )
    if patch_counts != expected_patches:
        raise ValueError("P9.7 upstream transformer patch state changed.")
    object_patches = _array(snapshot.get("object_patch_keys"), "object patch keys")
    expected_objects = (
        ["extra_conds"]
        if case.modifier is PatchInteropModifier.NEGPIP
        and case.model_family is PatchInteropModelFamily.ANIMA
        else []
    )
    if object_patches != expected_objects:
        raise ValueError("P9.7 upstream object patch state changed.")


def _validate_success(
    case: RegionalPatchInteropCase,
    workflow: BuiltRegionalPatchInteropWorkflow,
    observed: RegionalPatchInteropSuccess,
) -> ValidatedRegionalPatchInterop:
    """Require one exact single-trajectory regional ADAPTER_A execution."""

    metrics = observed.metrics
    if metrics.get("run_id") != workflow.metrics_run_id:
        raise ValueError("P9.7 metrics identity changed.")
    model_calls = _integer(metrics.get("model_call_count"), "model call count")
    _validate_model_call_count(case, model_calls)
    if _number(metrics.get("runtime_ms"), "runtime_ms") <= 0.0:
        raise ValueError("P9.7 measured runtime must be positive.")
    peak = _integer(metrics.get("peak_vram_bytes"), "peak VRAM")
    if peak < 0:
        raise ValueError("P9.7 peak VRAM must not be negative.")
    diagnostics = observed.diagnostics
    if diagnostics.get("run_id") != workflow.diagnostics_run_id:
        raise ValueError("P9.7 diagnostics identity changed.")
    snapshots = _array(diagnostics.get("snapshots"), "diagnostic snapshots")
    record_count = _integer(diagnostics.get("record_count"), "record count")
    if record_count != model_calls or len(snapshots) != record_count:
        raise ValueError("P9.7 requires one diagnostic record per model call.")
    if record_count < 1:
        raise ValueError("P9.7 diagnostics must contain aligned model-call records.")
    spatial_modes: set[str] = set()
    adapter_tokens: set[str] = set()
    for item in snapshots:
        snapshot = _object(item, "diagnostic snapshot")
        if (
            snapshot.get("strategy") != "attention_coupling"
            or snapshot.get("backend") != "comfy.ldm.anima.model.Anima"
        ):
            raise ValueError("P9.7 diagnostic strategy or backend changed.")
        spatial_modes.add(_string(snapshot.get("spatial_mode"), "spatial mode"))
        uses = _array(snapshot.get("adapter_uses"), "adapter uses")
        if len(uses) != 2:
            raise ValueError(
                "P9.7 accepted cases require paired positive/negative ADAPTER_A uses."
            )
        normalized_uses = tuple(_object(use, "adapter use") for use in uses)
        if {
            (_string(use.get("branch"), "adapter branch"), use.get("composition_index"))
            for use in normalized_uses
        } != {("positive", 0), ("negative", 1)}:
            raise ValueError("P9.7 paired regional ADAPTER_A branch ownership changed.")
        for use in normalized_uses:
            if (
                use.get("active") is not True
                or use.get("region_index") != 0
                or use.get("target_count") != 448
            ):
                raise ValueError("P9.7 exact regional ADAPTER_A execution changed.")
            if not math.isclose(
                _number(use.get("effective_strength"), "effective strength"),
                0.75,
                abs_tol=1e-8,
            ):
                raise ValueError("P9.7 regional ADAPTER_A strength changed.")
            adapter_tokens.add(_string(use.get("adapter_token"), "adapter token"))
        work = _object(snapshot.get("estimated_work"), "estimated work")
        if (
            work.get("active_adapter_uses") != 2
            or work.get("active_target_count") != 448
            or work.get("target_use_count") != 896
        ):
            raise ValueError("P9.7 paired LoRA target-use accounting changed.")
        if not math.isclose(
            _number(work.get("denoiser_call_multiplier"), "denoiser multiplier"),
            1.0,
            abs_tol=1e-8,
        ):
            raise ValueError("P9.7 denoiser trajectory multiplier changed.")
    expected_modes = {
        PatchInteropSpatialMode.FULL: {"full"},
        PatchInteropSpatialMode.TILED: {"tile"},
        PatchInteropSpatialMode.CONTEXTUAL: {"tile", "contextual_global"},
    }[case.spatial_mode]
    if not expected_modes <= spatial_modes:
        raise ValueError("P9.7 spatial diagnostics are incomplete.")
    if len(adapter_tokens) != 1:
        raise ValueError("P9.7 regional ADAPTER_A identity changed during sampling.")
    return ValidatedRegionalPatchInterop(
        "accepted",
        model_calls,
        record_count,
        tuple(sorted(spatial_modes)),
    )


def _validate_model_call_count(
    case: RegionalPatchInteropCase,
    model_calls: int,
) -> None:
    """Validate actual evaluations without counting cache-reused denoising steps."""

    if case.modifier is PatchInteropModifier.EASYCACHE:
        if not 1 <= model_calls <= STEPS:
            raise ValueError(
                "P9.7 cache model calls must remain within denoising steps."
            )
        return
    if case.spatial_mode is PatchInteropSpatialMode.FULL:
        if model_calls != STEPS:
            raise ValueError(
                "P9.7 full-context model calls must equal denoising steps."
            )
        return
    if model_calls < STEPS:
        raise ValueError("P9.7 spatial model calls must cover every denoising step.")


def _validate_rejection(
    case: RegionalPatchInteropCase,
    workflow: BuiltRegionalPatchInteropWorkflow,
    observed: RegionalPatchInteropError,
) -> ValidatedRegionalPatchInterop:
    """Require one named public-sampler conflict before execution."""

    if observed.node_id != workflow.sampler_node_id:
        raise ValueError("P9.7 rejection did not originate at the public sampler.")
    expected_node_type = str(workflow.prompt[workflow.sampler_node_id]["class_type"])
    if observed.node_type != expected_node_type:
        raise ValueError("P9.7 rejection node type changed.")
    if not observed.exception_type.endswith("ValueError"):
        raise ValueError("P9.7 rejection exception classification changed.")
    if any(
        fragment not in observed.exception_message
        for fragment in case.expected_error_fragments
    ):
        raise ValueError("P9.7 rejection message lost its named conflict.")
    if workflow.sampler_node_id in observed.executed_node_ids:
        raise ValueError("P9.7 rejected sampler was reported as executed.")
    return ValidatedRegionalPatchInterop(
        "rejected",
        0,
        0,
        (),
        observed.exception_type,
        observed.exception_message,
    )


def _integer(value: object, field: str) -> int:
    """Return one exact non-boolean integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"P9.7 {field} must be an integer.")
    return value


def _number(value: object, field: str) -> float:
    """Return one finite JSON number."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"P9.7 {field} must be finite numeric data.")
    return float(value)


def _array(value: object, field: str) -> list[object]:
    """Return one JSON array."""

    if not isinstance(value, list):
        raise TypeError(f"P9.7 {field} must be a list.")
    return value


def _string(value: object, field: str) -> str:
    """Return one nonempty JSON string."""

    if not isinstance(value, str) or not value:
        raise TypeError(f"P9.7 {field} must be a nonempty string.")
    return value


def _object(value: object, field: str) -> JsonObject:
    """Return one JSON object with string keys."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"P9.7 {field} must be an object.")
    return value
