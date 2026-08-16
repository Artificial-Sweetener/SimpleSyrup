# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate collaborator-owned MODEL modifier state before regional derivation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP

from ..domain.processed_regional_attention import ProcessedRegionalAttentionPlan
from ..domain.regional_attention_execution import RegionalAttentionExecutionMode
from ..domain.regional_model_capabilities import (
    RegionalModelCapabilities,
    RegionalModelFamily,
    RegionalPatchConflict,
)

LOGGER = logging.getLogger(__name__)

_NEGPIP_MODEL_OPTION = "ppm_negpip"
_NEGPIP_ANIMA_WRAPPER_KEY = "ppm_negpip_anima"
_EASYCACHE_OPTION = "easycache"
_ATTN2_PATCH_CONFLICTS = {
    RegionalPatchConflict.ATTN2_INPUT_PATCH: "attn2_patch",
    RegionalPatchConflict.ATTN2_OUTPUT_PATCH: "attn2_output_patch",
}


class RegionalPreservedModelModifier(StrEnum):
    """Identify one admitted upstream modifier category without local paths."""

    MODEL_FUNCTION_WRAPPER = "model_function_wrapper"
    DIFFUSION_MODEL_WRAPPER = "diffusion_model_wrapper"
    OPTIMIZED_ATTENTION_OVERRIDE = "optimized_attention_override"
    OBJECT_PATCH = "object_patch"
    MODEL_WEIGHT_PATCH = "model_weight_patch"
    EASYCACHE = "easycache"


@dataclass(frozen=True, slots=True)
class RegionalModelPatchInteropReport:
    """Publish immutable modifier categories preserved by one admitted MODEL."""

    model_family: RegionalModelFamily
    preserved_modifiers: tuple[RegionalPreservedModelModifier, ...]

    def __post_init__(self) -> None:
        """Require a typed family and canonical unique modifier order."""

        if not isinstance(self.model_family, RegionalModelFamily):
            raise TypeError("Regional interop report family has an invalid type.")
        if not isinstance(self.preserved_modifiers, tuple) or any(
            not isinstance(modifier, RegionalPreservedModelModifier)
            for modifier in self.preserved_modifiers
        ):
            raise TypeError("Regional interop report modifiers have invalid types.")
        if len(set(self.preserved_modifiers)) != len(self.preserved_modifiers):
            raise ValueError("Regional interop report modifiers must be unique.")

    @property
    def cache_modifier(self) -> RegionalPreservedModelModifier | None:
        """Return admitted EasyCache state when installed."""

        modifier = RegionalPreservedModelModifier.EASYCACHE
        if modifier in self.preserved_modifiers:
            return modifier
        return None


class RegionalModelPatchInteropValidator:
    """Admit preserved modifier surfaces and reject known branch collisions."""

    def validate(
        self,
        model: object,
        capabilities: RegionalModelCapabilities,
    ) -> RegionalModelPatchInteropReport:
        """Return preserved categories without changing supplied MODEL state."""

        if not isinstance(model, ModelPatcher):
            raise TypeError("Regional patch interop requires a Comfy ModelPatcher.")
        if not isinstance(capabilities, RegionalModelCapabilities):
            raise TypeError("Regional patch interop requires model capabilities.")

        model_options = _require_dictionary_attribute(model, "model_options")
        transformer_options = _require_nested_dictionary(
            model_options,
            "transformer_options",
            owner_label="MODEL",
        )
        wrappers = _require_wrapper_state(model)
        object_patches = _require_dictionary_attribute(model, "object_patches")
        model_weight_patches = _require_dictionary_attribute(model, "patches")
        patches = _require_optional_patch_state(transformer_options)

        self._reject_negpip(model_options, wrappers)
        cache_modifier = self._validate_cache_state(
            transformer_options,
            wrappers,
        )
        self._reject_attention_collisions(patches, capabilities)

        modifiers: list[RegionalPreservedModelModifier] = []
        model_wrapper = model_options.get("model_function_wrapper")
        if model_wrapper is not None:
            if not callable(model_wrapper):
                raise TypeError("MODEL model_function_wrapper must be callable.")
            modifiers.append(RegionalPreservedModelModifier.MODEL_FUNCTION_WRAPPER)
        if any(wrappers[wrapper_type] for wrapper_type in wrappers):
            if (
                WrappersMP.DIFFUSION_MODEL in wrappers
                and wrappers[WrappersMP.DIFFUSION_MODEL]
            ):
                modifiers.append(RegionalPreservedModelModifier.DIFFUSION_MODEL_WRAPPER)
        optimized_override = transformer_options.get("optimized_attention_override")
        if optimized_override is not None:
            if not callable(optimized_override):
                raise TypeError("MODEL optimized_attention_override must be callable.")
            modifiers.append(
                RegionalPreservedModelModifier.OPTIMIZED_ATTENTION_OVERRIDE
            )
        if object_patches:
            modifiers.append(RegionalPreservedModelModifier.OBJECT_PATCH)
        if model_weight_patches:
            modifiers.append(RegionalPreservedModelModifier.MODEL_WEIGHT_PATCH)
        if cache_modifier is not None:
            modifiers.append(cache_modifier)

        report = RegionalModelPatchInteropReport(
            capabilities.model_family,
            tuple(modifiers),
        )
        LOGGER.info(
            "Regional MODEL patch interoperability admitted",
            extra={
                "model_family": report.model_family.value,
                "preserved_modifiers": tuple(
                    modifier.value for modifier in report.preserved_modifiers
                ),
            },
        )
        return report

    def validate_execution(
        self,
        report: RegionalModelPatchInteropReport,
        processed_plan: ProcessedRegionalAttentionPlan,
        execution_mode: RegionalAttentionExecutionMode,
    ) -> None:
        """Reject cache state that cannot identify spatial or scheduled changes."""

        if not isinstance(report, RegionalModelPatchInteropReport):
            raise TypeError("Regional execution interop requires an admission report.")
        if not isinstance(processed_plan, ProcessedRegionalAttentionPlan):
            raise TypeError("Regional execution interop requires a processed plan.")
        if not isinstance(execution_mode, RegionalAttentionExecutionMode):
            raise TypeError("Regional execution interop requires an execution mode.")
        cache_modifier = report.cache_modifier
        if cache_modifier is None:
            return
        cache_name = cache_modifier.value
        if execution_mode is not RegionalAttentionExecutionMode.FULL:
            raise ValueError(
                f"Attention Coupling cannot compose {cache_name} with "
                f"{execution_mode.value} spatial views because the installed "
                "cache identity does not encode view coordinates. Remove the "
                "cache modifier before using this sampler."
            )
        if not processed_plan.is_time_invariant:
            raise ValueError(
                f"Attention Coupling cannot compose {cache_name} with scheduled "
                "regional execution because the installed cache identity does "
                "not encode regional context or LoRA schedule state. Remove the "
                "cache modifier or use time-invariant regional conditioning."
            )
        LOGGER.info(
            "Regional MODEL cache execution admitted",
            extra={
                "model_family": report.model_family.value,
                "cache_modifier": cache_name,
                "execution_mode": execution_mode.value,
            },
        )

    @staticmethod
    def _reject_negpip(
        model_options: dict[object, object],
        wrappers: dict[str, dict[object, list[object]]],
    ) -> None:
        """Reject installed NegPiP before its mask can enter branch packing."""

        marker = model_options.get(_NEGPIP_MODEL_OPTION, False)
        if not isinstance(marker, bool):
            raise TypeError("MODEL ppm_negpip marker must be boolean.")
        negpip_wrapper = bool(
            wrappers.get(WrappersMP.DIFFUSION_MODEL, {}).get(
                _NEGPIP_ANIMA_WRAPPER_KEY,
                (),
            )
        )
        if marker or negpip_wrapper:
            raise ValueError(
                "Attention Coupling does not support NegPiP because its attention "
                "mask is aligned to the ordinary conditioning batch rather than "
                "SimpleSyrup's regional branch batch. Remove CLIP NegPip before "
                "the Attention Coupling sampler."
            )

    @staticmethod
    def _validate_cache_state(
        transformer_options: dict[object, object],
        wrappers: dict[str, dict[object, list[object]]],
    ) -> RegionalPreservedModelModifier | None:
        """Require one complete core cache owner or no cache state."""

        easy_surfaces = (
            _has_wrapper(wrappers, WrappersMP.OUTER_SAMPLE, "easycache"),
            _has_wrapper(wrappers, WrappersMP.CALC_COND_BATCH, "easycache"),
            _has_wrapper(wrappers, WrappersMP.DIFFUSION_MODEL, "easycache"),
        )
        lazy_surfaces = (
            _has_wrapper(wrappers, WrappersMP.OUTER_SAMPLE, "lazycache"),
            _has_wrapper(wrappers, WrappersMP.PREDICT_NOISE, "lazycache"),
        )
        has_easy = any(easy_surfaces)
        has_lazy = any(lazy_surfaces)
        if has_easy and has_lazy:
            raise ValueError(
                "EasyCache and LazyCache cannot both own one MODEL; use exactly "
                "one cache modifier."
            )
        if has_easy and not all(easy_surfaces):
            raise ValueError("MODEL contains an incomplete EasyCache wrapper set.")
        if has_lazy and not all(lazy_surfaces):
            raise ValueError("MODEL contains an incomplete LazyCache wrapper set.")
        cache_state_exists = _EASYCACHE_OPTION in transformer_options
        if (has_easy or has_lazy) != cache_state_exists:
            raise ValueError(
                "MODEL cache wrappers and transformer option state are inconsistent."
            )
        if has_lazy:
            raise ValueError(
                "Attention Coupling does not support LazyCache because it skips "
                "complete denoiser evaluations and reuses previous-step prediction "
                "deltas, so skipped steps cannot execute exact regional "
                "conditioning or LoRA math. Remove LazyCache before the Attention "
                "Coupling sampler."
            )
        if has_easy:
            return RegionalPreservedModelModifier.EASYCACHE
        return None

    @staticmethod
    def _reject_attention_collisions(
        patches: dict[str, list[object]],
        capabilities: RegionalModelCapabilities,
    ) -> None:
        """Reject every populated attention surface owned by the backend."""

        conflicts = tuple(
            patch_name
            for conflict, patch_name in _ATTN2_PATCH_CONFLICTS.items()
            if conflict in capabilities.known_patch_conflicts
            and patches.get(patch_name)
        )
        if conflicts:
            raise ValueError(
                "Attention Coupling cannot compose with existing MODEL attention "
                f"patches: {', '.join(conflicts)}. Remove the upstream attention "
                "modifier; this backend requires exclusive branch-aware ownership."
            )


def _require_dictionary_attribute(
    owner: object,
    attribute_name: str,
) -> dict[object, object]:
    """Return one required mutable host dictionary without coercion."""

    value = getattr(owner, attribute_name, None)
    if not isinstance(value, dict):
        raise TypeError(f"MODEL {attribute_name} must be a dictionary.")
    return value


def _require_nested_dictionary(
    mapping: dict[object, object],
    key: str,
    *,
    owner_label: str,
) -> dict[object, object]:
    """Return one required nested host dictionary without changing it."""

    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{owner_label} {key} must be a dictionary.")
    return value


def _require_wrapper_state(
    model: ModelPatcher,
) -> dict[str, dict[object, list[object]]]:
    """Validate and return the installed keyed wrapper dictionary."""

    raw = getattr(model, "wrappers", None)
    if not isinstance(raw, dict):
        raise TypeError("MODEL wrappers must be a dictionary.")
    narrowed: dict[str, dict[object, list[object]]] = {}
    for wrapper_type, keyed in raw.items():
        if not isinstance(wrapper_type, str) or not isinstance(keyed, dict):
            raise TypeError("MODEL wrapper types must map to dictionaries.")
        narrowed_keyed: dict[object, list[object]] = {}
        for key, callbacks in keyed.items():
            if not isinstance(callbacks, list):
                raise TypeError("MODEL wrapper callbacks must be a list.")
            if any(not callable(callback) for callback in callbacks):
                raise TypeError("MODEL wrapper callbacks must be callable.")
            narrowed_keyed[key] = callbacks
        narrowed[wrapper_type] = narrowed_keyed
    return narrowed


def _require_optional_patch_state(
    transformer_options: dict[object, object],
) -> dict[str, list[object]]:
    """Validate and return the optional installed transformer patch lists."""

    raw = transformer_options.get("patches")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("MODEL transformer patches must be a dictionary.")
    narrowed: dict[str, list[object]] = {}
    for patch_name, callbacks in raw.items():
        if not isinstance(patch_name, str):
            raise TypeError("MODEL transformer patch names must be strings.")
        if not isinstance(callbacks, list):
            raise TypeError("MODEL transformer patch callbacks must be a list.")
        if any(not callable(callback) for callback in callbacks):
            raise TypeError("MODEL transformer patch callbacks must be callable.")
        narrowed[patch_name] = callbacks
    return narrowed


def _has_wrapper(
    wrappers: dict[str, dict[object, list[object]]],
    wrapper_type: str,
    key: str,
) -> bool:
    """Report whether one exact installed cache wrapper list is populated."""

    return bool(wrappers.get(wrapper_type, {}).get(key, ()))


REGIONAL_MODEL_PATCH_INTEROP_VALIDATOR = RegionalModelPatchInteropValidator()
