# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve regional WeightHook payloads through installed Comfy contracts."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

import comfy.lora
import comfy.model_patcher

from ...domain.regional_lora_plan import RegionalLoraAdapterPlan
from ..regional_lora_host_payload import RegionalLoraHostPayload
from ..regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from .comfy_adapter_evidence import (
    classify_initialized_sources,
    classify_raw_sources,
    issue,
    normalized_targets,
    source_keys,
)
from .comfy_adapter_resolution import (
    ComfyAdapterResolutionIssueCode,
    ComfyAdapterSourceEntry,
    ComfyAdapterSourceScope,
    ComfyRegionalAdapterResolution,
    ComfyRegionalLoraResolution,
)

LOGGER = logging.getLogger(__name__)


class ComfyRegionalAdapterResolver:
    """Resolve regional model LoRAs without model, hook, or tensor mutation."""

    def resolve(
        self,
        adaptation: RegionalLoraPlanAdaptation,
        *,
        model: object,
        clip_model: object | None = None,
        vae_key_map: Mapping[str, object] | None = None,
    ) -> ComfyRegionalLoraResolution:
        """Resolve every ordered payload and aggregate adapter-scoped failures."""

        base_model = _base_model(model)
        results = tuple(
            self._resolve_adapter(
                adapter,
                payload,
                model=base_model,
                clip_model=clip_model,
                vae_key_map=vae_key_map,
            )
            for adapter, payload in zip(
                adaptation.plan.adapters,
                adaptation.adapter_payloads,
                strict=True,
            )
        )
        return ComfyRegionalLoraResolution(
            adapters=results,
            issues=tuple(issue for result in results for issue in result.issues),
        )

    def _resolve_adapter(
        self,
        adapter: RegionalLoraAdapterPlan,
        payload: RegionalLoraHostPayload,
        *,
        model: object,
        clip_model: object | None,
        vae_key_map: Mapping[str, object] | None,
    ) -> ComfyRegionalAdapterResolution:
        """Resolve one payload while converting exceptions into scoped evidence."""

        try:
            return self._resolve_adapter_or_raise(
                adapter,
                payload,
                model=model,
                clip_model=clip_model,
                vae_key_map=vae_key_map,
            )
        except Exception as error:
            LOGGER.exception(
                "Comfy regional adapter resolution failed",
                extra={
                    "adapter_identity": adapter.adapter_identity.value,
                    "composition_index": adapter.composition_index,
                    "region_index": adapter.region_index,
                    "branch": adapter.branch.value,
                },
            )
            resolution_issue = issue(
                adapter,
                ComfyAdapterResolutionIssueCode.RESOLUTION_FAILED,
                f"Comfy regional adapter resolution failed: {error}",
            )
            return ComfyRegionalAdapterResolution(
                adapter=adapter,
                payload=payload,
                model_targets=(),
                source_entries=tuple(
                    ComfyAdapterSourceEntry(
                        source_key=source_key,
                        scope=ComfyAdapterSourceScope.UNRESOLVED,
                    )
                    for source_key in source_keys(payload.raw_weights)
                ),
                issues=(resolution_issue,),
            )

    def _resolve_adapter_or_raise(
        self,
        adapter: RegionalLoraAdapterPlan,
        payload: RegionalLoraHostPayload,
        *,
        model: object,
        clip_model: object | None,
        vae_key_map: Mapping[str, object] | None,
    ) -> ComfyRegionalAdapterResolution:
        """Use only Comfy key maps and decoding for one valid host payload."""

        if payload.needs_resolution:
            raw_weights = _require_source_mapping(payload.raw_weights)
            model_weights = comfy.lora.load_lora(
                raw_weights,
                comfy.lora.model_lora_keys_unet(model, {}),
                log_missing=False,
            )
            clip_weights = (
                comfy.lora.load_lora(
                    raw_weights,
                    comfy.lora.model_lora_keys_clip(clip_model, {}),
                    log_missing=False,
                )
                if clip_model is not None
                else {}
            )
            vae_weights = (
                comfy.lora.load_lora(
                    raw_weights,
                    dict(vae_key_map),
                    log_missing=False,
                )
                if vae_key_map is not None
                else {}
            )
            source_entries = classify_raw_sources(
                raw_weights,
                model_weights=model_weights,
                clip_weights=clip_weights,
                vae_weights=vae_weights,
            )
        else:
            model_weights = _require_mapping(payload.model_weights)
            clip_weights = _optional_mapping(payload.clip_weights)
            source_entries = classify_initialized_sources(
                model_weights=model_weights,
                clip_weights=clip_weights,
            )

        targets, issues = normalized_targets(adapter, model_weights)
        return ComfyRegionalAdapterResolution(
            adapter=adapter,
            payload=payload,
            model_targets=targets,
            source_entries=source_entries,
            issues=issues,
        )


def _require_mapping(value: object | None) -> Mapping[object, object]:
    """Require the mapping contract consumed and emitted by Comfy's resolver."""

    if not isinstance(value, Mapping):
        raise TypeError("Regional LoRA host payload weights must be a mapping.")
    return value


def _base_model(model: object) -> object:
    """Return the exact base model owned by one Comfy MODEL object."""

    if not isinstance(model, comfy.model_patcher.ModelPatcher):
        raise TypeError("Regional LoRA resolution requires a Comfy ModelPatcher.")
    return model.model


def _require_source_mapping(value: object | None) -> Mapping[str, object]:
    """Require the string-keyed raw source contract consumed by Comfy."""

    weights = _require_mapping(value)
    if any(not isinstance(key, str) for key in weights):
        raise TypeError("Regional LoRA raw source keys must be strings.")
    return cast(Mapping[str, object], weights)


def _optional_mapping(value: object | None) -> Mapping[object, object]:
    """Accept an absent initialized side payload as one empty mapping."""

    return {} if value is None else _require_mapping(value)


COMFY_REGIONAL_ADAPTER_RESOLVER = ComfyRegionalAdapterResolver()
