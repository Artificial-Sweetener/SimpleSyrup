# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve regional WeightHook payloads through installed Comfy contracts."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(slots=True)
class _ResolutionKeyMaps:
    """Build each successful host key map once per complete resolution."""

    model: object
    clip_model: object | None
    _model_map: Mapping[object, object] | None = None
    _clip_map: Mapping[object, object] | None = None

    def model_map(self) -> Mapping[object, object]:
        """Return the one cached UNet key map after successful construction."""

        if self._model_map is None:
            self._model_map = comfy.lora.model_lora_keys_unet(self.model, {})
        return self._model_map

    def clip_map(self) -> Mapping[object, object]:
        """Return the one cached CLIP key map after successful construction."""

        if self.clip_model is None:
            return {}
        if self._clip_map is None:
            self._clip_map = comfy.lora.model_lora_keys_clip(self.clip_model, {})
        return self._clip_map


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
        key_maps = _ResolutionKeyMaps(base_model, clip_model)
        interned: list[
            tuple[RegionalLoraHostPayload, ComfyRegionalAdapterResolution]
        ] = []
        results: list[ComfyRegionalAdapterResolution] = []
        for adapter, payload in zip(
            adaptation.plan.adapters,
            adaptation.adapter_payloads,
            strict=True,
        ):
            cached = next(
                (
                    result
                    for existing_payload, result in interned
                    if _same_payload(existing_payload, payload)
                ),
                None,
            )
            if cached is None:
                resolved = self._resolve_adapter(
                    adapter,
                    payload,
                    key_maps=key_maps,
                    vae_key_map=vae_key_map,
                )
                interned.append((payload, resolved))
            else:
                resolved = _rebind_resolution(cached, adapter, payload)
            results.append(resolved)
        return ComfyRegionalLoraResolution(
            adapters=tuple(results),
            issues=tuple(issue for result in results for issue in result.issues),
        )

    def _resolve_adapter(
        self,
        adapter: RegionalLoraAdapterPlan,
        payload: RegionalLoraHostPayload,
        *,
        key_maps: _ResolutionKeyMaps,
        vae_key_map: Mapping[str, object] | None,
    ) -> ComfyRegionalAdapterResolution:
        """Resolve one payload while converting exceptions into scoped evidence."""

        try:
            return self._resolve_adapter_or_raise(
                adapter,
                payload,
                key_maps=key_maps,
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
        key_maps: _ResolutionKeyMaps,
        vae_key_map: Mapping[str, object] | None,
    ) -> ComfyRegionalAdapterResolution:
        """Use only Comfy key maps and decoding for one valid host payload."""

        if payload.needs_resolution:
            raw_weights = _require_source_mapping(payload.raw_weights)
            model_weights = comfy.lora.load_lora(
                raw_weights,
                key_maps.model_map(),
                log_missing=False,
            )
            clip_weights = (
                comfy.lora.load_lora(
                    raw_weights,
                    key_maps.clip_map(),
                    log_missing=False,
                )
                if key_maps.clip_model is not None
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


def _same_payload(
    left: RegionalLoraHostPayload,
    right: RegionalLoraHostPayload,
) -> bool:
    """Match only exact host payload references and resolution representation."""

    return (
        left.needs_resolution is right.needs_resolution
        and left.raw_weights is right.raw_weights
        and left.model_weights is right.model_weights
        and left.clip_weights is right.clip_weights
    )


def _rebind_resolution(
    cached: ComfyRegionalAdapterResolution,
    adapter: RegionalLoraAdapterPlan,
    payload: RegionalLoraHostPayload,
) -> ComfyRegionalAdapterResolution:
    """Reuse immutable payload evidence while preserving authored issue scope."""

    return ComfyRegionalAdapterResolution(
        adapter=adapter,
        payload=payload,
        model_targets=cached.model_targets,
        source_entries=cached.source_entries,
        issues=tuple(
            issue(adapter, observed.code, observed.message)
            for observed in cached.issues
        ),
    )


COMFY_REGIONAL_ADAPTER_RESOLVER = ComfyRegionalAdapterResolver()
