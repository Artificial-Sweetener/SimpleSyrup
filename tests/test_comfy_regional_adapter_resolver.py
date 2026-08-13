# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify Comfy-owned regional adapter resolution and complete accounting."""

from __future__ import annotations

from collections.abc import Mapping

import comfy.lora
import comfy.model_patcher
import pytest
import torch
from comfy.weight_adapter.lora import LoRAAdapter

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.runtime.regional_lora.comfy_adapter_resolution import (
    ComfyAdapterResolutionIssueCode,
    ComfyAdapterSourceScope,
)
from simple_syrup.runtime.regional_lora.comfy_adapter_resolver import (
    ComfyRegionalAdapterResolver,
)
from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload
from simple_syrup.runtime.regional_lora_plan_adapter import (
    RegionalLoraPlanAdaptation,
)


def test_resolver_uses_comfy_maps_and_accounts_for_every_raw_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate naming to Comfy and classify model, clip, VAE, and unused keys."""

    raw_weights = {
        "model.up": object(),
        "model.down": object(),
        "clip.up": object(),
        "clip.down": object(),
        "vae.diff": object(),
        "unused": object(),
    }
    model_operation = _lora("model.up", "model.down")
    clip_operation = _lora("clip.up", "clip.down")
    model_path = ("diffusion_model.layer.weight", (0, 4, 8))
    calls: list[tuple[object, Mapping[object, object], bool]] = []
    base_model = torch.nn.Linear(1, 1)
    model = _patcher(base_model)
    clip_model = object()
    model_map = {"model": model_path}
    clip_map = {"clip": "text_model.layer.weight"}
    vae_map = {"vae": "decoder.layer.weight"}

    monkeypatch.setattr(
        comfy.lora,
        "model_lora_keys_unet",
        lambda observed, key_map: model_map if observed is base_model else key_map,
    )
    monkeypatch.setattr(
        comfy.lora,
        "model_lora_keys_clip",
        lambda observed, key_map: clip_map if observed is clip_model else key_map,
    )

    def load_lora(
        weights: Mapping[str, object],
        key_map: Mapping[object, object],
        *,
        log_missing: bool,
    ) -> Mapping[object, object]:
        calls.append((weights, key_map, log_missing))
        if key_map is model_map:
            return {model_path: model_operation}
        if key_map is clip_map:
            return {"text_model.layer.weight": clip_operation}
        return {"decoder.layer.weight": ("diff", (raw_weights["vae.diff"],))}

    monkeypatch.setattr(comfy.lora, "load_lora", load_lora)
    adaptation = _adaptation((RegionalLoraHostPayload.unresolved(raw_weights),))

    result = ComfyRegionalAdapterResolver().resolve(
        adaptation,
        model=model,
        clip_model=clip_model,
        vae_key_map=vae_map,
    )

    assert [call[0] is raw_weights for call in calls] == [True, True, True]
    assert [call[1] for call in calls] == [model_map, clip_map, vae_map]
    assert [call[2] for call in calls] == [False, False, False]
    target = result.adapters[0].model_targets[0]
    assert target.path.parameter_key == "diffusion_model.layer.weight"
    assert target.path.offset == (0, 4, 8)
    assert target.operation is model_operation
    assert target.source_keys == ("model.down", "model.up")
    assert target.ordinary_additive_lora is True
    assert result.adapters[0].payload is adaptation.adapter_payloads[0]
    assert {
        entry.source_key: entry.scope for entry in result.adapters[0].source_entries
    } == {
        "model.up": ComfyAdapterSourceScope.MODEL,
        "model.down": ComfyAdapterSourceScope.MODEL,
        "clip.up": ComfyAdapterSourceScope.TEXT_ENCODER,
        "clip.down": ComfyAdapterSourceScope.TEXT_ENCODER,
        "vae.diff": ComfyAdapterSourceScope.VAE,
        "unused": ComfyAdapterSourceScope.UNUSED,
    }
    assert result.issues == ()
    assert adaptation.adapter_payloads[0].raw_weights is raw_weights


def test_resolver_consumes_initialized_payloads_by_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypass key mapping and decoding after Comfy has initialized a hook."""

    model_operation = _lora("model.up", "model.down")
    clip_operation = _lora("clip.up", "clip.down")
    model_weights = {"diffusion_model.layer.weight": model_operation}
    clip_weights = {"text_model.layer.weight": clip_operation}
    payload = RegionalLoraHostPayload(False, None, model_weights, clip_weights)
    monkeypatch.setattr(
        comfy.lora,
        "load_lora",
        lambda *args, **kwargs: pytest.fail("Initialized payload was decoded again."),
    )
    monkeypatch.setattr(
        comfy.lora,
        "model_lora_keys_unet",
        lambda *args, **kwargs: pytest.fail("Initialized payload was remapped."),
    )

    result = ComfyRegionalAdapterResolver().resolve(
        _adaptation((payload,)),
        model=_patcher(torch.nn.Linear(1, 1)),
    )

    target = result.adapters[0].model_targets[0]
    assert target.operation is model_operation
    assert result.adapters[0].payload.model_weights is model_weights
    assert result.adapters[0].payload.clip_weights is clip_weights
    assert [
        (entry.source_key, entry.scope) for entry in result.adapters[0].source_entries
    ] == [
        ("model.down", ComfyAdapterSourceScope.MODEL),
        ("model.up", ComfyAdapterSourceScope.MODEL),
        ("clip.down", ComfyAdapterSourceScope.TEXT_ENCODER),
        ("clip.up", ComfyAdapterSourceScope.TEXT_ENCODER),
    ]


def test_resolver_preserves_normalized_lora_metadata_by_identity() -> None:
    """Retain alpha, middle, reshape, and tensor identities for U3 translation."""

    up = torch.ones((2, 1))
    down = torch.ones((1, 2))
    middle = torch.ones((1, 1, 1, 1))
    reshape = [2, 2]
    operation = LoRAAdapter(
        {"up", "down", "middle", "reshape"},
        (up, down, 0.75, middle, None, reshape),
    )
    payload = RegionalLoraHostPayload(
        False,
        None,
        {"diffusion_model.layer.weight": operation},
        None,
    )

    result = ComfyRegionalAdapterResolver().resolve(
        _adaptation((payload,)),
        model=_patcher(torch.nn.Linear(1, 1)),
    )

    retained = result.adapters[0].model_targets[0].operation
    assert retained is operation
    assert isinstance(retained, LoRAAdapter)
    assert retained.weights[0] is up
    assert retained.weights[1] is down
    assert retained.weights[2] == 0.75
    assert retained.weights[3] is middle
    assert retained.weights[4] is None
    assert retained.weights[5] is reshape


def test_resolver_reports_dora_and_every_other_operation_explicitly() -> None:
    """Admit only ordinary additive LoRA while retaining unsupported operations."""

    dora = _lora("dora.up", "dora.down", dora_scale=torch.ones(1))
    diff = ("diff", (torch.ones(1),))
    payload = RegionalLoraHostPayload(
        False,
        None,
        {
            "diffusion_model.dora.weight": dora,
            "diffusion_model.diff.weight": diff,
        },
        None,
    )

    result = ComfyRegionalAdapterResolver().resolve(
        _adaptation((payload,)),
        model=_patcher(torch.nn.Linear(1, 1)),
    )

    assert [target.operation for target in result.adapters[0].model_targets] == [
        dora,
        diff,
    ]
    assert [
        target.ordinary_additive_lora for target in result.adapters[0].model_targets
    ] == [
        False,
        False,
    ]
    assert [issue.code for issue in result.issues] == [
        ComfyAdapterResolutionIssueCode.DORA_UNSUPPORTED,
        ComfyAdapterResolutionIssueCode.UNSUPPORTED_OPERATION,
    ]


def test_resolver_aggregates_failures_without_hiding_other_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return all ordered adapter failures instead of stopping at the first one."""

    invalid = RegionalLoraHostPayload.unresolved(object())
    unsupported = RegionalLoraHostPayload(
        False,
        None,
        {"diffusion_model.layer.weight": ("set", (object(),))},
        None,
    )
    monkeypatch.setattr(comfy.lora, "model_lora_keys_unet", lambda model, key_map: {})

    result = ComfyRegionalAdapterResolver().resolve(
        _adaptation((invalid, unsupported)),
        model=_patcher(torch.nn.Linear(1, 1)),
    )

    assert len(result.adapters) == 2
    assert [issue.composition_index for issue in result.issues] == [0, 1]
    assert [issue.code for issue in result.issues] == [
        ComfyAdapterResolutionIssueCode.RESOLUTION_FAILED,
        ComfyAdapterResolutionIssueCode.UNSUPPORTED_OPERATION,
    ]
    assert [issue.adapter_identity for issue in result.issues] == [
        "adapter-0.safetensors",
        "adapter-1.safetensors",
    ]


def test_resolver_reports_invalid_comfy_target_paths() -> None:
    """Reject malformed host target paths without discarding valid siblings."""

    operation = _lora("up", "down")
    payload = RegionalLoraHostPayload(
        False,
        None,
        {3: operation, "diffusion_model.valid.weight": operation},
        None,
    )

    result = ComfyRegionalAdapterResolver().resolve(
        _adaptation((payload,)),
        model=_patcher(torch.nn.Linear(1, 1)),
    )

    assert [
        target.path.parameter_key for target in result.adapters[0].model_targets
    ] == ["diffusion_model.valid.weight"]
    assert [issue.code for issue in result.issues] == [
        ComfyAdapterResolutionIssueCode.INVALID_TARGET_PATH
    ]


def test_resolver_rejects_non_comfy_model_boundaries() -> None:
    """Require the same MODEL object contract supplied by Comfy workflows."""

    with pytest.raises(TypeError, match="requires a Comfy ModelPatcher"):
        ComfyRegionalAdapterResolver().resolve(
            _adaptation(()),
            model=object(),
        )


def _adaptation(
    payloads: tuple[RegionalLoraHostPayload, ...],
) -> RegionalLoraPlanAdaptation:
    """Return one valid ordered adaptation for resolver characterization."""

    adapters = tuple(
        RegionalLoraAdapterPlan(
            adapter_identity=RegionalLoraAdapterIdentity(
                f"adapter-{index}.safetensors"
            ),
            composition_index=index,
            region_index=index,
            branch=RegionalLoraBranch.POSITIVE,
            model_strength=0.8,
            schedule=(RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
        )
        for index in range(len(payloads))
    )
    return RegionalLoraPlanAdaptation(RegionalLoraPlan(adapters), payloads)


def _patcher(model: object) -> comfy.model_patcher.ModelPatcher:
    """Wrap one test base model in the installed Comfy MODEL boundary."""

    return comfy.model_patcher.ModelPatcher(
        model, torch.device("cpu"), torch.device("cpu")
    )


def _lora(
    up_key: str,
    down_key: str,
    *,
    dora_scale: torch.Tensor | None = None,
) -> LoRAAdapter:
    """Return one installed Comfy ordinary or DoRA normalized adapter."""

    return LoRAAdapter(
        {up_key, down_key},
        (torch.ones((2, 1)), torch.ones((1, 2)), 1.0, None, dora_scale, None),
    )
