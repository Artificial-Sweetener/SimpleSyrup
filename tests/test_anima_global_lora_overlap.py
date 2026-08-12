# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact static-global and regional Anima LoRA overlap rejection."""

from __future__ import annotations

from types import SimpleNamespace

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
from simple_syrup.runtime.regional_lora.anima_global_lora_overlap import (
    AnimaGlobalRegionalLoraOverlapError,
    AnimaGlobalRegionalLoraOverlapValidator,
)
from simple_syrup.runtime.regional_lora.anima_plan_admission import (
    AnimaRegionalLoraAdapterAdmission,
    AnimaRegionalLoraPlanAdmission,
)
from simple_syrup.runtime.regional_lora.anima_targets import (
    AnimaLoraAdmission,
    AnimaLoraTarget,
    AnimaLoraTargetFamily,
    anima_lora_target_name,
    expected_anima_lora_features,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget


def test_overlap_validator_accepts_empty_distinct_and_partial_global_state() -> None:
    """Preserve unpatched and content-distinct global MODEL LoRAs."""

    admission, targets = _admission(target_count=2)
    validator = AnimaGlobalRegionalLoraOverlapValidator()

    validator.validate(SimpleNamespace(patches={}), admission)
    validator.validate(
        SimpleNamespace(patches={_key(targets[0]): [_patch(targets[0])]}),
        admission,
    )
    changed = targets[0].up.clone()
    changed[0, 0] += 1.0
    validator.validate(
        SimpleNamespace(
            patches={
                _key(targets[0]): [_patch(targets[0], up=changed)],
                _key(targets[1]): [_patch(targets[1])],
            }
        ),
        admission,
    )


@pytest.mark.parametrize("clone_tensors", [False, True], ids=("identity", "content"))
def test_overlap_validator_rejects_complete_exact_global_content(
    clone_tensors: bool,
) -> None:
    """Reject exact adapter content with identity and cloned-tensor paths."""

    admission, targets = _admission(target_count=2)
    patches = {
        _key(target): [
            _patch(
                target,
                down=target.down.clone() if clone_tensors else target.down,
                up=target.up.clone() if clone_tensors else target.up,
            )
        ]
        for target in targets
    }

    with pytest.raises(
        AnimaGlobalRegionalLoraOverlapError,
        match="already applied globally.*regional.safetensors",
    ):
        AnimaGlobalRegionalLoraOverlapValidator().validate(
            SimpleNamespace(patches=patches),
            admission,
        )


@pytest.mark.parametrize(
    "patch_factory",
    (
        lambda target: _patch(target, strength=0.0),
        lambda target: (1.0, ("diff", (target.up,)), 1.0, None, None),
        lambda target: _patch(target, alpha=1.0),
    ),
    ids=("zero-strength", "non-lora", "alpha-form"),
)
def test_overlap_validator_preserves_noncomparable_global_patches(
    patch_factory: object,
) -> None:
    """Keep inactive and other global patch formats unchanged."""

    admission, targets = _admission(target_count=1)
    factory = patch_factory
    assert callable(factory)

    AnimaGlobalRegionalLoraOverlapValidator().validate(
        SimpleNamespace(patches={_key(targets[0]): [factory(targets[0])]}),
        admission,
    )


@pytest.mark.parametrize(
    ("patches", "message"),
    (
        (None, "requires MODEL patches"),
        ({"diffusion_model.blocks.0.self_attn.q_proj.weight": object()}, "list"),
        ({"diffusion_model.blocks.0.self_attn.q_proj.weight": [object()]}, "tuple"),
        (
            {
                "diffusion_model.blocks.0.self_attn.q_proj.weight": [
                    (float("nan"), object(), 1.0)
                ]
            },
            "finite",
        ),
    ),
)
def test_overlap_validator_fails_closed_on_malformed_model_patch_state(
    patches: object,
    message: str,
) -> None:
    """Reject installed-host patch drift before regional execution setup."""

    admission, _ = _admission(target_count=1)

    with pytest.raises((TypeError, ValueError), match=message):
        AnimaGlobalRegionalLoraOverlapValidator().validate(
            SimpleNamespace(patches=patches),
            admission,
        )


def _admission(
    *, target_count: int
) -> tuple[AnimaRegionalLoraPlanAdmission, tuple[StandardLoraTarget, ...]]:
    """Build one admitted regional adapter with small valid Anima targets."""

    families = tuple(AnimaLoraTargetFamily)[:target_count]
    targets = tuple(_target(family) for family in families)
    plan_entry = RegionalLoraAdapterPlan(
        adapter_identity=RegionalLoraAdapterIdentity("regional.safetensors"),
        composition_index=0,
        region_index=0,
        branch=RegionalLoraBranch.POSITIVE,
        model_strength=0.8,
        schedule=(RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
    )
    plan = RegionalLoraPlan((plan_entry,))
    admission = AnimaLoraAdmission(
        tuple(
            AnimaLoraTarget(0, family, target)
            for family, target in zip(families, targets, strict=True)
        )
    )
    return (
        AnimaRegionalLoraPlanAdmission(
            plan,
            (AnimaRegionalLoraAdapterAdmission(plan_entry, admission),),
        ),
        targets,
    )


def _target(family: AnimaLoraTargetFamily) -> StandardLoraTarget:
    """Return one rank-one target with installed Anima feature dimensions."""

    input_features, output_features = expected_anima_lora_features(family)
    return StandardLoraTarget(
        target=anima_lora_target_name(0, family),
        down=torch.arange(input_features, dtype=torch.float32).reshape(1, -1),
        up=torch.arange(output_features, dtype=torch.float32).reshape(-1, 1),
        rank=1,
        input_features=input_features,
        output_features=output_features,
    )


def _key(target: StandardLoraTarget) -> str:
    """Return the installed Comfy MODEL patch key for one target."""

    return f"{target.target}.weight"


def _patch(
    target: StandardLoraTarget,
    *,
    strength: float = 1.0,
    down: torch.Tensor | None = None,
    up: torch.Tensor | None = None,
    alpha: float | None = None,
) -> tuple[object, ...]:
    """Return one installed-Comfy standard static LoRA patch entry."""

    adapter = LoRAAdapter(
        set(),
        (
            target.up if up is None else up,
            target.down if down is None else down,
            alpha,
            None,
            None,
            None,
        ),
    )
    return (strength, adapter, 1.0, None, None)
