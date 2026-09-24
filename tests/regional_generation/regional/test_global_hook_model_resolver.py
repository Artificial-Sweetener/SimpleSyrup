# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify pre-derivation Comfy MODEL selection for global prompt hooks."""

from __future__ import annotations

from comfy.hooks import HookGroup

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.runtime.global_hook_model_resolver import GlobalHookModelResolver


class _Model:
    """Expose the dynamic-model boundary used by Comfy's CFG guider."""

    def __init__(self, *, dynamic: bool, delegate: object | None = None) -> None:
        """Retain configured dynamic state and delegate result."""

        self.dynamic = dynamic
        self.delegate = delegate
        self.delegate_calls = 0

    def is_dynamic(self) -> bool:
        """Return the configured Comfy model mode."""

        return self.dynamic

    def get_non_dynamic_delegate(self) -> object:
        """Return and record the configured static delegate."""

        self.delegate_calls += 1
        return self.delegate


def test_unhooked_global_entry_preserves_dynamic_model() -> None:
    """Leave regional-only hooks to the existing custom regional runtime."""

    model = _Model(dynamic=True)
    conditioning = ConditioningBatch((_conditioning(), _conditioning(HookGroup())))

    resolved = GlobalHookModelResolver().resolve(
        model,
        positive=conditioning,
        negative=conditioning,
    )

    assert resolved is model
    assert model.delegate_calls == 0


def test_global_hook_preserves_already_static_model() -> None:
    """Avoid unnecessary model replacement when hook execution is already static."""

    model = _Model(dynamic=False)

    resolved = GlobalHookModelResolver().resolve(
        model,
        positive=_conditioning(HookGroup()),
        negative=_conditioning(HookGroup()),
    )

    assert resolved is model
    assert model.delegate_calls == 0


def test_global_hook_selects_static_delegate_before_regional_derivation() -> None:
    """Bind regional wrappers to the same static graph Comfy samples with hooks."""

    delegate = _Model(dynamic=False)
    model = _Model(dynamic=True, delegate=delegate)

    resolved = GlobalHookModelResolver().resolve(
        model,
        positive=ConditioningBatch(
            (_conditioning(HookGroup()), _conditioning(HookGroup()))
        ),
        negative=ConditioningBatch(
            (_conditioning(HookGroup()), _conditioning(HookGroup()))
        ),
    )

    assert resolved is delegate
    assert model.delegate_calls == 1


def _conditioning(hooks: HookGroup | None = None) -> list[list[object]]:
    """Return one standard conditioning with optional model hooks."""

    metadata: dict[str, object] = {}
    if hooks is not None:
        metadata["hooks"] = hooks
    return [["embedding", metadata]]
