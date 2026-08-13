# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify atomic generic operation installation across Comfy state owners."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from simple_syrup.runtime.model_object_patch_batch import (
    ExactModelObjectReplacement,
    ModelObjectPatchBatchMutation,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.operation_cache_lifecycle import (
    RegionalOperationCacheLifecycle,
)
from simple_syrup.runtime.regional_lora.operation_installation_mutation import (
    RegionalOperationInstallationMutation,
)


class _Plan:
    """Expose one no-op release contract."""

    def clear(self) -> None:
        """Release no retained data."""


@pytest.mark.parametrize("failure", [RuntimeError("failure"), KeyboardInterrupt()])
def test_callback_failure_rolls_back_object_patches_and_callbacks(
    failure: BaseException,
) -> None:
    """Restore both state owners after host failure or interruption."""

    class FailingCallbackPatcher:
        """Wrap one real patcher and fail only callback installation."""

        def __init__(self) -> None:
            """Expose the exact state and methods consumed by the mutation."""

            self.real = _patcher()
            self.object_patches = self.real.object_patches
            self.object_patches_backup = self.real.object_patches_backup
            self.callbacks = self.real.callbacks

        def get_model_object(self, name: str) -> object:
            """Delegate exact object lookup."""

            return self.real.get_model_object(name)

        def add_object_patch(self, name: str, obj: object) -> None:
            """Delegate one pending exact object patch."""

            self.real.add_object_patch(name, obj)

        def get_callbacks(self, call_type: str, key: str) -> list[object]:
            """Delegate keyed callback lookup."""

            callbacks = self.real.get_callbacks(call_type, key)
            assert isinstance(callbacks, list)
            return callbacks

        def add_callback_with_key(
            self,
            call_type: str,
            key: str,
            callback: object,
        ) -> None:
            """Emulate failure after object-patch installation."""

            del call_type, key, callback
            raise failure

    model = FailingCallbackPatcher()
    before_callbacks = {
        event: {key: values.copy() for key, values in keyed.items()}
        for event, keyed in model.callbacks.items()
    }
    expected = model.get_model_object("linear")
    mutation = RegionalOperationInstallationMutation(
        ModelObjectPatchBatchMutation(
            (
                ExactModelObjectReplacement(
                    "linear",
                    expected,
                    nn.Identity(),
                ),
            )
        ),
        RegionalOperationCacheLifecycle((_Plan(),), RegionalLoraExecutionCache()),
    )

    with pytest.raises(type(failure)):
        mutation.apply(model)

    assert model.object_patches == {}
    assert model.callbacks == before_callbacks


def _patcher() -> Any:
    """Return one real Comfy patcher around an exact module path."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(nn.ModuleDict({"linear": nn.Linear(2, 2)}), device, device)
