# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for SimpleSyrup's shared torch model device manager."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, cast

import pytest
import torch

from simple_syrup.runtime.model_device_manager import (
    TorchModelDeviceManager,
    external_model_inference,
)


class RecordingModel:
    """Small PyTorch-style model double."""

    def __init__(self, device: str | None = None) -> None:
        """Create a model that records device movement."""

        self.device = torch.device(device) if device is not None else None
        self.to_calls: list[str] = []
        self.eval_calls = 0

    def to(self, device: str | torch.device) -> None:
        """Record model movement."""

        self.device = torch.device(device)
        self.to_calls.append(str(device))

    def eval(self) -> None:
        """Record eval mode selection."""

        self.eval_calls += 1


def test_manager_cpu_policy_does_not_call_comfy_gpu_loader() -> None:
    """CPU inference keeps the model on CPU without Comfy GPU loading."""

    model = RecordingModel()
    managed = TorchModelDeviceManager().manage(model, "model", "source")

    with TorchModelDeviceManager().inference(managed, "cpu") as loaded:
        assert loaded.device == torch.device("cpu")
        assert loaded.model is model

    assert model.to_calls == ["cpu"]
    assert model.eval_calls >= 2


def test_manager_auto_policy_uses_comfy_model_patcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto inference loads owned models through Comfy's model manager."""

    state: dict[str, object] = {"loaded": []}

    class FakeModelPatcher:
        """Fake Comfy ModelPatcher boundary."""

        def __init__(
            self,
            model: object,
            load_device: torch.device,
            offload_device: torch.device,
        ) -> None:
            """Record patcher construction and mimic Comfy's device attribute."""

            self.model = model
            self.load_device = load_device
            self.offload_device = offload_device
            cast(Any, model).device = load_device

    comfy = ModuleType("comfy")
    model_management = ModuleType("comfy.model_management")
    model_patcher = ModuleType("comfy.model_patcher")
    model_management.get_torch_device = lambda: torch.device("cuda:0")  # type: ignore[attr-defined]
    model_management.text_encoder_offload_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    model_management.load_model_gpu = lambda patcher: state["loaded"].append(patcher)  # type: ignore[attr-defined]
    model_patcher.ModelPatcher = FakeModelPatcher  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "comfy.model_patcher", model_patcher)

    model = RecordingModel("cpu")
    manager = TorchModelDeviceManager()
    managed = manager.manage(model, "model", "source")

    with manager.inference(managed, "auto") as loaded:
        assert loaded.device == torch.device("cuda:0")

    loaded_patchers = state["loaded"]
    assert isinstance(loaded_patchers, list)
    assert len(loaded_patchers) == 1
    assert managed.patcher is loaded_patchers[0]


def test_manager_auto_policy_bypasses_patcher_for_read_only_device_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models like SAM with read-only `device` properties use bounded `.to(...)`."""

    class ReadOnlyDeviceModel:
        """PyTorch-like model with a read-only device property."""

        def __init__(self) -> None:
            """Create a model on CPU."""

            self._device = torch.device("cpu")
            self.to_calls: list[str] = []
            self.eval_calls = 0

        @property
        def device(self) -> torch.device:
            """Return the current device without allowing assignment."""

            return self._device

        def to(self, device: str | torch.device) -> None:
            """Record model movement."""

            self._device = torch.device(device)
            self.to_calls.append(str(device))

        def eval(self) -> None:
            """Record eval mode selection."""

            self.eval_calls += 1

    state: dict[str, object] = {"loaded": [], "emptied": 0}

    comfy = ModuleType("comfy")
    model_management = ModuleType("comfy.model_management")
    model_patcher = ModuleType("comfy.model_patcher")
    model_management.get_torch_device = lambda: torch.device("cuda:0")  # type: ignore[attr-defined]
    model_management.text_encoder_offload_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    model_management.load_model_gpu = lambda patcher: state["loaded"].append(patcher)  # type: ignore[attr-defined]
    model_management.soft_empty_cache = lambda: state.__setitem__("emptied", 1)  # type: ignore[attr-defined]
    model_patcher.ModelPatcher = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "comfy.model_patcher", model_patcher)

    model = ReadOnlyDeviceModel()
    manager = TorchModelDeviceManager()
    managed = manager.manage(model, "sam", "source")

    with manager.inference(managed, "auto") as loaded:
        assert loaded.device == torch.device("cuda:0")
        assert model.device == torch.device("cuda:0")

    assert state["loaded"] == []
    assert state["emptied"] == 1
    assert managed.patcher is None
    assert model.device == torch.device("cpu")
    assert model.to_calls == ["cuda:0", "cpu"]


def test_external_model_inference_restores_original_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External raw model movement is bounded to the inference context."""

    comfy = ModuleType("comfy")
    model_management = ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cuda:0")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    model = RecordingModel("cpu")

    with external_model_inference(model, "auto") as loaded:
        assert loaded.device == torch.device("cuda:0")
        assert model.device == torch.device("cuda:0")

    assert model.device == torch.device("cpu")
    assert model.to_calls == ["cuda:0", "cpu"]
