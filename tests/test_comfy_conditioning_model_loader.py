"""Verify official Comfy model loading before Anima conditioning processing."""

from __future__ import annotations

from typing import Any

from simple_syrup.runtime.comfy_conditioning_model_loader import (
    COMFY_CONDITIONING_MODEL_LOADER,
)


def test_loader_uses_official_comfy_model_residency_boundary(monkeypatch: Any) -> None:
    """Load the exact patcher so Anima preprocessing weights match its CUDA inputs."""

    model = object()
    calls: list[object] = []
    monkeypatch.setattr(
        "simple_syrup.runtime.comfy_conditioning_model_loader."
        "model_management.load_model_gpu",
        calls.append,
    )

    COMFY_CONDITIONING_MODEL_LOADER.load(model)

    assert calls == [model]
