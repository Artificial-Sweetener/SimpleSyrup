# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the separate Simple Load FLUX and FLUX.2 v3 nodes."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from simple_syrup.nodes_v3.simple_load_flux import SimpleLoadFluxV3
from simple_syrup.nodes_v3.simple_load_flux2 import SimpleLoadFlux2V3
from simple_syrup.runtime.model_downloads import ComfyProgressReporter


class FakeFolderPaths(ModuleType):
    """Provide deterministic model choices for loader schemas."""

    def __init__(self) -> None:
        """Create model lists covering every loader input family."""

        super().__init__("folder_paths")
        self.files = {
            "diffusion_models": ["flux.safetensors", "flux2.safetensors"],
            "text_encoders": ["manual_encoder.safetensors"],
            "vae": ["manual_vae.safetensors"],
            "vae_approx": [],
        }

    def get_filename_list(self, folder_name: str) -> list[str]:
        """Return the configured filenames for one ComfyUI model folder."""

        return self.files[folder_name]


@pytest.mark.parametrize(
    ("node", "node_id", "input_ids"),
    (
        (
            SimpleLoadFluxV3,
            "SimpleSyrup.SimpleLoadFlux",
            [
                "diffusion_model",
                "diffusion_weight_dtype",
                "clip_l",
                "t5_xxl",
                "text_encoder_device",
                "vae",
            ],
        ),
        (
            SimpleLoadFlux2V3,
            "SimpleSyrup.SimpleLoadFlux2",
            [
                "diffusion_model",
                "diffusion_weight_dtype",
                "text_encoder",
                "text_encoder_device",
                "vae",
            ],
        ),
    ),
)
def test_flux_loader_schema_contracts_are_separate(
    monkeypatch: pytest.MonkeyPatch,
    node: type[SimpleLoadFluxV3] | type[SimpleLoadFlux2V3],
    node_id: str,
    input_ids: list[str],
) -> None:
    """Each generation exposes only the encoder choices relevant to its setup."""

    monkeypatch.setitem(sys.modules, "folder_paths", FakeFolderPaths())

    schema = node.define_schema()

    assert schema.node_id == node_id
    assert [input_item.id for input_item in schema.inputs] == input_ids
    assert [output.io_type for output in schema.outputs] == ["MODEL", "CLIP", "VAE"]
    auto_inputs = [
        input_item
        for input_item in schema.inputs
        if input_item.id in {"clip_l", "t5_xxl", "text_encoder", "vae"}
    ]
    assert all(input_item.options[0] == "auto" for input_item in auto_inputs)
    assert all(input_item.default == "auto" for input_item in auto_inputs)
    assert schema.inputs[0].advanced is None
    assert all(input_item.advanced is True for input_item in schema.inputs[1:])
    assert "download" in schema.description.lower()


def test_flux1_execute_supplies_comfy_progress_reporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FLUX.1 execution exposes automatic download progress to ComfyUI."""

    class FakeService:
        """Capture FLUX.1 execution arguments."""

        def __init__(self) -> None:
            """Create an empty keyword capture."""

            self.kwargs: dict[str, object] = {}

        def load_models(self, **kwargs: object) -> tuple[str, str, str]:
            """Capture arguments and return fixed node outputs."""

            self.kwargs = kwargs
            return "model", "clip", "vae"

    service = FakeService()
    monkeypatch.setattr(SimpleLoadFluxV3, "_service", service)

    result = SimpleLoadFluxV3.execute(
        "flux.safetensors",
        "default",
        "auto",
        "auto",
        "default",
        "auto",
    )

    assert result == ("model", "clip", "vae")
    assert isinstance(service.kwargs["progress"], ComfyProgressReporter)


def test_flux2_execute_supplies_comfy_progress_reporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FLUX.2 execution exposes automatic download progress to ComfyUI."""

    class FakeService:
        """Capture FLUX.2 execution arguments."""

        def __init__(self) -> None:
            """Create an empty keyword capture."""

            self.kwargs: dict[str, object] = {}

        def load_models(self, **kwargs: object) -> tuple[str, str, str]:
            """Capture arguments and return fixed node outputs."""

            self.kwargs = kwargs
            return "model", "clip", "vae"

    service = FakeService()
    monkeypatch.setattr(SimpleLoadFlux2V3, "_service", service)

    result = SimpleLoadFlux2V3.execute(
        "flux2.safetensors",
        "default",
        "auto",
        "default",
        "auto",
    )

    assert result == ("model", "clip", "vae")
    assert isinstance(service.kwargs["progress"], ComfyProgressReporter)
