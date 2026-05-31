# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coverage tests for Comfy v3 node tooltip metadata."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Protocol

import pytest


class _V3Node(Protocol):
    """Protocol for Comfy v3 node schema declarations."""

    @classmethod
    def define_schema(cls) -> Any:
        """Return a Comfy v3 schema object."""


class _FakeFolderPaths(ModuleType):
    """Small folder_paths fake for deterministic v3 loader schemas."""

    def __init__(self) -> None:
        """Create deterministic ComfyUI filename lists."""

        super().__init__("folder_paths")
        self.models_dir = "E:\\ComfyUI\\models"
        self.user_directory = "E:\\ComfyUI\\user"
        self._files = {
            "checkpoints": ["model.safetensors"],
            "diffusion_models": ["diffusion.safetensors"],
            "text_encoders": ["text_encoder.safetensors"],
            "unet": ["model.safetensors"],
            "vae": ["manual_vae.safetensors"],
            "vae_approx": [],
        }

    def get_filename_list(self, folder_name: str) -> list[str]:
        """Return deterministic filenames for a model folder."""

        return self._files.get(folder_name, [])


def test_v3_nodes_provide_tooltip_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """All supported v3 schemas expose descriptions and field-level help."""

    monkeypatch.setitem(sys.modules, "folder_paths", _FakeFolderPaths())
    nodes_v3 = __import__("simple_syrup.nodes_v3", fromlist=["get_nodes"])
    monkeypatch.setattr(nodes_v3, "prompt_control_is_available", lambda: True)

    for schema_class in nodes_v3.get_nodes():
        schema = schema_class.define_schema()
        assert isinstance(schema.description, str) and schema.description.strip(), (
            f"{schema.node_id} v3 schema is missing description."
        )
        for input_item in schema.inputs:
            tooltip = getattr(input_item, "tooltip", None)
            assert isinstance(tooltip, str) and tooltip.strip(), (
                f"{schema.node_id} input.{input_item.id} is missing tooltip metadata."
            )
        for output in schema.outputs:
            tooltip = getattr(output, "tooltip", None)
            assert isinstance(tooltip, str) and tooltip.strip(), (
                f"{schema.node_id} output.{output.id} is missing tooltip metadata."
            )


def test_high_impact_tooltips_explain_direction_and_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Important numeric controls explain units or practical direction."""

    monkeypatch.setitem(sys.modules, "folder_paths", _FakeFolderPaths())
    nodes_v3 = __import__("simple_syrup.nodes_v3", fromlist=["get_nodes"])
    monkeypatch.setattr(nodes_v3, "prompt_control_is_available", lambda: False)
    schemas = {
        node.define_schema().node_id: node.define_schema()
        for node in nodes_v3.get_nodes()
    }

    detail_inputs = _inputs_by_id(schemas["SimpleSyrup.DetailSEGSByScaleFactor"])
    assert "lower" in detail_inputs["denoise"].tooltip.lower()
    assert "higher" in detail_inputs["denoise"].tooltip.lower()
    assert "pixels" in detail_inputs["feather"].tooltip.lower()

    tiled_inputs = _inputs_by_id(schemas["SimpleSyrup.KSamplerTiledDiffusion"])
    assert "overlap" in tiled_inputs["latent_tile_overlap"].tooltip.lower()
    assert "seams" in tiled_inputs["latent_tile_overlap"].tooltip.lower()
    assert "memory" in tiled_inputs["latent_tile_batch_size"].tooltip.lower()


def _inputs_by_id(schema: Any) -> dict[str, Any]:
    """Return schema inputs keyed by id."""

    return {input_item.id: input_item for input_item in schema.inputs}
