# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coverage tests for ComfyUI node tooltip metadata."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any, Protocol, cast

import pytest

from simple_syrup.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from simple_syrup.nodes_v3.batch_region_conditioning import BatchRegionConditioningV3
from simple_syrup.nodes_v3.batch_segs import BatchSEGSV3
from simple_syrup.nodes_v3.encode_prompt_batch_with_prompt_control import (
    EncodePromptBatchWithPromptControl,
)
from simple_syrup.nodes_v3.scale_factor import ScaleFactorV3
from simple_syrup.nodes_v3.schedule_and_encode_prompts_with_prompt_control import (
    ScheduleAndEncodePromptsWithPromptControl,
)
from simple_syrup.nodes_v3.simple_load_checkpoint import SimpleLoadCheckpointV3
from simple_syrup.nodes_v3.tag_segs_with_wd14 import TagSEGSWithWD14V3
from simple_syrup.nodes_v3.tile_and_tag_segs import TileAndTagSEGSV3
from simple_syrup.nodes_v3.vae_decode_options import VAEDecodeOptionsV3
from simple_syrup.nodes_v3.vae_encode_options import VAEEncodeOptionsV3
from simple_syrup.nodes_v3.wd14_tagger_loader import WD14TaggerLoaderV3


class _LegacyNode(Protocol):
    """Protocol for legacy ComfyUI node declarations."""

    DESCRIPTION: str

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Return ComfyUI legacy input metadata."""


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
            "vae": ["manual_vae.safetensors"],
            "vae_approx": [],
        }

    def get_filename_list(self, folder_name: str) -> list[str]:
        """Return deterministic filenames for a model folder."""

        return self._files[folder_name]


def test_legacy_nodes_provide_tooltip_metadata() -> None:
    """All exported legacy nodes expose descriptions and field-level help."""

    for node_id, raw_node_class in NODE_CLASS_MAPPINGS.items():
        node_class = cast(_LegacyNode, raw_node_class)
        assert node_id in NODE_DISPLAY_NAME_MAPPINGS
        description = getattr(node_class, "DESCRIPTION", None)
        assert isinstance(description, str) and description.strip(), (
            f"{node_id} is missing DESCRIPTION."
        )

        input_types = node_class.INPUT_TYPES()
        assert isinstance(input_types, Mapping), f"{node_id} INPUT_TYPES is invalid."
        for section_name in ("required", "optional", "hidden"):
            section = input_types.get(section_name, {})
            assert isinstance(section, Mapping), (
                f"{node_id} {section_name} inputs must be a mapping."
            )
            for field_name, declaration in section.items():
                if section_name == "hidden" and _legacy_hidden_sentinel(declaration):
                    continue
                assert _tooltip_from_legacy_declaration(declaration), (
                    f"{node_id} {section_name}.{field_name} is missing tooltip "
                    "metadata."
                )


def test_legacy_named_outputs_provide_tooltips() -> None:
    """All named legacy outputs provide matching output tooltip metadata."""

    for node_id, node_class in NODE_CLASS_MAPPINGS.items():
        return_names = getattr(node_class, "RETURN_NAMES", None)
        if return_names is None:
            continue

        output_tooltips = getattr(node_class, "OUTPUT_TOOLTIPS", None)
        assert isinstance(output_tooltips, tuple), (
            f"{node_id} is missing OUTPUT_TOOLTIPS."
        )
        assert len(output_tooltips) == len(return_names), (
            f"{node_id} OUTPUT_TOOLTIPS must match RETURN_NAMES length."
        )
        for output_name, tooltip in zip(return_names, output_tooltips, strict=True):
            assert isinstance(tooltip, str) and tooltip.strip(), (
                f"{node_id} output.{output_name} is missing tooltip metadata."
            )


@pytest.mark.parametrize(
    "schema_class",
    [
        SimpleLoadCheckpointV3,
        ScaleFactorV3,
        BatchSEGSV3,
        BatchRegionConditioningV3,
        TagSEGSWithWD14V3,
        TileAndTagSEGSV3,
        VAEDecodeOptionsV3,
        VAEEncodeOptionsV3,
        WD14TaggerLoaderV3,
        EncodePromptBatchWithPromptControl,
        ScheduleAndEncodePromptsWithPromptControl,
    ],
)
def test_v3_nodes_provide_tooltip_metadata(
    monkeypatch: pytest.MonkeyPatch,
    schema_class: type[_V3Node],
) -> None:
    """All supported v3 schemas expose descriptions and field-level help."""

    monkeypatch.setitem(sys.modules, "folder_paths", _FakeFolderPaths())

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


def test_high_impact_tooltips_explain_direction_and_units() -> None:
    """Important numeric controls explain units or practical direction."""

    detail_node = cast(
        _LegacyNode,
        NODE_CLASS_MAPPINGS["SimpleSyrup.DetailSEGSByScaleFactor"],
    )
    detail_inputs = detail_node.INPUT_TYPES()["required"]
    denoise = _tooltip_from_legacy_declaration(detail_inputs["denoise"])
    feather = _tooltip_from_legacy_declaration(detail_inputs["feather"])
    assert "lower" in denoise.lower() and "higher" in denoise.lower()
    assert "pixels" in feather.lower()

    tiled_node = cast(
        _LegacyNode,
        NODE_CLASS_MAPPINGS["SimpleSyrup.KSamplerTiledDiffusion"],
    )
    tiled_inputs = tiled_node.INPUT_TYPES()["required"]
    overlap = _tooltip_from_legacy_declaration(tiled_inputs["latent_tile_overlap"])
    batch_size = _tooltip_from_legacy_declaration(
        tiled_inputs["latent_tile_batch_size"]
    )
    assert "overlap" in overlap.lower() and "seams" in overlap.lower()
    assert "memory" in batch_size.lower()


def _tooltip_from_legacy_declaration(declaration: object) -> str:
    """Return a legacy ComfyUI field tooltip or an empty string."""

    if not isinstance(declaration, tuple) or len(declaration) < 2:
        return ""
    options = declaration[1]
    if not isinstance(options, dict):
        return ""
    tooltip: Any = options.get("tooltip")
    if not isinstance(tooltip, str):
        return ""
    return tooltip.strip()


def _legacy_hidden_sentinel(declaration: object) -> bool:
    """Return whether a declaration is a Comfy legacy hidden input sentinel."""

    return isinstance(declaration, str) and declaration in {
        "PROMPT",
        "DYNPROMPT",
        "EXTRA_PNGINFO",
        "UNIQUE_ID",
        "AUTH_TOKEN_COMFY_ORG",
        "API_KEY_COMFY_ORG",
    }
