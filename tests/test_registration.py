# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for SimpleSyrup Comfy v3-only node registration."""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

import pytest

BASE_NODE_IDS = [
    "SimpleSyrup.BatchRegionConditioning",
    "SimpleSyrup.BatchSEGS",
    "SimpleSyrup.ConditioningBatchAppend",
    "SimpleSyrup.ConditioningBatchStart",
    "SimpleSyrup.DetailSEGSAsRegions",
    "SimpleSyrup.DetailSEGSByScaleFactorTiledDiffusion",
    "SimpleSyrup.DetailSEGSByScaleFactor",
    "SimpleSyrup.DetectSEGSWithUltralytics",
    "SimpleSyrup.EncodePromptBatch",
    "SimpleSyrup.ExternalLLMPrompt",
    "SimpleSyrup.GroundedSAMModelInfo",
    "SimpleSyrup.GroundingDINOModelLoader",
    "SimpleSyrup.KSamplerExtras",
    "SimpleSyrup.KSamplerTiledDiffusion",
    "SimpleSyrup.LatentDiagnostics",
    "SimpleSyrup.LayerStyleSAMModelsAdapter",
    "SimpleSyrup.LoadUltralyticsModel",
    "SimpleSyrup.PromptEncodeStyleAndNormalization",
    "SimpleSyrup.PromptEncodeStyle",
    "SimpleSyrup.PromptSEGSWithSAM",
    "SimpleSyrup.ResizeImageToTarget",
    "SimpleSyrup.SAMModelLoader",
    "SimpleSyrup.ScaleFactor",
    "SimpleSyrup.Seed",
    "SimpleSyrup.SimpleLoadAnima",
    "SimpleSyrup.SimpleLoadCheckpoint",
    "SimpleSyrup.SimpleVAEEncode",
    "SimpleSyrup.TagSEGSWithExternalLLM",
    "SimpleSyrup.TagSEGSWithWD14",
    "SimpleSyrup.TileAndTagSEGS",
    "SimpleSyrup.UpscaleLatentFromImage",
    "SimpleSyrup.VAEDecodeOptions",
    "SimpleSyrup.VAEEncodeOptions",
    "SimpleSyrup.ViTMatteModelLoader",
    "SimpleSyrup.WD14TaggerLoader",
]

PROMPT_CONTROL_NODE_IDS = [
    "SimpleSyrup.EncodePromptBatchWithPromptControl",
    "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
]


class _V3Node(Protocol):
    """Protocol for Comfy v3 node schema declarations."""

    @classmethod
    def define_schema(cls) -> Any:
        """Return a Comfy v3 schema object."""


def test_package_exports_v3_entrypoint_only() -> None:
    """Root package exposes Comfy v3 registration and no legacy mappings."""

    package = importlib.import_module("SimpleSyrup")

    assert hasattr(package, "comfy_entrypoint")
    assert package.WEB_DIRECTORY == "./web/dist"
    assert not hasattr(package, "NODE_CLASS_MAPPINGS")
    assert not hasattr(package, "NODE_DISPLAY_NAME_MAPPINGS")
    assert package.__all__ == ["WEB_DIRECTORY", "comfy_entrypoint"]


def test_nodes_package_is_not_a_legacy_registry() -> None:
    """The implementation package no longer owns ComfyUI registration."""

    nodes_package = importlib.import_module("SimpleSyrup.simple_syrup.nodes")

    assert not hasattr(nodes_package, "NODE_CLASS_MAPPINGS")
    assert not hasattr(nodes_package, "NODE_DISPLAY_NAME_MAPPINGS")


def test_package_imports_from_custom_nodes_parent_path() -> None:
    """ComfyUI-style import works without the repository root on sys.path."""

    project_root = Path(__file__).resolve().parents[1]
    custom_nodes_root = project_root.parent
    script = (
        "import importlib, pathlib, sys; "
        f"project = pathlib.Path({str(project_root)!r}).resolve(); "
        "sys.path = [p for p in sys.path "
        "if pathlib.Path(p or '.').resolve() != project]; "
        f"sys.path.insert(0, {str(custom_nodes_root)!r}); "
        "package = importlib.import_module('SimpleSyrup'); "
        "assert hasattr(package, 'comfy_entrypoint'); "
        "assert not hasattr(package, 'NODE_CLASS_MAPPINGS'); "
        "assert not hasattr(package, 'NODE_DISPLAY_NAME_MAPPINGS'); "
        "assert 'server' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=custom_nodes_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_comfy_import_exposes_stable_internal_package_alias() -> None:
    """ComfyUI-style import exposes `simple_syrup` for vendored runtime imports."""

    project_root = Path(__file__).resolve().parents[1]
    custom_nodes_root = project_root.parent
    script = (
        "import importlib, pathlib, sys; "
        f"project = pathlib.Path({str(project_root)!r}).resolve(); "
        "sys.path = [p for p in sys.path "
        "if pathlib.Path(p or '.').resolve() != project]; "
        f"sys.path.insert(0, {str(custom_nodes_root)!r}); "
        "importlib.import_module('SimpleSyrup'); "
        "runtime = importlib.import_module("
        "'simple_syrup.third_party.groundingdino_runtime.models'"
        "); "
        "assert runtime.__name__.endswith('groundingdino_runtime.models')"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=custom_nodes_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_registration_import_does_not_require_torchlanc() -> None:
    """Importing registration does not eagerly import TorchLanc."""

    sys.modules.pop("torchlanc", None)
    importlib.import_module("SimpleSyrup")

    imported_module: ModuleType | None = sys.modules.get("torchlanc")
    assert imported_module is None


def test_v3_entrypoint_exports_all_base_nodes_without_prompt_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comfy v3 entrypoint exports every maintained non-conditional node."""

    sys.modules.pop("prompt_control.nodes_lazy", None)
    package = importlib.import_module("SimpleSyrup")
    nodes_v3 = importlib.import_module("SimpleSyrup.simple_syrup.nodes_v3")
    monkeypatch.setattr(nodes_v3, "prompt_control_is_available", lambda: False)

    extension = asyncio.run(package.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())

    assert _node_ids(cast(list[type[_V3Node]], nodes)) == BASE_NODE_IDS
    assert "prompt_control.nodes_lazy" not in sys.modules


def test_v3_entrypoint_adds_only_prompt_control_nodes_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt Control availability adds conditional nodes without removing others."""

    sys.modules.pop("prompt_control.nodes_lazy", None)
    package = importlib.import_module("SimpleSyrup")
    nodes_v3 = importlib.import_module("SimpleSyrup.simple_syrup.nodes_v3")
    monkeypatch.setattr(nodes_v3, "prompt_control_is_available", lambda: True)

    extension = asyncio.run(package.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())

    assert _node_ids(cast(list[type[_V3Node]], nodes)) == [
        *BASE_NODE_IDS,
        *PROMPT_CONTROL_NODE_IDS,
    ]
    assert "prompt_control.nodes_lazy" not in sys.modules


def _node_ids(nodes: list[type[_V3Node]]) -> list[str]:
    """Return node ids from v3 schemas."""

    ids: list[str] = []
    for node in nodes:
        schema = node.define_schema()
        node_id: Any = schema.node_id
        if not isinstance(node_id, str):
            raise AssertionError(f"{node.__name__} has invalid node_id.")
        ids.append(node_id)
    return ids
