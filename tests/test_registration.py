# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for SimpleSyrup ComfyUI node registration."""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def test_package_exports_node_mappings() -> None:
    """Root package import exposes ComfyUI mapping dictionaries."""

    package = importlib.import_module("SimpleSyrup")

    assert hasattr(package, "NODE_CLASS_MAPPINGS")
    assert hasattr(package, "NODE_DISPLAY_NAME_MAPPINGS")
    assert hasattr(package, "comfy_entrypoint")
    assert package.WEB_DIRECTORY == "./web/dist"


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
        "assert 'SimpleSyrup.PromptSEGSWithSAM' in package.NODE_CLASS_MAPPINGS; "
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


def test_resize_node_is_registered() -> None:
    """Resize node id maps to the expected node class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.ResizeImageToTarget"]

    assert registered.__name__ == "ResizeImageToTarget"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.ResizeImageToTarget"]
        == "Resize Image to Target"
    )


def test_ksampler_extras_node_is_registered() -> None:
    """KSampler Extras node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.KSamplerExtras"]

    assert registered.__name__ == "KSamplerExtras"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.KSamplerExtras"]
        == "KSampler (Extras)"
    )


def test_ksampler_tiled_diffusion_node_is_registered() -> None:
    """KSampler tiled diffusion node maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.KSamplerTiledDiffusion"]

    assert registered.__name__ == "KSamplerTiledDiffusion"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.KSamplerTiledDiffusion"]
        == "KSampler (Tiled Diffusion)"
    )
    assert (
        "KSamplerTiledDiffusion"
        in importlib.import_module("SimpleSyrup.simple_syrup.nodes").__all__
    )
    assert "SimpleSyrup.KSamplerMixtureOfDiffusers" not in package.NODE_CLASS_MAPPINGS
    assert "SimpleSyrup.KSamplerMultiDiffusion" not in package.NODE_CLASS_MAPPINGS
    assert (
        "SimpleSyrup.KSamplerMixtureOfDiffusers"
        not in package.NODE_DISPLAY_NAME_MAPPINGS
    )
    assert (
        "SimpleSyrup.KSamplerMultiDiffusion" not in package.NODE_DISPLAY_NAME_MAPPINGS
    )


def test_latent_diagnostics_node_is_registered() -> None:
    """Latent Diagnostics node maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.LatentDiagnostics"]

    assert registered.__name__ == "LatentDiagnostics"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.LatentDiagnostics"]
        == "Latent Diagnostics"
    )


def test_prompt_encode_style_node_is_registered() -> None:
    """Prompt Encode Style node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.PromptEncodeStyle"]

    assert registered.__name__ == "PromptEncodeStyle"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.PromptEncodeStyle"]
        == "Prompt Encode Style"
    )


def test_prompt_encode_style_and_normalization_node_is_registered() -> None:
    """Prompt Encode Style & Normalization node maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS[
        "SimpleSyrup.PromptEncodeStyleAndNormalization"
    ]

    assert registered.__name__ == "PromptEncodeStyleAndNormalization"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS[
            "SimpleSyrup.PromptEncodeStyleAndNormalization"
        ]
        == "Prompt Encode Style & Normalization"
    )


def test_prompt_control_encode_style_clean_break_id_is_removed() -> None:
    """Old Prompt Control Encode Style node id is not registered."""

    package = importlib.import_module("SimpleSyrup")

    assert "SimpleSyrup.PromptControlEncodeStyle" not in package.NODE_CLASS_MAPPINGS
    assert (
        "SimpleSyrup.PromptControlEncodeStyle" not in package.NODE_DISPLAY_NAME_MAPPINGS
    )


def test_prompt_segs_with_sam_node_is_registered() -> None:
    """Prompt SEGS w/ SAM node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.PromptSEGSWithSAM"]

    assert registered.__name__ == "PromptSEGSWithSAM"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.PromptSEGSWithSAM"]
        == "Prompt SEGS w/ SAM"
    )
    assert "SimpleSyrup.PromptSAMMask" not in package.NODE_CLASS_MAPPINGS
    assert "SimpleSyrup.PromptSAMMask" not in package.NODE_DISPLAY_NAME_MAPPINGS


def test_sam_model_loader_node_is_registered() -> None:
    """SAM loader node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.SAMModelLoader"]

    assert registered.__name__ == "SAMModelLoader"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.SAMModelLoader"]
        == "SAM Model Loader"
    )


def test_scale_factor_node_is_registered() -> None:
    """Scale Factor node maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.ScaleFactor"]

    assert registered.__name__ == "ScaleFactor"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.ScaleFactor"] == (
        "Scale Factor"
    )
    assert (
        "ScaleFactor"
        in importlib.import_module("SimpleSyrup.simple_syrup.nodes").__all__
    )


def test_seed_node_is_registered() -> None:
    """Seed node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.Seed"]

    assert registered.__name__ == "Seed"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.Seed"] == "Seed"


def test_grounding_dino_model_loader_node_is_registered() -> None:
    """GroundingDINO loader node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.GroundingDINOModelLoader"]

    assert registered.__name__ == "GroundingDINOModelLoader"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.GroundingDINOModelLoader"]
        == "GroundingDINO Model Loader"
    )


def test_vitmatte_model_loader_node_is_registered() -> None:
    """ViTMatte loader node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.ViTMatteModelLoader"]

    assert registered.__name__ == "ViTMatteModelLoader"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.ViTMatteModelLoader"]
        == "ViTMatte Model Loader"
    )


def test_wd14_tagger_loader_node_is_registered() -> None:
    """WD14 tagger loader node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.WD14TaggerLoader"]

    assert registered.__name__ == "WD14TaggerLoader"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.WD14TaggerLoader"]
        == "Load WD14 Tagger"
    )


def test_load_ultralytics_model_node_is_registered() -> None:
    """Load Ultralytics Model node maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.LoadUltralyticsModel"]

    assert registered.__name__ == "LoadUltralyticsModel"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.LoadUltralyticsModel"]
        == "Load Ultralytics Model"
    )


def test_detect_segs_with_ultralytics_node_is_registered() -> None:
    """Detect SEGS w/ Ultralytics node maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.DetectSEGSWithUltralytics"]

    assert registered.__name__ == "DetectSEGSWithUltralytics"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.DetectSEGSWithUltralytics"]
        == "Detect SEGS w/ Ultralytics"
    )


def test_detail_segs_by_scale_factor_node_is_registered() -> None:
    """Detail SEGS by Scale Factor node maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.DetailSEGSByScaleFactor"]

    assert registered.__name__ == "DetailSEGSByScaleFactor"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.DetailSEGSByScaleFactor"]
        == "Detail SEGS by Scale Factor"
    )


def test_tiled_detail_segs_by_scale_factor_node_is_registered() -> None:
    """Tiled Detail SEGS by Scale Factor node maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS[
        "SimpleSyrup.DetailSEGSByScaleFactorTiledDiffusion"
    ]

    assert registered.__name__ == "DetailSEGSByScaleFactorTiledDiffusion"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS[
            "SimpleSyrup.DetailSEGSByScaleFactorTiledDiffusion"
        ]
        == "Detail SEGS by Scale Factor w/ Tiled Diffusion"
    )
    assert (
        "DetailSEGSByScaleFactorTiledDiffusion"
        in importlib.import_module("SimpleSyrup.simple_syrup.nodes").__all__
    )


def test_detail_segs_as_regions_node_is_registered() -> None:
    """Detail SEGS as Regions node maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.DetailSEGSAsRegions"]

    assert registered.__name__ == "DetailSEGSAsRegions"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.DetailSEGSAsRegions"]
        == "Detail SEGS as Regions"
    )


def test_tile_and_tag_segs_node_is_registered() -> None:
    """Tile & Tag SEGS node maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.TileAndTagSEGS"]

    assert registered.__name__ == "TileAndTagSEGS"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.TileAndTagSEGS"]
        == "Tile & Tag SEGS"
    )


def test_conditioning_batch_nodes_are_registered() -> None:
    """Conditioning batch nodes map to their classes and display names."""

    package = importlib.import_module("SimpleSyrup")

    assert (
        package.NODE_CLASS_MAPPINGS["SimpleSyrup.ConditioningBatchStart"].__name__
        == "ConditioningBatchStart"
    )
    assert (
        package.NODE_CLASS_MAPPINGS["SimpleSyrup.ConditioningBatchAppend"].__name__
        == "ConditioningBatchAppend"
    )
    assert (
        package.NODE_CLASS_MAPPINGS["SimpleSyrup.EncodePromptBatch"].__name__
        == "EncodePromptBatch"
    )
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.ConditioningBatchStart"]
        == "Conditioning Batch Start"
    )
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.ConditioningBatchAppend"]
        == "Conditioning Batch Append"
    )
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.EncodePromptBatch"]
        == "Encode Prompt Batch"
    )


def test_layerstyle_adapter_node_is_registered() -> None:
    """LayerStyle adapter node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.LayerStyleSAMModelsAdapter"]

    assert registered.__name__ == "LayerStyleSAMModelsAdapter"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.LayerStyleSAMModelsAdapter"]
        == "LayerStyle SAM Models Adapter"
    )


def test_grounded_sam_model_info_node_is_registered() -> None:
    """Grounded SAM model info node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.GroundedSAMModelInfo"]

    assert registered.__name__ == "GroundedSAMModelInfo"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.GroundedSAMModelInfo"]
        == "Grounded SAM Model Info"
    )


def test_simple_load_anima_node_is_registered() -> None:
    """Simple Load Anima node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.SimpleLoadAnima"]

    assert registered.__name__ == "SimpleLoadAnima"
    assert registered.RETURN_TYPES == ("MODEL", "CLIP", "VAE")
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.SimpleLoadAnima"]
        == "Simple Load Anima"
    )


def test_simple_load_checkpoint_node_is_registered() -> None:
    """Simple Load Checkpoint node id maps to its class and display name."""

    package = importlib.import_module("SimpleSyrup")
    registered = package.NODE_CLASS_MAPPINGS["SimpleSyrup.SimpleLoadCheckpoint"]

    assert registered.__name__ == "SimpleLoadCheckpoint"
    assert registered.RETURN_TYPES == ("MODEL", "CLIP", "VAE")
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.SimpleLoadCheckpoint"]
        == "Simple Load Checkpoint"
    )


def test_provenance_latent_nodes_are_registered() -> None:
    """Provenance-aware latent nodes map to their classes and display names."""

    package = importlib.import_module("SimpleSyrup")
    nodes_package = importlib.import_module("SimpleSyrup.simple_syrup.nodes")

    simple_vae = package.NODE_CLASS_MAPPINGS["SimpleSyrup.SimpleVAEEncode"]
    upscale = package.NODE_CLASS_MAPPINGS["SimpleSyrup.UpscaleLatentFromImage"]

    assert simple_vae.__name__ == "SimpleVAEEncode"
    assert upscale.__name__ == "UpscaleLatentFromImage"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.SimpleVAEEncode"]
        == "Simple VAE Encode"
    )
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.UpscaleLatentFromImage"]
        == "Upscale Latent From Image"
    )
    assert "SimpleVAEEncode" in nodes_package.__all__
    assert "UpscaleLatentFromImage" in nodes_package.__all__


def test_vae_options_nodes_are_registered() -> None:
    """VAE options nodes map to their classes and display names."""

    package = importlib.import_module("SimpleSyrup")
    nodes_package = importlib.import_module("SimpleSyrup.simple_syrup.nodes")

    encode = package.NODE_CLASS_MAPPINGS["SimpleSyrup.VAEEncodeOptions"]
    decode = package.NODE_CLASS_MAPPINGS["SimpleSyrup.VAEDecodeOptions"]

    assert encode.__name__ == "VAEEncodeOptions"
    assert decode.__name__ == "VAEDecodeOptions"
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.VAEEncodeOptions"]
        == "VAE Encode (Options)"
    )
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["SimpleSyrup.VAEDecodeOptions"]
        == "VAE Decode (Options)"
    )
    assert "VAEEncodeOptions" in nodes_package.__all__
    assert "VAEDecodeOptions" in nodes_package.__all__


def test_registration_import_does_not_require_torchlanc() -> None:
    """Importing registration does not eagerly import TorchLanc."""

    sys.modules.pop("torchlanc", None)
    importlib.import_module("SimpleSyrup")

    imported_module: ModuleType | None = sys.modules.get("torchlanc")
    assert imported_module is None


def test_v3_entrypoint_registers_tile_and_prompt_control_batch_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comfy v3 entrypoint exposes native v3 nodes without Prompt Control imports."""

    sys.modules.pop("prompt_control.nodes_lazy", None)
    package = importlib.import_module("SimpleSyrup")
    nodes_v3 = importlib.import_module("SimpleSyrup.simple_syrup.nodes_v3")
    monkeypatch.setattr(nodes_v3, "prompt_control_is_available", lambda: True)

    extension = asyncio.run(package.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())

    assert [node.__name__ for node in nodes] == [
        "WD14TaggerLoaderV3",
        "TileAndTagSEGSV3",
        "SimpleLoadCheckpointV3",
        "ScaleFactorV3",
        "VAEDecodeOptionsV3",
        "VAEEncodeOptionsV3",
        "EncodePromptBatchWithPromptControl",
    ]
    assert "prompt_control.nodes_lazy" not in sys.modules


def test_v3_entrypoint_keeps_tile_node_when_prompt_control_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comfy v3 entrypoint omits only Prompt Control nodes when unavailable."""

    sys.modules.pop("prompt_control.nodes_lazy", None)
    package = importlib.import_module("SimpleSyrup")
    nodes_v3 = importlib.import_module("SimpleSyrup.simple_syrup.nodes_v3")
    monkeypatch.setattr(nodes_v3, "prompt_control_is_available", lambda: False)

    extension = asyncio.run(package.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())

    assert [node.__name__ for node in nodes] == [
        "WD14TaggerLoaderV3",
        "TileAndTagSEGSV3",
        "SimpleLoadCheckpointV3",
        "ScaleFactorV3",
        "VAEDecodeOptionsV3",
        "VAEEncodeOptionsV3",
    ]
    assert "prompt_control.nodes_lazy" not in sys.modules
