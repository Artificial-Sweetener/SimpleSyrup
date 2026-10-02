# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 wrappers for nodes whose behavior still lives in legacy modules."""

from __future__ import annotations

from ..nodes.conditioning_batch_pack import (
    ConditioningBatchAppend,
    ConditioningBatchStart,
)
from ..nodes.detail_segs_as_regions import DetailSEGSAsRegions
from ..nodes.detail_segs_by_scale_factor import DetailSEGSByScaleFactor
from ..nodes.detail_segs_by_scale_factor_tiled_diffusion import (
    DetailSEGSByScaleFactorTiledDiffusion,
)
from ..nodes.detect_segs_with_ultralytics import DetectSEGSWithUltralytics
from ..nodes.encode_prompt_batch import EncodePromptBatch
from ..nodes.grounded_sam_model_info import GroundedSAMModelInfo
from ..nodes.grounding_dino_model_loader import GroundingDINOModelLoader
from ..nodes.image_resize_to_target import ResizeImageToTarget
from ..nodes.ksampler_extras import KSamplerExtras
from ..nodes.latent_diagnostics import LatentDiagnostics
from ..nodes.layerstyle_sam_models_adapter import LayerStyleSAMModelsAdapter
from ..nodes.load_ultralytics_model import LoadUltralyticsModel
from ..nodes.prompt_encode_style import PromptEncodeStyle
from ..nodes.prompt_encode_style_and_normalization import (
    PromptEncodeStyleAndNormalization,
)
from ..nodes.prompt_segs_with_sam import PromptSEGSWithSAM
from ..nodes.provenance_latent import SimpleVAEEncode, UpscaleLatentFromImage
from ..nodes.sam_model_loader import SAMModelLoader
from ..nodes.seed import Seed
from ..nodes.segs_from_sam_output import SEGSFromSAMOutput
from ..nodes.simple_load_anima import SimpleLoadAnima
from ..nodes.simple_preview_segs import SimplePreviewSEGS
from ..nodes.vitmatte_model_loader import ViTMatteModelLoader
from .legacy_inversion_node_adapter import LegacyInversionNodeV3Adapter
from .legacy_node_adapter import LegacyNodeV3Adapter
from .legacy_workflow_input_order import (
    DETAIL_SEGS_AS_REGIONS_INPUT_ORDER,
    DETAIL_SEGS_BY_SCALE_FACTOR_INPUT_ORDER,
    DETAIL_SEGS_BY_SCALE_FACTOR_TILED_INPUT_ORDER,
    KSAMPLER_EXTRAS_INPUT_ORDER,
)


class ConditioningBatchStartV3(LegacyNodeV3Adapter):
    """Expose Conditioning Batch Start through Comfy v3 only."""

    LEGACY_NODE_CLASS = ConditioningBatchStart
    NODE_ID = "SimpleSyrup.ConditioningBatchStart"
    DISPLAY_NAME = "Conditioning Batch Start"


class ConditioningBatchAppendV3(LegacyNodeV3Adapter):
    """Expose Conditioning Batch Append through Comfy v3 only."""

    LEGACY_NODE_CLASS = ConditioningBatchAppend
    NODE_ID = "SimpleSyrup.ConditioningBatchAppend"
    DISPLAY_NAME = "Conditioning Batch Append"


class GroundedSAMModelInfoV3(LegacyNodeV3Adapter):
    """Expose Grounded SAM Model Info through Comfy v3 only."""

    LEGACY_NODE_CLASS = GroundedSAMModelInfo
    NODE_ID = "SimpleSyrup.GroundedSAMModelInfo"
    DISPLAY_NAME = "Grounded SAM Model Info"


class GroundingDINOModelLoaderV3(LegacyNodeV3Adapter):
    """Expose GroundingDINO Model Loader through Comfy v3 only."""

    LEGACY_NODE_CLASS = GroundingDINOModelLoader
    NODE_ID = "SimpleSyrup.GroundingDINOModelLoader"
    DISPLAY_NAME = "GroundingDINO Model Loader"


class KSamplerExtrasV3(LegacyNodeV3Adapter):
    """Expose KSampler Extras through Comfy v3 only."""

    LEGACY_NODE_CLASS = KSamplerExtras
    NODE_ID = "SimpleSyrup.KSamplerExtras"
    DISPLAY_NAME = "KSampler (Extras)"
    WORKFLOW_INPUT_ORDER = KSAMPLER_EXTRAS_INPUT_ORDER


class LayerStyleSAMModelsAdapterV3(LegacyNodeV3Adapter):
    """Expose LayerStyle SAM Models Adapter through Comfy v3 only."""

    LEGACY_NODE_CLASS = LayerStyleSAMModelsAdapter
    NODE_ID = "SimpleSyrup.LayerStyleSAMModelsAdapter"
    DISPLAY_NAME = "LayerStyle SAM Models Adapter"


class LatentDiagnosticsV3(LegacyNodeV3Adapter):
    """Expose Latent Diagnostics through Comfy v3 only."""

    LEGACY_NODE_CLASS = LatentDiagnostics
    NODE_ID = "SimpleSyrup.LatentDiagnostics"
    DISPLAY_NAME = "Latent Diagnostics"


class PromptEncodeStyleV3(LegacyNodeV3Adapter):
    """Expose Prompt Encode Style through Comfy v3 only."""

    LEGACY_NODE_CLASS = PromptEncodeStyle
    NODE_ID = "SimpleSyrup.PromptEncodeStyle"
    DISPLAY_NAME = "Prompt Encode Style"


class PromptEncodeStyleAndNormalizationV3(LegacyNodeV3Adapter):
    """Expose Prompt Encode Style and Normalization through Comfy v3 only."""

    LEGACY_NODE_CLASS = PromptEncodeStyleAndNormalization
    NODE_ID = "SimpleSyrup.PromptEncodeStyleAndNormalization"
    DISPLAY_NAME = "Prompt Encode Style & Normalization"


class PromptSEGSWithSAMV3(LegacyNodeV3Adapter):
    """Expose Prompt SEGS w/ SAM through Comfy v3 only."""

    LEGACY_NODE_CLASS = PromptSEGSWithSAM
    NODE_ID = "SimpleSyrup.PromptSEGSWithSAM"
    DISPLAY_NAME = "Prompt SEGS w/ SAM"


class SimpleVAEEncodeV3(LegacyNodeV3Adapter):
    """Expose Simple VAE Encode through Comfy v3 only."""

    LEGACY_NODE_CLASS = SimpleVAEEncode
    NODE_ID = "SimpleSyrup.SimpleVAEEncode"
    DISPLAY_NAME = "Simple VAE Encode"
    ENABLE_EXPAND = True


class UpscaleLatentFromImageV3(LegacyNodeV3Adapter):
    """Expose Upscale Latent From Image through Comfy v3 only."""

    LEGACY_NODE_CLASS = UpscaleLatentFromImage
    NODE_ID = "SimpleSyrup.UpscaleLatentFromImage"
    DISPLAY_NAME = "Upscale Latent From Image"
    ENABLE_EXPAND = True


class ResizeImageToTargetV3(LegacyNodeV3Adapter):
    """Expose Resize Image to Target through Comfy v3 only."""

    LEGACY_NODE_CLASS = ResizeImageToTarget
    NODE_ID = "SimpleSyrup.ResizeImageToTarget"
    DISPLAY_NAME = "Resize Image to Target"


class DetailSEGSAsRegionsV3(LegacyInversionNodeV3Adapter):
    """Expose Detail SEGS as Regions through Comfy v3 only."""

    LEGACY_NODE_CLASS = DetailSEGSAsRegions
    NODE_ID = "SimpleSyrup.DetailSEGSAsRegions"
    DISPLAY_NAME = "Detail SEGS as Regions"
    WORKFLOW_INPUT_ORDER = DETAIL_SEGS_AS_REGIONS_INPUT_ORDER


class DetailSEGSByScaleFactorV3(LegacyNodeV3Adapter):
    """Expose Detail SEGS by Scale Factor through Comfy v3 only."""

    LEGACY_NODE_CLASS = DetailSEGSByScaleFactor
    NODE_ID = "SimpleSyrup.DetailSEGSByScaleFactor"
    DISPLAY_NAME = "Detail SEGS by Scale Factor"
    WORKFLOW_INPUT_ORDER = DETAIL_SEGS_BY_SCALE_FACTOR_INPUT_ORDER


class DetailSEGSByScaleFactorTiledDiffusionV3(LegacyInversionNodeV3Adapter):
    """Expose Detail SEGS by Scale Factor with Tiled Diffusion through Comfy v3."""

    LEGACY_NODE_CLASS = DetailSEGSByScaleFactorTiledDiffusion
    NODE_ID = "SimpleSyrup.DetailSEGSByScaleFactorTiledDiffusion"
    DISPLAY_NAME = "Detail SEGS by Scale Factor w/ Tiled Diffusion"
    WORKFLOW_INPUT_ORDER = DETAIL_SEGS_BY_SCALE_FACTOR_TILED_INPUT_ORDER


class SAMModelLoaderV3(LegacyNodeV3Adapter):
    """Expose SAM Model Loader through Comfy v3 only."""

    LEGACY_NODE_CLASS = SAMModelLoader
    NODE_ID = "SimpleSyrup.SAMModelLoader"
    DISPLAY_NAME = "SAM Model Loader"


class SEGSFromSAMOutputV3(LegacyNodeV3Adapter):
    """Expose automatic SAM-to-SEGS generation through Comfy v3 only."""

    LEGACY_NODE_CLASS = SEGSFromSAMOutput
    NODE_ID = "SimpleSyrup.SEGSFromSAMOutput"
    DISPLAY_NAME = "SEGS from SAM Output"


class SimplePreviewSEGSV3(LegacyNodeV3Adapter):
    """Expose the interactive Simple Preview SEGS node through Comfy v3 only."""

    LEGACY_NODE_CLASS = SimplePreviewSEGS
    NODE_ID = "SimpleSyrup.SimplePreviewSEGS"
    DISPLAY_NAME = "Simple Preview SEGS"


class SeedV3(LegacyNodeV3Adapter):
    """Expose Seed through Comfy v3 only."""

    LEGACY_NODE_CLASS = Seed
    NODE_ID = "SimpleSyrup.Seed"
    DISPLAY_NAME = "Seed"


class SimpleLoadAnimaV3(LegacyNodeV3Adapter):
    """Expose Simple Load Anima through Comfy v3 only."""

    LEGACY_NODE_CLASS = SimpleLoadAnima
    NODE_ID = "SimpleSyrup.SimpleLoadAnima"
    DISPLAY_NAME = "Simple Load Anima"


class LoadUltralyticsModelV3(LegacyNodeV3Adapter):
    """Expose Load Ultralytics Model through Comfy v3 only."""

    LEGACY_NODE_CLASS = LoadUltralyticsModel
    NODE_ID = "SimpleSyrup.LoadUltralyticsModel"
    DISPLAY_NAME = "Load Ultralytics Model"


class DetectSEGSWithUltralyticsV3(LegacyNodeV3Adapter):
    """Expose Detect SEGS w/ Ultralytics through Comfy v3 only."""

    LEGACY_NODE_CLASS = DetectSEGSWithUltralytics
    NODE_ID = "SimpleSyrup.DetectSEGSWithUltralytics"
    DISPLAY_NAME = "Detect SEGS w/ Ultralytics"


class EncodePromptBatchV3(LegacyNodeV3Adapter):
    """Expose Encode Prompt Batch through Comfy v3 only."""

    LEGACY_NODE_CLASS = EncodePromptBatch
    NODE_ID = "SimpleSyrup.EncodePromptBatch"
    DISPLAY_NAME = "Encode Prompt Batch"


class ViTMatteModelLoaderV3(LegacyNodeV3Adapter):
    """Expose ViTMatte Model Loader through Comfy v3 only."""

    LEGACY_NODE_CLASS = ViTMatteModelLoader
    NODE_ID = "SimpleSyrup.ViTMatteModelLoader"
    DISPLAY_NAME = "ViTMatte Model Loader"


__all__ = [
    "ConditioningBatchAppendV3",
    "ConditioningBatchStartV3",
    "DetailSEGSAsRegionsV3",
    "DetailSEGSByScaleFactorTiledDiffusionV3",
    "DetailSEGSByScaleFactorV3",
    "DetectSEGSWithUltralyticsV3",
    "EncodePromptBatchV3",
    "GroundedSAMModelInfoV3",
    "GroundingDINOModelLoaderV3",
    "KSamplerExtrasV3",
    "LatentDiagnosticsV3",
    "LayerStyleSAMModelsAdapterV3",
    "LoadUltralyticsModelV3",
    "PromptEncodeStyleAndNormalizationV3",
    "PromptEncodeStyleV3",
    "PromptSEGSWithSAMV3",
    "ResizeImageToTargetV3",
    "SAMModelLoaderV3",
    "SEGSFromSAMOutputV3",
    "SeedV3",
    "SimpleLoadAnimaV3",
    "SimplePreviewSEGSV3",
    "SimpleVAEEncodeV3",
    "UpscaleLatentFromImageV3",
    "ViTMatteModelLoaderV3",
]
