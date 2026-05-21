# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node registration for SimpleSyrup."""

from __future__ import annotations

from .conditioning_batch_pack import ConditioningBatchAppend, ConditioningBatchStart
from .detail_segs_as_regions import DetailSEGSAsRegions
from .detail_segs_by_scale_factor import DetailSEGSByScaleFactor
from .detail_segs_by_scale_factor_tiled_diffusion import (
    DetailSEGSByScaleFactorTiledDiffusion,
)
from .detect_segs_with_ultralytics import DetectSEGSWithUltralytics
from .encode_prompt_batch import EncodePromptBatch
from .grounded_sam_model_info import GroundedSAMModelInfo
from .grounding_dino_model_loader import GroundingDINOModelLoader
from .image_resize_to_target import ResizeImageToTarget
from .ksampler_extras import KSamplerExtras
from .ksampler_tiled_diffusion import KSamplerTiledDiffusion
from .latent_diagnostics import LatentDiagnostics
from .layerstyle_sam_models_adapter import LayerStyleSAMModelsAdapter
from .load_ultralytics_model import LoadUltralyticsModel
from .prompt_encode_style import PromptEncodeStyle
from .prompt_encode_style_and_normalization import PromptEncodeStyleAndNormalization
from .prompt_segs_with_sam import PromptSEGSWithSAM
from .provenance_latent import SimpleVAEEncode, UpscaleLatentFromImage
from .sam_model_loader import SAMModelLoader
from .scale_factor import ScaleFactor
from .seed import Seed
from .simple_load_anima import SimpleLoadAnima
from .simple_load_checkpoint import SimpleLoadCheckpoint
from .tile_and_tag_segs import TileAndTagSEGS
from .vitmatte_model_loader import ViTMatteModelLoader
from .wd14_tagger_loader import WD14TaggerLoader

NODE_CLASS_MAPPINGS = {
    "SimpleSyrup.ConditioningBatchAppend": ConditioningBatchAppend,
    "SimpleSyrup.ConditioningBatchStart": ConditioningBatchStart,
    "SimpleSyrup.GroundedSAMModelInfo": GroundedSAMModelInfo,
    "SimpleSyrup.GroundingDINOModelLoader": GroundingDINOModelLoader,
    "SimpleSyrup.KSamplerExtras": KSamplerExtras,
    "SimpleSyrup.KSamplerTiledDiffusion": KSamplerTiledDiffusion,
    "SimpleSyrup.LayerStyleSAMModelsAdapter": LayerStyleSAMModelsAdapter,
    "SimpleSyrup.LatentDiagnostics": LatentDiagnostics,
    "SimpleSyrup.PromptEncodeStyle": PromptEncodeStyle,
    "SimpleSyrup.PromptEncodeStyleAndNormalization": PromptEncodeStyleAndNormalization,
    "SimpleSyrup.PromptSEGSWithSAM": PromptSEGSWithSAM,
    "SimpleSyrup.SimpleVAEEncode": SimpleVAEEncode,
    "SimpleSyrup.UpscaleLatentFromImage": UpscaleLatentFromImage,
    "SimpleSyrup.ResizeImageToTarget": ResizeImageToTarget,
    "SimpleSyrup.DetailSEGSAsRegions": DetailSEGSAsRegions,
    "SimpleSyrup.DetailSEGSByScaleFactor": DetailSEGSByScaleFactor,
    "SimpleSyrup.DetailSEGSByScaleFactorTiledDiffusion": (
        DetailSEGSByScaleFactorTiledDiffusion
    ),
    "SimpleSyrup.SAMModelLoader": SAMModelLoader,
    "SimpleSyrup.ScaleFactor": ScaleFactor,
    "SimpleSyrup.Seed": Seed,
    "SimpleSyrup.SimpleLoadAnima": SimpleLoadAnima,
    "SimpleSyrup.SimpleLoadCheckpoint": SimpleLoadCheckpoint,
    "SimpleSyrup.LoadUltralyticsModel": LoadUltralyticsModel,
    "SimpleSyrup.DetectSEGSWithUltralytics": DetectSEGSWithUltralytics,
    "SimpleSyrup.EncodePromptBatch": EncodePromptBatch,
    "SimpleSyrup.TileAndTagSEGS": TileAndTagSEGS,
    "SimpleSyrup.ViTMatteModelLoader": ViTMatteModelLoader,
    "SimpleSyrup.WD14TaggerLoader": WD14TaggerLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleSyrup.ConditioningBatchAppend": "Conditioning Batch Append",
    "SimpleSyrup.ConditioningBatchStart": "Conditioning Batch Start",
    "SimpleSyrup.GroundedSAMModelInfo": "Grounded SAM Model Info",
    "SimpleSyrup.GroundingDINOModelLoader": "GroundingDINO Model Loader",
    "SimpleSyrup.KSamplerExtras": "KSampler (Extras)",
    "SimpleSyrup.KSamplerTiledDiffusion": "KSampler (Tiled Diffusion)",
    "SimpleSyrup.LayerStyleSAMModelsAdapter": "LayerStyle SAM Models Adapter",
    "SimpleSyrup.LatentDiagnostics": "Latent Diagnostics",
    "SimpleSyrup.PromptEncodeStyle": "Prompt Encode Style",
    "SimpleSyrup.PromptEncodeStyleAndNormalization": (
        "Prompt Encode Style & Normalization"
    ),
    "SimpleSyrup.PromptSEGSWithSAM": "Prompt SEGS w/ SAM",
    "SimpleSyrup.SimpleVAEEncode": "Simple VAE Encode",
    "SimpleSyrup.UpscaleLatentFromImage": "Upscale Latent From Image",
    "SimpleSyrup.ResizeImageToTarget": "Resize Image to Target",
    "SimpleSyrup.DetailSEGSAsRegions": "Detail SEGS as Regions",
    "SimpleSyrup.DetailSEGSByScaleFactor": "Detail SEGS by Scale Factor",
    "SimpleSyrup.DetailSEGSByScaleFactorTiledDiffusion": (
        "Detail SEGS by Scale Factor w/ Tiled Diffusion"
    ),
    "SimpleSyrup.SAMModelLoader": "SAM Model Loader",
    "SimpleSyrup.ScaleFactor": "Scale Factor",
    "SimpleSyrup.Seed": "Seed",
    "SimpleSyrup.SimpleLoadAnima": "Simple Load Anima",
    "SimpleSyrup.SimpleLoadCheckpoint": "Simple Load Checkpoint",
    "SimpleSyrup.LoadUltralyticsModel": "Load Ultralytics Model",
    "SimpleSyrup.DetectSEGSWithUltralytics": "Detect SEGS w/ Ultralytics",
    "SimpleSyrup.EncodePromptBatch": "Encode Prompt Batch",
    "SimpleSyrup.TileAndTagSEGS": "Tile & Tag SEGS",
    "SimpleSyrup.ViTMatteModelLoader": "ViTMatte Model Loader",
    "SimpleSyrup.WD14TaggerLoader": "Load WD14 Tagger",
}

__all__ = [
    "ConditioningBatchAppend",
    "ConditioningBatchStart",
    "GroundedSAMModelInfo",
    "GroundingDINOModelLoader",
    "KSamplerExtras",
    "KSamplerTiledDiffusion",
    "LayerStyleSAMModelsAdapter",
    "LatentDiagnostics",
    "DetectSEGSWithUltralytics",
    "DetailSEGSAsRegions",
    "DetailSEGSByScaleFactor",
    "DetailSEGSByScaleFactorTiledDiffusion",
    "EncodePromptBatch",
    "LoadUltralyticsModel",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PromptEncodeStyle",
    "PromptEncodeStyleAndNormalization",
    "PromptSEGSWithSAM",
    "ResizeImageToTarget",
    "SAMModelLoader",
    "ScaleFactor",
    "Seed",
    "SimpleLoadAnima",
    "SimpleLoadCheckpoint",
    "SimpleVAEEncode",
    "TileAndTagSEGS",
    "UpscaleLatentFromImage",
    "ViTMatteModelLoader",
    "WD14TaggerLoader",
]
