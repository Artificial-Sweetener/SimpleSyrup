# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 wrappers for nodes whose behavior still lives in legacy modules."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

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
from ..nodes.ksampler_tiled_diffusion import KSamplerTiledDiffusion
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
from ..nodes.simple_load_anima import SimpleLoadAnima
from ..nodes.vitmatte_model_loader import ViTMatteModelLoader

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        hidden: ClassVar[Any]
        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io

_HIDDEN_INPUTS = {
    "PROMPT": "prompt",
    "DYNPROMPT": "dynprompt",
    "EXTRA_PNGINFO": "extra_pnginfo",
    "UNIQUE_ID": "unique_id",
    "AUTH_TOKEN_COMFY_ORG": "auth_token_comfy_org",
    "API_KEY_COMFY_ORG": "api_key_comfy_org",
}


class LegacyNodeV3Adapter(_ComfyNodeBase):
    """Build a v3 schema and execution bridge for a legacy implementation class."""

    LEGACY_NODE_CLASS: ClassVar[type[Any]]
    NODE_ID: ClassVar[str]
    DISPLAY_NAME: ClassVar[str]
    ENABLE_EXPAND: ClassVar[bool] = False

    @classmethod
    def define_schema(cls) -> Any:
        """Declare a v3 schema from the implementation class contract."""

        legacy = cls.LEGACY_NODE_CLASS
        return _comfy_io.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category=str(getattr(legacy, "CATEGORY", "SimpleSyrup")),
            description=str(getattr(legacy, "DESCRIPTION", "")),
            search_aliases=list(getattr(legacy, "SEARCH_ALIASES", [])),
            inputs=_v3_inputs(legacy.INPUT_TYPES()),
            outputs=_v3_outputs(legacy),
            hidden=_v3_hidden_inputs(legacy.INPUT_TYPES()),
            is_input_list=bool(getattr(legacy, "INPUT_IS_LIST", False)),
            is_output_node=bool(getattr(legacy, "OUTPUT_NODE", False)),
            enable_expand=cls.ENABLE_EXPAND,
        )

    @classmethod
    def execute(cls, **kwargs: object) -> Any:
        """Run the wrapped implementation with v3-provided inputs."""

        values = dict(kwargs)
        for name, hidden_attr in _legacy_hidden_inputs(
            cls.LEGACY_NODE_CLASS.INPUT_TYPES()
        ).items():
            if name not in values:
                values[name] = getattr(cls.hidden, hidden_attr)

        function_name = str(cls.LEGACY_NODE_CLASS.FUNCTION)
        implementation = cls.LEGACY_NODE_CLASS()
        function = getattr(implementation, function_name)
        return function(**values)


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


class KSamplerTiledDiffusionV3(LegacyNodeV3Adapter):
    """Expose KSampler Tiled Diffusion through Comfy v3 only."""

    LEGACY_NODE_CLASS = KSamplerTiledDiffusion
    NODE_ID = "SimpleSyrup.KSamplerTiledDiffusion"
    DISPLAY_NAME = "KSampler (Tiled Diffusion)"


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


class DetailSEGSAsRegionsV3(LegacyNodeV3Adapter):
    """Expose Detail SEGS as Regions through Comfy v3 only."""

    LEGACY_NODE_CLASS = DetailSEGSAsRegions
    NODE_ID = "SimpleSyrup.DetailSEGSAsRegions"
    DISPLAY_NAME = "Detail SEGS as Regions"


class DetailSEGSByScaleFactorV3(LegacyNodeV3Adapter):
    """Expose Detail SEGS by Scale Factor through Comfy v3 only."""

    LEGACY_NODE_CLASS = DetailSEGSByScaleFactor
    NODE_ID = "SimpleSyrup.DetailSEGSByScaleFactor"
    DISPLAY_NAME = "Detail SEGS by Scale Factor"


class DetailSEGSByScaleFactorTiledDiffusionV3(LegacyNodeV3Adapter):
    """Expose Detail SEGS by Scale Factor with Tiled Diffusion through Comfy v3."""

    LEGACY_NODE_CLASS = DetailSEGSByScaleFactorTiledDiffusion
    NODE_ID = "SimpleSyrup.DetailSEGSByScaleFactorTiledDiffusion"
    DISPLAY_NAME = "Detail SEGS by Scale Factor w/ Tiled Diffusion"


class SAMModelLoaderV3(LegacyNodeV3Adapter):
    """Expose SAM Model Loader through Comfy v3 only."""

    LEGACY_NODE_CLASS = SAMModelLoader
    NODE_ID = "SimpleSyrup.SAMModelLoader"
    DISPLAY_NAME = "SAM Model Loader"


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


def _v3_inputs(input_types: Mapping[str, Mapping[str, object]]) -> list[Any]:
    """Return v3 input declarations from legacy required and optional inputs."""

    inputs: list[Any] = []
    for section_name, optional in (("required", False), ("optional", True)):
        section = input_types.get(section_name, {})
        for name, declaration in section.items():
            inputs.append(_v3_input(name, declaration, optional=optional))
    return inputs


def _v3_input(name: str, declaration: object, *, optional: bool) -> Any:
    """Return one v3 input declaration from a legacy field declaration."""

    if not isinstance(declaration, tuple) or not declaration:
        raise TypeError(f"legacy input {name} declaration must be a tuple.")

    io_declaration = declaration[0]
    options = _input_options(declaration)
    tooltip = _string_option(options, "tooltip")
    advanced = _bool_option(options, "advanced")
    raw_link = _bool_option(options, "rawLink") or _bool_option(options, "raw_link")
    force_input = _bool_option(options, "forceInput") or _bool_option(
        options, "force_input"
    )

    if isinstance(io_declaration, (list, tuple)):
        return _comfy_io.Combo.Input(
            name,
            options=list(io_declaration),
            optional=optional,
            default=options.get("default"),
            control_after_generate=options.get("control_after_generate"),
            tooltip=tooltip,
            raw_link=raw_link,
            advanced=advanced,
        )

    if not isinstance(io_declaration, str):
        raise TypeError(f"legacy input {name} type must be a string or options list.")

    input_type = io_declaration
    input_class = _io_class(input_type)
    common_options = {
        "optional": optional,
        "tooltip": tooltip,
        "raw_link": raw_link,
        "advanced": advanced,
    }

    if input_type == "INT":
        return input_class.Input(
            name,
            default=options.get("default"),
            min=options.get("min"),
            max=options.get("max"),
            step=options.get("step"),
            control_after_generate=options.get("control_after_generate"),
            **common_options,
        )
    if input_type == "FLOAT":
        return input_class.Input(
            name,
            default=options.get("default"),
            min=options.get("min"),
            max=options.get("max"),
            step=options.get("step"),
            round=options.get("round"),
            **common_options,
        )
    if input_type == "STRING":
        return input_class.Input(
            name,
            default=options.get("default"),
            multiline=bool(options.get("multiline", False)),
            force_input=force_input,
            **common_options,
        )
    if input_type == "BOOLEAN":
        return input_class.Input(
            name,
            default=options.get("default"),
            label_on=options.get("label_on"),
            label_off=options.get("label_off"),
            **common_options,
        )

    return input_class.Input(name, **common_options)


def _v3_outputs(legacy: type[Any]) -> list[Any]:
    """Return v3 output declarations from legacy return metadata."""

    return_types = tuple(getattr(legacy, "RETURN_TYPES", ()))
    return_names = getattr(legacy, "RETURN_NAMES", None)
    output_tooltips = tuple(getattr(legacy, "OUTPUT_TOOLTIPS", ()))
    output_is_list = tuple(
        getattr(legacy, "OUTPUT_IS_LIST", (False,) * len(return_types))
    )
    outputs: list[Any] = []
    for index, io_type in enumerate(return_types):
        output_name = None
        if isinstance(return_names, tuple) and index < len(return_names):
            output_name = str(return_names[index])
        tooltip = None
        if index < len(output_tooltips):
            tooltip = str(output_tooltips[index])
        is_output_list = index < len(output_is_list) and bool(output_is_list[index])
        outputs.append(
            _io_class(str(io_type)).Output(
                output_name,
                tooltip=tooltip,
                is_output_list=is_output_list,
            )
        )
    return outputs


def _v3_hidden_inputs(input_types: Mapping[str, Mapping[str, object]]) -> list[Any]:
    """Return v3 hidden declarations requested by legacy hidden inputs."""

    hidden_values = set(_legacy_hidden_inputs(input_types).values())
    return [getattr(_comfy_io.Hidden, value) for value in sorted(hidden_values)]


def _legacy_hidden_inputs(
    input_types: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    """Return legacy hidden input names mapped to v3 hidden holder attributes."""

    hidden_inputs: dict[str, str] = {}
    for name, sentinel in input_types.get("hidden", {}).items():
        if isinstance(sentinel, str) and sentinel in _HIDDEN_INPUTS:
            hidden_inputs[name] = _HIDDEN_INPUTS[sentinel]
    return hidden_inputs


def _io_class(io_type: str) -> Any:
    """Return the v3 IO class for a legacy Comfy type string."""

    known_types = {
        "BOOLEAN": _comfy_io.Boolean,
        "INT": _comfy_io.Int,
        "FLOAT": _comfy_io.Float,
        "STRING": _comfy_io.String,
        "IMAGE": _comfy_io.Image,
        "MASK": _comfy_io.Mask,
        "LATENT": _comfy_io.Latent,
        "MODEL": _comfy_io.Model,
        "CLIP": _comfy_io.Clip,
        "VAE": _comfy_io.Vae,
        "CONDITIONING": _comfy_io.Conditioning,
        "SEGS": _comfy_io.SEGS,
    }
    return known_types.get(io_type, _comfy_io.Custom(io_type))


def _input_options(declaration: tuple[object, ...]) -> dict[str, object]:
    """Return an input options dictionary from a legacy declaration."""

    if len(declaration) < 2 or not isinstance(declaration[1], dict):
        return {}
    return dict(declaration[1])


def _string_option(options: Mapping[str, object], name: str) -> str | None:
    """Return a string option when present."""

    value = options.get(name)
    if isinstance(value, str):
        return value
    return None


def _bool_option(options: Mapping[str, object], name: str) -> bool | None:
    """Return a boolean option when present."""

    value = options.get(name)
    if isinstance(value, bool):
        return value
    return None


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
    "KSamplerTiledDiffusionV3",
    "LatentDiagnosticsV3",
    "LayerStyleSAMModelsAdapterV3",
    "LoadUltralyticsModelV3",
    "PromptEncodeStyleAndNormalizationV3",
    "PromptEncodeStyleV3",
    "PromptSEGSWithSAMV3",
    "ResizeImageToTargetV3",
    "SAMModelLoaderV3",
    "SeedV3",
    "SimpleLoadAnimaV3",
    "SimpleVAEEncodeV3",
    "UpscaleLatentFromImageV3",
    "ViTMatteModelLoaderV3",
]
