# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose the benchmark-only static Anima regional profile node."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from .anima_regional_profile import (
    STATIC_ANIMA_REGIONAL_PROFILE_BUILDER,
    StaticAnimaRegionalProfileBuilder,
)
from .anima_regional_profile_spec import decode_visual_regional_adapters

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Provide the type-checking boundary for the benchmark node."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io
_mixed_conditioning_io: Any = (
    None if TYPE_CHECKING else _comfy_io.Custom("CONDITIONING,CONDITIONING_BATCH")
)


class StaticAnimaRegionalProfileV3(_ComfyNodeBase):
    """Delegate one fixed visual-evidence model profile to its runtime owner."""

    profile_builder: ClassVar[StaticAnimaRegionalProfileBuilder] = (
        STATIC_ANIMA_REGIONAL_PROFILE_BUILDER
    )

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the dev-only fixed-profile schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.StaticAnimaRegionalProfile",
            display_name="Benchmark Static Anima Regional Profile",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Model.Input("model"),
                _mixed_conditioning_io.Input("positive"),
                _mixed_conditioning_io.Input("negative"),
                _comfy_io.Mask.Input("region_masks"),
                _comfy_io.Latent.Input("latent"),
                _comfy_io.String.Input("regional_adapters_json", multiline=True),
            ],
            outputs=[
                _comfy_io.Model.Output("model"),
                _comfy_io.Conditioning.Output("positive"),
                _comfy_io.Conditioning.Output("negative"),
            ],
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        positive: object,
        negative: object,
        region_masks: object,
        latent: dict[str, Any],
        regional_adapters_json: str,
    ) -> Any:
        """Return the derived benchmark model without owning profile behavior."""

        built = cls.profile_builder.build(
            model=model,
            positive=positive,
            negative=negative,
            region_masks=region_masks,
            latent=latent,
            adapters=decode_visual_regional_adapters(regional_adapters_json),
        )
        return _comfy_io.NodeOutput(built.model, built.positive, built.negative)
