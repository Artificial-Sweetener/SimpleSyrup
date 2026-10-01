# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026 Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configure regional attention without applying MODEL patches in the options graph."""

from __future__ import annotations

from typing import Any

from ..domain.sampler_options import (
    AttentionCouplingOptions,
    SamplerOptions,
    append_sampler_capability,
)
from .ksampler_schema import attention_coupling_ksampler_inputs
from .sampler_options_schema import (
    COMFY_IO,
    OptionsNodeBase,
    options_input,
    options_output,
)


class AttentionCouplingOptionsV3(OptionsNodeBase):
    """Pair global-first conditioning on the sampler with ordered region masks."""

    @classmethod
    def define_schema(cls) -> Any:
        """Expose regional masks with established strength and feathering controls."""
        controls = attention_coupling_ksampler_inputs(COMFY_IO)
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.AttentionCouplingOptions",
            display_name="Attention Coupling Options",
            category="SimpleSyrup/Sampling/Options",
            description=(
                "Routes global-first sampler conditioning to ordered image "
                "regions through regional attention and LoRA hooks."
            ),
            inputs=[
                options_input(COMFY_IO),
                *[
                    control
                    for control in controls
                    if control.id
                    in {"region_masks", "regional_prompt_weight", "region_mask_feather"}
                ],
            ],
            outputs=[options_output(COMFY_IO)],
        )

    @classmethod
    def execute(
        cls,
        region_masks: object,
        regional_prompt_weight: float = 1.0,
        region_mask_feather: int = 0,
        options: SamplerOptions | None = None,
    ) -> tuple[SamplerOptions]:
        """Append regional attention controls while deferring model preparation."""
        return (
            append_sampler_capability(
                options,
                AttentionCouplingOptions(
                    region_masks=region_masks,
                    regional_prompt_weight=regional_prompt_weight,
                    region_mask_feather=region_mask_feather,
                ),
            ),
        )
