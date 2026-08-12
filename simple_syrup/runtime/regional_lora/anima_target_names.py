# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Normalize supported external Anima LoRA target names."""

from __future__ import annotations

import re

_SD_SCRIPTS_TARGET = re.compile(r"^lora_unet_blocks_(\d+)_(.+)$")
_SD_SCRIPTS_FAMILIES = {
    "self_attn_q_proj": "self_attn.q_proj",
    "self_attn_k_proj": "self_attn.k_proj",
    "self_attn_v_proj": "self_attn.v_proj",
    "self_attn_output_proj": "self_attn.output_proj",
    "cross_attn_q_proj": "cross_attn.q_proj",
    "cross_attn_k_proj": "cross_attn.k_proj",
    "cross_attn_v_proj": "cross_attn.v_proj",
    "cross_attn_output_proj": "cross_attn.output_proj",
    "mlp_layer1": "mlp.layer1",
    "mlp_layer2": "mlp.layer2",
    "adaln_modulation_self_attn_1": "adaln_modulation_self_attn.1",
    "adaln_modulation_self_attn_2": "adaln_modulation_self_attn.2",
    "adaln_modulation_cross_attn_1": "adaln_modulation_cross_attn.1",
    "adaln_modulation_cross_attn_2": "adaln_modulation_cross_attn.2",
    "adaln_modulation_mlp_1": "adaln_modulation_mlp.1",
    "adaln_modulation_mlp_2": "adaln_modulation_mlp.2",
}


class AnimaLoraTargetNameNormalizer:
    """Translate recognized Anima aliases into canonical model paths."""

    def normalize(self, target: str) -> str:
        """Return a canonical path while leaving unknown names rejectable."""

        match = _SD_SCRIPTS_TARGET.fullmatch(target)
        if match is None:
            return target
        family = _SD_SCRIPTS_FAMILIES.get(match.group(2))
        if family is None:
            return target
        return f"diffusion_model.blocks.{match.group(1)}.{family}"


ANIMA_LORA_TARGET_NAME_NORMALIZER = AnimaLoraTargetNameNormalizer()
