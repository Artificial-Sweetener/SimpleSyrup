# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own compatible sequence alignment for standard-UNet model calls."""

from __future__ import annotations

from ..regional_attention_batching import RegionalAttentionBatchingService
from ..regional_attention_model_call import RegionalAttentionModelCallResolver
from ..regional_attention_sequence_alignment import (
    REGIONAL_ATTENTION_SEQUENCE_ALIGNER,
)

STANDARD_UNET_MODEL_CALL_RESOLVER = RegionalAttentionModelCallResolver(
    batching=RegionalAttentionBatchingService(
        sequence_aligner=REGIONAL_ATTENTION_SEQUENCE_ALIGNER
    )
)
