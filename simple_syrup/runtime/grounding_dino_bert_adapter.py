# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt GroundingDINO attention masks to supported Transformers BERT APIs."""

from __future__ import annotations

import torch
import transformers
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from torch import nn

SUPPORTED_TRANSFORMERS = SpecifierSet(">=4.50.3,<6")
SUPPORTED_TRANSFORMERS_LABEL = "Transformers >=4.50.3,<6"


def prepare_grounding_dino_bert_attention_mask(
    attention_mask: torch.Tensor,
    *,
    text_encoder: nn.Module,
    transformers_version: str | None = None,
) -> torch.Tensor:
    """Prepare one GroundingDINO mask for the installed native BERT forward API.

    GroundingDINO supplies a three-dimensional Boolean sub-sentence mask. The
    final Transformers v4 line accepts that representation directly, while v5
    accepts an already prepared four-dimensional additive mask. Ordinary
    two-dimensional tokenizer padding masks remain owned by native BERT.

    Args:
        attention_mask: Tokenizer padding mask or GroundingDINO block mask.
        text_encoder: BERT module whose parameter dtype determines mask dtype.
        transformers_version: Explicit version used by deterministic tests.

    Returns:
        The representation accepted by the selected native BERT generation.

    Raises:
        RuntimeError: The Transformers version is outside the supported range.
        ValueError: The GroundingDINO block mask or encoder dtype is invalid.
    """

    version = _validated_transformers_version(
        transformers_version or transformers.__version__
    )
    if attention_mask.ndim == 2:
        return attention_mask
    if attention_mask.ndim != 3:
        raise ValueError(
            "GroundingDINO BERT attention masks must have two or three dimensions."
        )
    if attention_mask.dtype is not torch.bool:
        raise ValueError("GroundingDINO sub-sentence attention masks must be boolean.")
    if attention_mask.shape[-2] != attention_mask.shape[-1]:
        raise ValueError("GroundingDINO sub-sentence attention masks must be square.")
    if version.major == 4:
        return attention_mask

    parameter = next(text_encoder.parameters(), None)
    if parameter is None or not parameter.dtype.is_floating_point:
        raise ValueError(
            "GroundingDINO's BERT text encoder must have floating-point parameters."
        )
    allowed_tokens = attention_mask.unsqueeze(1)
    blocked_value = torch.finfo(parameter.dtype).min
    return torch.full(
        allowed_tokens.shape,
        blocked_value,
        dtype=parameter.dtype,
        device=attention_mask.device,
    ).masked_fill(allowed_tokens, 0.0)


def _validated_transformers_version(version_text: str) -> Version:
    """Parse and enforce the Transformers API generations verified by the project."""

    try:
        version = Version(version_text)
    except InvalidVersion as error:
        raise RuntimeError(
            f"{SUPPORTED_TRANSFORMERS_LABEL} is required for GroundingDINO; "
            f"found invalid version {version_text!r}."
        ) from error
    if version not in SUPPORTED_TRANSFORMERS:
        raise RuntimeError(
            f"{SUPPORTED_TRANSFORMERS_LABEL} is required for GroundingDINO; "
            f"found {version}."
        )
    return version
