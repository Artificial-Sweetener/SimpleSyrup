# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apply PPM-compatible NegPiP encoding for standard cross-attention models."""

# NegPiP behavior is adapted from ComfyUI-ppm and its credited predecessors.
# See third_party/manifest.toml and third_party/NOTICE.md.

from __future__ import annotations

from typing import Any

import torch
from comfy import model_management
from comfy.sd1_clip import SDClipModel, gen_empty_tokens


def standard_attn2_negpip(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    extra_options: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select magnitude embeddings for keys and signed embeddings for values."""

    del extra_options
    return query, key[:, 0::2], value[:, 1::2]


def encode_token_weights_negpip(
    encoder: SDClipModel,
    token_weight_pairs: list[list[tuple[object, float]]],
) -> tuple[object, ...]:
    """Encode absolute prompt magnitude and interleave signed value embeddings."""

    tokens_to_encode: list[list[object]] = []
    maximum_length = 0
    has_weights = False
    for section in token_weight_pairs:
        tokens = [pair[0] for pair in section]
        maximum_length = max(len(tokens), maximum_length)
        has_weights = has_weights or any(pair[1] != 1.0 for pair in section)
        tokens_to_encode.append(tokens)

    section_count = len(tokens_to_encode)
    if has_weights or section_count == 0:
        if hasattr(encoder, "gen_empty_tokens"):
            empty_tokens = encoder.gen_empty_tokens(
                encoder.special_tokens,
                maximum_length,
            )
        else:
            empty_tokens = gen_empty_tokens(encoder.special_tokens, maximum_length)
        tokens_to_encode.append(empty_tokens)

    encoded = encoder.encode(tokens_to_encode)
    output_tensor, pooled = encoded[:2]
    if not isinstance(output_tensor, torch.Tensor):
        raise TypeError("NegPiP text encoder output must be a tensor.")
    first_pooled = (
        pooled[0:1].to(device=model_management.intermediate_device())
        if isinstance(pooled, torch.Tensor)
        else pooled
    )

    outputs: list[torch.Tensor] = []
    for section_index in range(section_count):
        key_embedding = output_tensor[section_index : section_index + 1].to(copy=True)
        value_embedding = key_embedding.to(copy=True)
        if has_weights:
            empty_embedding = output_tensor[-1]
            for batch_index in range(len(key_embedding)):
                for token_index in range(len(key_embedding[batch_index])):
                    weight = token_weight_pairs[section_index][token_index][1]
                    if weight == 1.0:
                        continue
                    magnitude = abs(weight)
                    key_embedding[batch_index][token_index] = (
                        key_embedding[batch_index][token_index]
                        - empty_embedding[token_index]
                    ) * magnitude + empty_embedding[token_index]
                    value_embedding[batch_index][token_index] = (
                        value_embedding[batch_index][token_index]
                        - empty_embedding[token_index]
                    ) * magnitude + empty_embedding[token_index]
                    if weight < 0.0:
                        value_embedding[batch_index][token_index].neg_()

        interleaved = torch.zeros_like(key_embedding).repeat(1, 2, 1)
        interleaved[:, 0::2, :] = key_embedding
        interleaved[:, 1::2, :] = value_embedding
        outputs.append(interleaved)

    if outputs:
        result: tuple[object, ...] = (
            torch.cat(outputs, dim=-2).to(
                device=model_management.intermediate_device()
            ),
            first_pooled,
        )
    else:
        result = (
            output_tensor[-1:].to(device=model_management.intermediate_device()),
            first_pooled,
        )

    if len(encoded) <= 2:
        return result
    source_extra = encoded[2]
    if not isinstance(source_extra, dict):
        raise TypeError("NegPiP text encoder metadata must be a dictionary.")
    extra: dict[str, object] = {}
    for key, value in source_extra.items():
        if key == "attention_mask" and isinstance(value, torch.Tensor):
            value = (
                value[:section_count]
                .flatten()
                .unsqueeze(dim=0)
                .to(device=model_management.intermediate_device())
            )
        extra[str(key)] = value
    return (*result, extra)
