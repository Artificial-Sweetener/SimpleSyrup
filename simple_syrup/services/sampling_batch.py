# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Share latent-batch adaptation across sampling application services."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch

Latent: TypeAlias = dict[str, Any]


def latent_batch_size(latent_image: Latent) -> int:
    """Return the validated number of latent batch items."""

    samples = latent_image.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise TypeError("Sampler latent samples must be a torch.Tensor.")
    if samples.ndim < 1:
        raise ValueError("Sampler latent samples must include a batch axis.")
    return int(samples.shape[0])


def single_item_latent(latent_image: Latent, index: int) -> Latent:
    """Return one latent batch item with aligned batch metadata and mask."""

    samples = latent_image.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise TypeError("Sampler latent samples must be a torch.Tensor.")
    item = latent_image.copy()
    item["samples"] = samples[index : index + 1]
    if "batch_index" in item:
        item["batch_index"] = [item["batch_index"][index]]
    noise_mask = item.get("noise_mask")
    if isinstance(noise_mask, torch.Tensor) and noise_mask.shape[0] == int(
        samples.shape[0]
    ):
        item["noise_mask"] = noise_mask[index : index + 1]
    return item


def combine_latent_outputs(
    latent_image: Latent,
    outputs: list[torch.Tensor],
) -> Latent:
    """Return one latent dictionary containing ordered sampled batch outputs."""

    if not outputs:
        raise ValueError("Sampler batch execution produced no outputs.")
    result = latent_image.copy()
    result.pop("downscale_ratio_spacial", None)
    result["samples"] = torch.cat(outputs, dim=0)
    return result
