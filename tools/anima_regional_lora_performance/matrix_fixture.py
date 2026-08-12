"""Build deterministic identical-context tensors and canonical matrix masks."""

from __future__ import annotations

import torch

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)

from .matrix_manifest import (
    PerformanceMaskLayout,
    ScalingPerformanceManifest,
    ScalingPerformanceProfile,
)


def build_matrix_attention_fixture(
    manifest: ScalingPerformanceManifest,
    profile: ScalingPerformanceProfile,
    *,
    device: torch.device,
) -> AnimaRegionalAttentionExecution:
    """Create one normal-CFG attention plan with tensor-identical contexts."""

    if profile not in manifest.profiles:
        raise ValueError("Scaling attention profile must belong to the manifest.")
    base = torch.zeros(
        (2, manifest.context_tokens, manifest.context_features),
        device=device,
        dtype=torch.bfloat16,
    )
    contexts = BatchedRegionalAttentionContexts(
        latent_batch_size=1,
        chunks=(
            RegionalAttentionChunkBatch(
                0,
                RegionalAttentionBranch.POSITIVE,
                0,
                1,
            ),
            RegionalAttentionChunkBatch(
                1,
                RegionalAttentionBranch.NEGATIVE,
                1,
                2,
            ),
        ),
        base_context=base,
        regions=tuple(
            BatchedRegionalAttentionRegion(
                region_index,
                (BatchedRegionalAttentionEntry(0, base, (1.0, 1.0)),),
            )
            for region_index in range(profile.region_count)
        ),
    )
    latent_height = manifest.height // 8
    latent_width = manifest.width // 8
    masks = _canonical_masks(
        profile,
        height=latent_height,
        width=latent_width,
    )
    bank = RegionalMaskBank(
        masks.detach().clone(),
        masks.detach().clone(),
        latent_width,
        latent_height,
    )
    return AnimaRegionalAttentionExecution(
        contexts,
        bank,
        (1.0,) * profile.region_count,
    )


def build_matrix_input_fixture(
    manifest: ScalingPerformanceManifest,
    attention: AnimaRegionalAttentionExecution,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the fixed matrix latent, base context, and descending sigmas."""

    generator = torch.Generator(device=device)
    generator.manual_seed(manifest.seed)
    latent = torch.randn(
        (
            2,
            manifest.latent_channels,
            1,
            manifest.height // 8,
            manifest.width // 8,
        ),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    sample_sigmas = torch.linspace(
        1.0,
        0.0,
        manifest.denoiser_calls + 1,
        device=device,
        dtype=torch.float32,
    )
    return latent, attention.contexts.base_context, sample_sigmas


def _canonical_masks(
    profile: ScalingPerformanceProfile,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Materialize only the declared full-canvas canonical geometry."""

    masks = torch.zeros((profile.region_count, height, width), dtype=torch.float32)
    if profile.mask_layout is PerformanceMaskLayout.HALF_FIRST:
        if profile.region_count != 1:
            raise ValueError("Half-first scaling geometry requires one region.")
        masks[0, :, : width // 2] = 1.0
    elif profile.mask_layout is PerformanceMaskLayout.HARD_PARTITION:
        if width % profile.region_count:
            raise ValueError("Scaling partition width must divide by region count.")
        stripe_width = width // profile.region_count
        for region_index in range(profile.region_count):
            start = region_index * stripe_width
            masks[region_index, :, start : start + stripe_width] = 1.0
    elif profile.mask_layout is PerformanceMaskLayout.FULL_FIRST:
        if profile.region_count != 1:
            raise ValueError("Full-first scaling geometry requires one region.")
        masks[0] = 1.0
    elif profile.mask_layout is PerformanceMaskLayout.REGION_ZERO_FULL:
        if profile.region_count != 4:
            raise ValueError("Region-zero scaling geometry requires four regions.")
        masks[0] = 1.0
    else:
        raise ValueError("Scaling profile has an unsupported mask layout.")
    return masks
