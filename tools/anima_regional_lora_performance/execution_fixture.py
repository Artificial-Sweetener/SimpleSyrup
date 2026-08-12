"""Build deterministic Anima performance attention and tensor inputs."""

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

from .manifest import PerformanceManifest


def build_attention_fixture(
    manifest: PerformanceManifest,
    *,
    device: torch.device,
) -> AnimaRegionalAttentionExecution:
    """Build normal-CFG contexts and one hard half-canvas region."""

    base = torch.zeros(
        (2, manifest.context_tokens, manifest.context_features),
        device=device,
        dtype=torch.bfloat16,
    )
    region = torch.full_like(base, 0.01)
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
        regions=(
            BatchedRegionalAttentionRegion(
                0,
                (BatchedRegionalAttentionEntry(0, region, (1.0, 1.0)),),
            ),
        ),
    )
    latent_height = manifest.height // 8
    latent_width = manifest.width // 8
    mask = torch.zeros((1, latent_height, latent_width), dtype=torch.float32)
    mask[:, :, : latent_width // 2] = 1.0
    bank = RegionalMaskBank(
        mask.detach().clone(),
        mask.detach().clone(),
        latent_width,
        latent_height,
    )
    return AnimaRegionalAttentionExecution(contexts, bank, (1.0,))


def build_input_fixture(
    manifest: PerformanceManifest,
    attention: AnimaRegionalAttentionExecution,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create fixed latent, context, and descending sample sigmas."""

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
