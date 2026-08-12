# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove complete-output regional blending at Anima cross-attention only."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import comfy.ops
import pytest
import torch
from comfy.ldm.anima.model import Anima
from comfy.ldm.cosmos.predict2 import Attention, Block
from regional_attention_test_values import single_entry_regions
from torch import nn

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.patcher_lifecycle import PATCHER_LIFECYCLE
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationContext,
    AnimaActivationGeometry,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_branch_batch import (
    AnimaRegionalBranchInvocation,
)
from simple_syrup.runtime.regional_lora.anima_composition_phase import (
    AnimaCompositionPhase,
    AnimaCompositionStage,
)
from simple_syrup.runtime.regional_lora.anima_composition_phase_context import (
    AnimaCompositionPhaseContext,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention import (
    AnimaRegionalCrossAttentionPatch,
    anima_cross_attention_mutations,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention_context import (
    AnimaCrossAttentionInvocationContext,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import (
    ANIMA_MODULE_SURFACE_DISCOVERY,
    AnimaModuleSurface,
)
from simple_syrup.runtime.regional_lora.anima_targets import ANIMA_BLOCK_COUNT


class _DeterministicCrossAttention(nn.Module):
    """Return context-owned constants while recording one complete branch call."""

    def __init__(
        self,
        invocation_context: AnimaCrossAttentionInvocationContext,
        callback: Callable[[], None] | None = None,
    ) -> None:
        """Initialize an empty call log and optional failure callback."""

        super().__init__()
        _install_cross_attention_children(self)
        self.invocation_context = invocation_context
        self.callback = callback
        self.calls: list[tuple[torch.Tensor, torch.Tensor, object, object]] = []
        self.invocations: list[AnimaRegionalBranchInvocation] = []

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        rope_emb: object,
        transformer_options: object,
    ) -> torch.Tensor:
        """Return each branch context mean at every query token."""

        self.calls.append((x, context, rope_emb, transformer_options))
        self.invocations.append(self.invocation_context.require_current())
        if self.callback is not None:
            self.callback()
        values = context.mean(dim=1, keepdim=True)
        return values.expand(-1, int(x.shape[1]), -1)


class _InvalidOutputAttention(nn.Module):
    """Return a deliberately truncated branch batch."""

    def __init__(self) -> None:
        """Expose the installed attention child contract."""

        super().__init__()
        _install_cross_attention_children(self)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Ignore inputs and return one invalid output shape."""

        del args, kwargs
        return torch.zeros((1, 1, 1))


class _FullRegionalPhaseContext(AnimaCompositionPhaseContext):
    """Publish specialization behavior for focused attention tests."""

    def require_current(self) -> AnimaCompositionPhase:
        """Return one stable phase without global composition residual."""

        return AnimaCompositionPhase(
            AnimaCompositionStage.SPECIALIZATION,
            0.5,
            True,
            1.0,
        )


class _AnimaModelRoot(nn.Module):
    """Expose the installed diffusion model at Comfy's object-patch root path."""

    def __init__(self, diffusion_model: nn.Module) -> None:
        """Retain one diffusion model under its installed owner name."""

        super().__init__()
        self.diffusion_model = diffusion_model


def _install_cross_attention_children(module: nn.Module) -> None:
    """Install the seven host-traversed attention children on one test double."""

    module.q_proj = nn.Identity()
    module.q_norm = nn.Identity()
    module.k_proj = nn.Identity()
    module.k_norm = nn.Identity()
    module.v_proj = nn.Identity()
    module.output_proj = nn.Identity()
    module.output_dropout = nn.Identity()


@pytest.fixture(scope="module")
def anima_surface() -> AnimaModuleSurface:
    """Discover the exact installed Anima graph with allocation-free weights."""

    model = Anima(
        max_img_h=2,
        max_img_w=2,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=2048,
        num_blocks=ANIMA_BLOCK_COUNT,
        num_heads=16,
        mlp_ratio=4.0,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d",
        pos_emb_learnable=False,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=256,
        extra_per_block_abs_pos_emb=False,
        device=torch.device("meta"),
        dtype=torch.float16,
        operations=comfy.ops.disable_weight_init,
    )
    return ANIMA_MODULE_SURFACE_DISCOVERY.discover(model)


@pytest.mark.parametrize(
    ("masks", "expected", "expected_regions"),
    [
        (
            torch.zeros((2, 2, 2)),
            torch.zeros((1, 4, 1)),
            (None,),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ]
            ),
            torch.tensor([[[4.0 / 3.0], [8.0 / 3.0], [4.0 / 3.0], [8.0 / 3.0]]]),
            (None, 0, 1),
        ),
        (
            torch.ones((2, 2, 2)),
            torch.full((1, 4, 1), 2.0),
            (None, 0, 1),
        ),
    ],
)
def test_patch_batches_complete_branches_and_blends_expected_outputs(
    masks: torch.Tensor,
    expected: torch.Tensor,
    expected_regions: tuple[int | None, ...],
) -> None:
    """Satisfy zero, hard-partition, and overlap output invariants exactly."""

    activation_context = AnimaActivationContext()
    invocation_context = AnimaCrossAttentionInvocationContext()
    original = _DeterministicCrossAttention(invocation_context)
    base = torch.zeros((1, 2, 1))
    regions = (torch.full_like(base, 2.0), torch.full_like(base, 4.0))
    execution = _execution(base, regions, masks)
    patch = AnimaRegionalCrossAttentionPatch(
        original,
        execution,
        activation_context=activation_context,
        invocation_context=invocation_context,
        phase_context=_FullRegionalPhaseContext(),
    )
    query = torch.full((1, 4, 1), 99.0)
    options = {"preserved": object()}
    rope = object()

    with activation_context.activate(_geometry(batch=1, height=2, width=2)):
        output = patch(
            query,
            base,
            rope_emb=rope,
            transformer_options=options,
        )

    torch.testing.assert_close(output, expected)
    assert len(original.calls) == 1
    branch_x, branch_context, observed_rope, observed_options = original.calls[0]
    assert branch_x.shape == (len(expected_regions), 4, 1)
    expected_context = torch.cat(
        tuple(
            base if region is None else regions[region] for region in expected_regions
        )
    )
    assert torch.equal(branch_context, expected_context)
    assert observed_rope is rope
    assert observed_options is options
    assert original.invocations == [
        AnimaRegionalBranchInvocation(
            1,
            (0,) * len(expected_regions),
            expected_regions,
        )
    ]
    assert invocation_context.current_or_none() is None


def test_cross_attention_backing_module_does_not_leak_a_host_weight_namespace() -> None:
    """Keep the retained installed attention outside PyTorch child discovery."""

    activation_context = AnimaActivationContext()
    invocation_context = AnimaCrossAttentionInvocationContext()
    original = _DeterministicCrossAttention(invocation_context)
    base = torch.zeros((1, 1, 1))
    execution = _execution(base, (base.clone(),), torch.ones((1, 1, 1)))
    patch = AnimaRegionalCrossAttentionPatch(
        original,
        execution,
        activation_context=activation_context,
        invocation_context=invocation_context,
        phase_context=_FullRegionalPhaseContext(),
    )

    assert "original" not in dict(patch.named_modules())
    original_projection = original.q_proj
    patch.q_proj = nn.Identity()
    assert original.q_proj is original_projection


def test_patch_aligns_view_major_masks_with_chunk_and_latent_contexts() -> None:
    """Blend each view over its complete chunk-major source model batch."""

    activation_context = AnimaActivationContext()
    invocation_context = AnimaCrossAttentionInvocationContext()
    original = _DeterministicCrossAttention(invocation_context)
    base = torch.zeros((4, 1, 1))
    regions = (
        torch.tensor([2.0, 12.0, 22.0, 32.0]).reshape(4, 1, 1),
        torch.tensor([4.0, 14.0, 24.0, 34.0]).reshape(4, 1, 1),
    )
    masks = torch.tensor(
        [
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        ]
    )
    execution = _execution(base, regions, masks, latent_batch_size=1)
    patch = AnimaRegionalCrossAttentionPatch(
        original,
        execution,
        activation_context=activation_context,
        invocation_context=invocation_context,
        phase_context=_FullRegionalPhaseContext(),
    )
    layout = SpatialBatchLayout(
        canvas_width=4,
        canvas_height=2,
        views=(
            SpatialView(SpatialViewKind.TILE, 0, 0, 2, 2, 2, 2),
            SpatialView(SpatialViewKind.TILE, 2, 0, 2, 2, 2, 2),
        ),
        input_batch_size=2,
    )
    geometry = _geometry(batch=4, height=2, width=2, layout=layout)

    with activation_context.activate(geometry):
        output = patch(torch.zeros((4, 4, 1)), base)

    expected_batches = torch.tensor([4.0 / 3.0, 8.0, 16.0, 68.0 / 3.0])
    torch.testing.assert_close(output[:, 0, 0], expected_batches)
    assert len(original.calls) == 1


def test_patch_returns_one_output_for_one_unchanged_residual_path() -> None:
    """Leave self-attention, residual addition, AdaLN, and MLP outside duplication."""

    activation_context = AnimaActivationContext()
    invocation_context = AnimaCrossAttentionInvocationContext()
    original = _DeterministicCrossAttention(invocation_context)
    base = torch.zeros((1, 1, 1))
    execution = _execution(
        base,
        (torch.full_like(base, 2.0),),
        torch.ones((1, 1, 1)),
    )
    patch = AnimaRegionalCrossAttentionPatch(
        original,
        execution,
        activation_context=activation_context,
        invocation_context=invocation_context,
        phase_context=_FullRegionalPhaseContext(),
    )
    counts = {"self": 0, "residual": 0, "mlp": 0}

    def deterministic_block(x: torch.Tensor) -> torch.Tensor:
        """Model the unchanged owners surrounding the patched cross-attention."""

        counts["self"] += 1
        with activation_context.activate(_geometry(batch=1, height=1, width=1)):
            cross_output = patch(x, base)
        counts["residual"] += 1
        x = x + cross_output
        counts["mlp"] += 1
        return x + 3.0

    output = deterministic_block(torch.ones((1, 1, 1)))

    torch.testing.assert_close(output, torch.full((1, 1, 1), 16.0 / 3.0))
    assert counts == {"self": 1, "residual": 1, "mlp": 1}
    assert len(original.calls) == 1


def test_patch_combines_all_active_entries_before_spatial_normalization() -> None:
    """Apply Comfy entry strengths inside a region without choosing the first."""

    activation_context = AnimaActivationContext()
    invocation_context = AnimaCrossAttentionInvocationContext()
    original = _DeterministicCrossAttention(invocation_context)
    base = torch.zeros((1, 1, 1))
    contexts = BatchedRegionalAttentionContexts(
        1,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),),
        base,
        (
            BatchedRegionalAttentionRegion(
                0,
                (
                    BatchedRegionalAttentionEntry(
                        0, torch.full_like(base, 2.0), (0.25,)
                    ),
                    BatchedRegionalAttentionEntry(
                        1, torch.full_like(base, 6.0), (0.75,)
                    ),
                ),
            ),
        ),
    )
    patch = AnimaRegionalCrossAttentionPatch(
        original,
        AnimaRegionalAttentionExecution(
            contexts,
            _bank(torch.ones((1, 1, 1))),
            (1.0,),
        ),
        activation_context=activation_context,
        invocation_context=invocation_context,
        phase_context=_FullRegionalPhaseContext(),
    )

    with activation_context.activate(_geometry(batch=1, height=1, width=1)):
        output = patch(torch.zeros((1, 1, 1)), base)

    torch.testing.assert_close(output, torch.full_like(output, 10.0 / 3.0))
    assert original.invocations == [
        AnimaRegionalBranchInvocation(1, (0, 0, 0), (None, 0, 0))
    ]
    assert original.calls[0][0].shape[0] == 3


def test_invocation_context_restores_after_original_attention_failure() -> None:
    """Clear branch identity even when the original attention implementation fails."""

    def fail() -> None:
        """Raise one deterministic original-attention failure."""

        raise RuntimeError("attention failure")

    activation_context = AnimaActivationContext()
    invocation_context = AnimaCrossAttentionInvocationContext()
    original = _DeterministicCrossAttention(invocation_context, fail)
    base = torch.zeros((1, 1, 1))
    patch = AnimaRegionalCrossAttentionPatch(
        original,
        _execution(base, (torch.ones_like(base),), torch.ones((1, 1, 1))),
        activation_context=activation_context,
        invocation_context=invocation_context,
        phase_context=_FullRegionalPhaseContext(),
    )

    with activation_context.activate(_geometry(batch=1, height=1, width=1)):
        with pytest.raises(RuntimeError, match="attention failure"):
            patch(torch.zeros((1, 1, 1)), base)

    assert invocation_context.current_or_none() is None


def test_builds_and_injects_all_28_exact_clone_local_object_patches(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Replace and restore every discovered cross-attention owner without globals."""

    base = torch.zeros((1, 1, 1024))
    execution = _execution(
        base,
        (torch.ones_like(base),),
        torch.ones((1, 2, 2)),
    )
    activation_context = AnimaActivationContext()
    invocation_context = AnimaCrossAttentionInvocationContext()
    mutations = anima_cross_attention_mutations(
        anima_surface,
        execution,
        activation_context=activation_context,
        invocation_context=invocation_context,
        phase_context=_FullRegionalPhaseContext(),
    )
    root = _AnimaModelRoot(anima_surface.diffusion_model)
    source = _patcher(root)
    original_block_forward = Block.forward
    original_attention_forward = Attention.forward

    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        mutations,
        operation="Anima regional cross-attention object patches",
    )

    assert len(mutations) == ANIMA_BLOCK_COUNT
    assert tuple(mutation.path for mutation in mutations) == tuple(
        f"diffusion_model.blocks.{index}.cross_attn"
        for index in range(ANIMA_BLOCK_COUNT)
    )
    assert source.object_patches == {}
    assert len(derived.object_patches) == ANIMA_BLOCK_COUNT
    derived.patch_model(load_weights=False)
    try:
        for block_index, block in enumerate(anima_surface.diffusion_model.blocks):
            replacement = derived.object_patches[
                f"diffusion_model.blocks.{block_index}.cross_attn"
            ]
            assert block.cross_attn is replacement
            assert isinstance(replacement, AnimaRegionalCrossAttentionPatch)
            assert (
                replacement.q_proj
                is anima_surface.blocks[block_index].cross_attention.q_proj
            )
            assert (
                replacement.k_proj
                is anima_surface.blocks[block_index].cross_attention.k_proj
            )
            assert (
                replacement.v_proj
                is anima_surface.blocks[block_index].cross_attention.v_proj
            )
            assert (
                replacement.output_proj
                is anima_surface.blocks[block_index].cross_attention.output_proj
            )
            assert (
                replacement.q_norm
                is anima_surface.blocks[block_index].cross_attention.q_norm
            )
            assert (
                replacement.k_norm
                is anima_surface.blocks[block_index].cross_attention.k_norm
            )
            assert (
                replacement.output_dropout
                is anima_surface.blocks[block_index].cross_attention.output_dropout
            )
    finally:
        derived.unpatch_model(unpatch_weights=False)

    assert all(
        block.cross_attn is anima_surface.blocks[index].cross_attention
        for index, block in enumerate(anima_surface.diffusion_model.blocks)
    )
    assert Block.forward is original_block_forward
    assert Attention.forward is original_attention_forward


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ("query_batch", "query batch does not match"),
        ("query_length", "query length does not match"),
        ("context_shape", "base cross-attention context does not match"),
        ("activation_batch", "activation batch does not match"),
        ("canvas", "activation H/W must match"),
        ("output", "returned an invalid branch batch"),
    ],
)
def test_patch_rejects_every_execution_alignment_failure(
    mutate: str,
    message: str,
) -> None:
    """Fail closed before returning a misordered or mis-shaped blended output."""

    activation_context = AnimaActivationContext()
    invocation_context = AnimaCrossAttentionInvocationContext()
    base = torch.zeros((1, 1, 1))
    original: nn.Module = (
        _InvalidOutputAttention()
        if mutate == "output"
        else _DeterministicCrossAttention(invocation_context)
    )
    execution = _execution(
        base,
        (torch.ones_like(base),),
        torch.ones((1, 2, 2)),
    )
    patch = AnimaRegionalCrossAttentionPatch(
        original,
        execution,
        activation_context=activation_context,
        invocation_context=invocation_context,
        phase_context=_FullRegionalPhaseContext(),
    )
    query = torch.zeros((1, 4, 1))
    context = base
    geometry = _geometry(batch=1, height=2, width=2)
    if mutate == "query_batch":
        query = torch.zeros((2, 4, 1))
    elif mutate == "query_length":
        query = torch.zeros((1, 3, 1))
    elif mutate == "context_shape":
        context = torch.zeros((1, 2, 1))
    elif mutate == "activation_batch":
        geometry = _geometry(batch=2, height=2, width=2)
    elif mutate == "canvas":
        geometry = _geometry(batch=1, height=3, width=2)
        query = torch.zeros((1, 6, 1))

    with activation_context.activate(geometry):
        with pytest.raises(ValueError, match=message):
            patch(query, context)


def test_execution_and_invocation_values_reject_inconsistent_authorities() -> None:
    """Reject region count, strength count, and branch order drift at construction."""

    base = torch.zeros((1, 1, 1))
    contexts = _contexts(base, (torch.ones_like(base),))
    bank = _bank(torch.ones((2, 1, 1)))

    with pytest.raises(ValueError, match="context count must match"):
        AnimaRegionalAttentionExecution(contexts, bank, (1.0, 1.0))
    with pytest.raises(ValueError, match="strength count must match"):
        AnimaRegionalAttentionExecution(contexts, _bank(torch.ones((1, 1, 1))), ())
    with pytest.raises(RuntimeError, match="unavailable outside"):
        AnimaCrossAttentionInvocationContext().require_current()


def _execution(
    base: torch.Tensor,
    regions: tuple[torch.Tensor, ...],
    masks: torch.Tensor,
    *,
    latent_batch_size: int = 1,
) -> AnimaRegionalAttentionExecution:
    """Build one static execution plan around deterministic tensors."""

    return AnimaRegionalAttentionExecution(
        contexts=_contexts(base, regions, latent_batch_size=latent_batch_size),
        mask_bank=_bank(masks),
        region_strengths=(1.0,) * len(regions),
    )


def _contexts(
    base: torch.Tensor,
    regions: tuple[torch.Tensor, ...],
    *,
    latent_batch_size: int = 1,
) -> BatchedRegionalAttentionContexts:
    """Build valid contiguous chunk descriptors for one aligned context bank."""

    if int(base.shape[0]) % latent_batch_size != 0:
        raise AssertionError("Test context batch must divide into latent batches")
    chunk_count = int(base.shape[0]) // latent_batch_size
    return BatchedRegionalAttentionContexts(
        latent_batch_size=latent_batch_size,
        chunks=tuple(
            RegionalAttentionChunkBatch(
                chunk_index,
                (
                    RegionalAttentionBranch.POSITIVE
                    if chunk_index % 2 == 0
                    else RegionalAttentionBranch.NEGATIVE
                ),
                chunk_index * latent_batch_size,
                (chunk_index + 1) * latent_batch_size,
            )
            for chunk_index in range(chunk_count)
        ),
        base_context=base,
        regions=single_entry_regions(regions),
    )


def _bank(masks: torch.Tensor) -> RegionalMaskBank:
    """Build distinct planning and conditioning mask storage."""

    return RegionalMaskBank(
        planning_masks=masks.clone(),
        conditioning_masks=masks.clone(),
        canvas_width=int(masks.shape[-1]),
        canvas_height=int(masks.shape[-2]),
    )


def _geometry(
    *,
    batch: int,
    height: int,
    width: int,
    layout: SpatialBatchLayout | None = None,
) -> AnimaActivationGeometry:
    """Build a patch-size-one geometry for deterministic query-grid tests."""

    return AnimaActivationGeometry(
        input_batch_size=batch,
        activation_time=1,
        activation_height=height,
        activation_width=width,
        patch_temporal=1,
        patch_spatial=1,
        query_time=1,
        query_height=height,
        query_width=width,
        spatial_layout=layout,
    )


def _patcher(model: nn.Module) -> Any:
    """Create a real CPU Comfy MODEL patcher for injection lifecycle tests."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)
