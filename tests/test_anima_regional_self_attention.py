# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove scheduled mask-authoritative Anima image self-attention execution."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from regional_attention_test_values import single_entry_regions
from torch import nn

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationContext,
    AnimaActivationGeometry,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_composition_phase import (
    AnimaCompositionPhase,
    AnimaCompositionStage,
)
from simple_syrup.runtime.regional_lora.anima_composition_phase_context import (
    AnimaCompositionPhaseContext,
)
from simple_syrup.runtime.regional_lora.anima_query_activity import (
    AnimaRegionalQueryActivityContext,
)
from simple_syrup.runtime.regional_lora.anima_query_mask_context import (
    AnimaQueryMaskContext,
)
from simple_syrup.runtime.regional_lora.anima_self_attention import (
    AnimaRegionalSelfAttentionPatch,
)
from simple_syrup.runtime.regional_lora.anima_self_attention_partition_cache import (
    AnimaSelfAttentionPartitionCache,
)
from simple_syrup.runtime.regional_self_attention_coherence import (
    RegionalSelfAttentionCoherencePolicy,
)


class _DeterministicSelfAttention(nn.Module):
    """Expose the installed attention surface with deterministic Q/K/V."""

    def __init__(self) -> None:
        """Install identity children and an empty global-call log."""

        super().__init__()
        for name in (
            "q_proj",
            "q_norm",
            "k_proj",
            "k_norm",
            "v_proj",
            "v_norm",
            "output_proj",
            "output_dropout",
        ):
            setattr(self, name, nn.Identity())
        self.n_heads = 1
        self.global_calls = 0
        self.attn_op = self._default_attention

    @staticmethod
    def _default_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        transformer_options: object,
    ) -> torch.Tensor:
        """Return flattened queries for deterministic compact-path tests."""

        del k, v, transformer_options
        return q.reshape(q.shape[0], q.shape[1], -1)

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        transformer_options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one-head projections without changing token values."""

        del context, rope_emb, transformer_options
        projected = x.unsqueeze(2)
        return projected, projected, projected

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Record the late global path and return a recognizable value."""

        del context, rope_emb, transformer_options
        self.global_calls += 1
        return x + 10.0


def test_restricted_path_executes_compact_owner_partitions() -> None:
    """Use one Q/K/V evaluation and compact exact owner-local attention."""

    calls: list[tuple[torch.Size, torch.Size, torch.Size, object]] = []

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        transformer_options: object,
    ) -> torch.Tensor:
        """Capture compact installed attention calls and preserve query values."""

        calls.append((q.shape, k.shape, v.shape, transformer_options))
        return q.reshape(q.shape[0], q.shape[1], -1)

    original = _DeterministicSelfAttention()
    original.attn_op = attention
    patch, activation, phase, _ = _patch(original=original)
    query = torch.arange(4.0).reshape(1, 4, 1)

    with activation.activate(_geometry()):
        with phase.activate(_phase(restricted=True)):
            output = patch(query, transformer_options={"preserved": True})

    assert torch.equal(output, query)
    assert original.global_calls == 0
    assert calls == [
        (
            torch.Size([2, 2, 1, 1]),
            torch.Size([2, 3, 1, 1]),
            torch.Size([2, 3, 1, 1]),
            {"preserved": True},
        ),
        (
            torch.Size([1, 4, 1, 1]),
            torch.Size([1, 4, 1, 1]),
            torch.Size([1, 4, 1, 1]),
            {"preserved": True},
        ),
    ]


def test_reopened_path_delegates_exact_installed_attention() -> None:
    """Restore the original global image-attention path after the boundary."""

    patch, activation, phase, original = _patch()
    query = torch.zeros((1, 4, 1))

    with activation.activate(_geometry()):
        with phase.activate(_phase(restricted=False)):
            output = patch(query, transformer_options={})

    assert torch.equal(output, torch.full_like(query, 10.0))
    assert original.global_calls == 0


def test_restricted_path_rejects_query_geometry_drift() -> None:
    """Fail closed before applying a mask to misaligned image tokens."""

    patch, activation, phase, _ = _patch()

    with activation.activate(_geometry()):
        with phase.activate(_phase(restricted=True)):
            with pytest.raises(ValueError, match="match active B/Q geometry"):
                patch(torch.zeros((1, 3, 1)))


def _patch(
    *,
    original: _DeterministicSelfAttention | None = None,
) -> tuple[
    AnimaRegionalSelfAttentionPatch,
    AnimaActivationContext,
    AnimaCompositionPhaseContext,
    _DeterministicSelfAttention,
]:
    """Build focused attention collaborators over a hard left/right split."""

    masks = torch.tensor(
        [
            [[1.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 1.0]],
        ]
    ).reshape(2, 1, 1, 4)
    contexts = BatchedRegionalAttentionContexts(
        1,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),),
        torch.zeros((1, 1, 1)),
        single_entry_regions((torch.zeros((1, 1, 1)),) * 2),
    )
    execution = AnimaRegionalAttentionExecution(
        contexts,
        RegionalMaskBank(masks[:, 0].clone(), masks[:, 0].clone(), 4, 1),
        (1.0, 1.0),
    )
    activation = AnimaActivationContext()
    phase = AnimaCompositionPhaseContext()
    original = original or _DeterministicSelfAttention()
    return (
        AnimaRegionalSelfAttentionPatch(
            original,
            execution,
            activation_context=activation,
            phase_context=phase,
            query_activity=AnimaRegionalQueryActivityContext(AnimaQueryMaskContext()),
            partition_cache=AnimaSelfAttentionPartitionCache(
                coherence=RegionalSelfAttentionCoherencePolicy(radius=0)
            ),
        ),
        activation,
        phase,
        original,
    )


def _geometry() -> AnimaActivationGeometry:
    """Return the exact one-row four-token activation geometry."""

    return AnimaActivationGeometry(1, 1, 1, 4, 1, 1, 1, 1, 4, None)


def _phase(*, restricted: bool) -> AnimaCompositionPhase:
    """Return one focused specialization or consolidation phase."""

    return AnimaCompositionPhase(
        (
            AnimaCompositionStage.SPECIALIZATION
            if restricted
            else AnimaCompositionStage.CONSOLIDATION
        ),
        0.5,
        restricted,
        1.0,
    )
