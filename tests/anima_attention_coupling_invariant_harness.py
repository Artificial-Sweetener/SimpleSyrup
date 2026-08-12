"""Adapt Anima cross-attention to the shared invariant test contract."""

from __future__ import annotations

import torch
from attention_coupling_invariant_contract import (
    AttentionCouplingInvariantEntry,
    AttentionCouplingInvariantHarness,
    AttentionCouplingInvariantObservation,
    AttentionCouplingInvariantScenario,
)
from torch import nn

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
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
from simple_syrup.runtime.regional_lora.anima_cross_attention import (
    AnimaRegionalCrossAttentionPatch,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention_context import (
    AnimaCrossAttentionInvocationContext,
)


class _DeterministicAnimaAttention(nn.Module):
    """Evaluate one packed branch call using recognizable context values."""

    def __init__(self) -> None:
        """Initialize the installed child surface and empty observations."""

        super().__init__()
        self.q_proj = nn.Identity()
        self.q_norm = nn.Identity()
        self.k_proj = nn.Identity()
        self.k_norm = nn.Identity()
        self.v_proj = nn.Identity()
        self.output_proj = nn.Identity()
        self.output_dropout = nn.Identity()
        self.calls: list[torch.Tensor] = []

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        rope_emb: object,
        transformer_options: object,
    ) -> torch.Tensor:
        """Return each packed context value at every query token."""

        del rope_emb, transformer_options
        self.calls.append(context)
        return context.mean(dim=1, keepdim=True).expand(-1, query.shape[1], -1)


class _FullRegionalPhaseContext(AnimaCompositionPhaseContext):
    """Publish full regional influence for backend-neutral invariants."""

    def require_current(self) -> AnimaCompositionPhase:
        """Return one stable specialization phase."""

        return AnimaCompositionPhase(
            AnimaCompositionStage.SPECIALIZATION, 0.5, True, 1.0
        )


class AnimaAttentionCouplingInvariantHarness(AttentionCouplingInvariantHarness):
    """Execute shared invariants through production Anima attention owners."""

    def execute(
        self,
        scenario: AttentionCouplingInvariantScenario,
    ) -> AttentionCouplingInvariantObservation:
        """Run one complete Anima cross-attention call and capture shared facts."""

        contexts = _contexts(scenario)
        mask_bank = _mask_bank(scenario.masks)
        execution = AnimaRegionalAttentionExecution(
            contexts,
            mask_bank,
            scenario.region_strengths,
        )
        activation = AnimaActivationContext()
        original = _DeterministicAnimaAttention()
        patch = AnimaRegionalCrossAttentionPatch(
            original,
            execution,
            activation_context=activation,
            invocation_context=AnimaCrossAttentionInvocationContext(),
            phase_context=_FullRegionalPhaseContext(),
        )
        height, width = (int(size) for size in scenario.masks.shape[-2:])
        query = torch.zeros(len(scenario.base_values), height * width, 1)
        query_before = query.clone()
        contexts_before = _context_snapshots(contexts)
        masks_before = scenario.masks.clone()

        with activation.activate(
            AnimaActivationGeometry(
                input_batch_size=len(scenario.base_values),
                activation_time=1,
                activation_height=height,
                activation_width=width,
                patch_temporal=1,
                patch_spatial=1,
                query_time=1,
                query_height=height,
                query_width=width,
                spatial_layout=None,
            )
        ):
            output = patch(query, contexts.base_context)

        packed_context = original.calls[-1]
        return AttentionCouplingInvariantObservation(
            output=output,
            native_attention_calls=len(original.calls),
            packed_batch_size=int(packed_context.shape[0]),
            packed_context_values=tuple(
                float(value) for value in packed_context[:, 0, 0].tolist()
            ),
            query_unchanged=torch.equal(query, query_before),
            contexts_unchanged=_snapshots_equal(
                _context_snapshots(contexts),
                contexts_before,
            ),
            masks_unchanged=torch.equal(scenario.masks, masks_before),
        )

    def validate_masks(self, masks: torch.Tensor) -> None:
        """Construct one Anima execution to validate canonical masks."""

        scenario = AttentionCouplingInvariantScenario(
            masks=masks,
            base_values=(0.0,),
            regions=tuple(
                (AttentionCouplingInvariantEntry((1.0,), (1.0,)),)
                for _ in range(int(masks.shape[0]))
            ),
            region_strengths=(1.0,) * int(masks.shape[0]),
            expected=torch.empty(0),
        )
        AnimaRegionalAttentionExecution(
            _contexts(scenario),
            _mask_bank(masks),
            scenario.region_strengths,
        )

    def expected_output(
        self,
        scenario: AttentionCouplingInvariantScenario,
    ) -> torch.Tensor:
        """Return the explicit Anima global-preserving expectation."""

        return (
            scenario.anima_expected
            if scenario.anima_expected is not None
            else scenario.expected
        )


def _contexts(
    scenario: AttentionCouplingInvariantScenario,
) -> BatchedRegionalAttentionContexts:
    """Build aligned context values from one backend-neutral scenario."""

    batch_size = len(scenario.base_values)
    return BatchedRegionalAttentionContexts(
        latent_batch_size=batch_size,
        chunks=(
            RegionalAttentionChunkBatch(
                0,
                RegionalAttentionBranch.POSITIVE,
                0,
                batch_size,
            ),
        ),
        base_context=_context_tensor(scenario.base_values),
        regions=tuple(
            BatchedRegionalAttentionRegion(
                region_index,
                tuple(
                    BatchedRegionalAttentionEntry(
                        entry_index,
                        _context_tensor(entry.batch_values),
                        entry.strengths,
                    )
                    for entry_index, entry in enumerate(entries)
                ),
            )
            for region_index, entries in enumerate(scenario.regions)
        ),
    )


def _context_tensor(values: tuple[float, ...]) -> torch.Tensor:
    """Return Bx1x1 context storage for recognizable scalar values."""

    return torch.tensor(values).reshape(len(values), 1, 1)


def _mask_bank(masks: torch.Tensor) -> RegionalMaskBank:
    """Build one canonical planning/conditioning mask authority."""

    return RegionalMaskBank(
        masks.clone(),
        masks,
        int(masks.shape[-1]),
        int(masks.shape[-2]),
    )


def _context_snapshots(
    contexts: BatchedRegionalAttentionContexts,
) -> tuple[torch.Tensor, ...]:
    """Clone every context tensor without exposing backend branch structures."""

    return (
        contexts.base_context.clone(),
        *(
            entry.context.clone()
            for region in contexts.regions
            for entry in region.entries
        ),
    )


def _snapshots_equal(
    observed: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> bool:
    """Compare context snapshots without tensor truth-value coercion."""

    return len(observed) == len(expected) and all(
        torch.equal(actual, reference)
        for actual, reference in zip(observed, expected, strict=True)
    )
