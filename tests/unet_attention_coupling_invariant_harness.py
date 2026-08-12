"""Adapt standard-UNet attn2 execution to the shared invariant contract."""

from __future__ import annotations

import torch
from attention_coupling_invariant_contract import (
    AttentionCouplingInvariantEntry,
    AttentionCouplingInvariantHarness,
    AttentionCouplingInvariantObservation,
    AttentionCouplingInvariantScenario,
)

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_execution import (
    UnetAttn2Execution,
)


class _DeterministicUnetAttention:
    """Evaluate one packed attn2 batch with deterministic context values."""

    def __init__(self) -> None:
        """Initialize an empty native attention call count."""

        self.calls = 0

    def __call__(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Return each packed context value at every query token."""

        del value
        self.calls += 1
        return context.mean(dim=1, keepdim=True).expand(-1, query.shape[1], -1)


class UnetAttentionCouplingInvariantHarness(AttentionCouplingInvariantHarness):
    """Execute shared invariants through production standard-UNet owners."""

    def execute(
        self,
        scenario: AttentionCouplingInvariantScenario,
    ) -> AttentionCouplingInvariantObservation:
        """Run one complete UNet attn2 execution and capture shared facts."""

        contexts = _contexts(scenario)
        query_masks = _query_masks(scenario.masks, len(scenario.base_values))
        execution = UnetAttn2Execution(
            contexts,
            query_masks,
            scenario.region_strengths,
        )
        query = torch.zeros(
            len(scenario.base_values),
            int(query_masks.shape[-1]),
            1,
        )
        query_before = query.clone()
        contexts_before = _context_snapshots(contexts)
        masks_before = scenario.masks.clone()
        expanded = execution.expand(query, contexts.base_context, contexts.base_context)
        native_attention = _DeterministicUnetAttention()
        packed_output = native_attention(
            expanded.query,
            expanded.context,
            expanded.value,
        )

        return AttentionCouplingInvariantObservation(
            output=execution.blend(packed_output),
            native_attention_calls=native_attention.calls,
            packed_batch_size=int(expanded.context.shape[0]),
            packed_context_values=tuple(
                float(value) for value in expanded.context[:, 0, 0].tolist()
            ),
            query_unchanged=torch.equal(query, query_before),
            contexts_unchanged=_snapshots_equal(
                _context_snapshots(contexts),
                contexts_before,
            ),
            masks_unchanged=torch.equal(scenario.masks, masks_before),
        )

    def validate_masks(self, masks: torch.Tensor) -> None:
        """Construct one UNet execution to validate canonical masks."""

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
        UnetAttn2Execution(
            _contexts(scenario),
            _query_masks(masks, 1),
            scenario.region_strengths,
        )

    def expected_output(
        self,
        scenario: AttentionCouplingInvariantScenario,
    ) -> torch.Tensor:
        """Return the shared Comfy-complement expectation."""

        return scenario.expected


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


def _query_masks(masks: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Expand canonical full-canvas masks to the UNet R/B/Q contract."""

    return masks.flatten(start_dim=1).unsqueeze(1).expand(-1, batch_size, -1)


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
