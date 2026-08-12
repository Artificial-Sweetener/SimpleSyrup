"""Verify exact standard-UNet attn2 branch execution and spatial blending."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_attention_weights import (
    RegionalAttentionWeightingPolicy,
)
from simple_syrup.domain.regional_conditioning_output import (
    REGIONAL_CONDITIONING_OUTPUT_COMBINER,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_execution import (
    UnetAttn2Execution,
)


def test_unet_attn2_execution_matches_explicit_multi_entry_reference() -> None:
    """Match complete branch attention, entry combination, and spatial blending."""

    contexts = _contexts()
    masks = torch.tensor(
        [
            [[1.0, 0.25], [0.0, 0.0]],
            [[0.0, 0.25], [1.0, 0.0]],
        ]
    )
    execution = UnetAttn2Execution(contexts, masks, (1.0, 1.0))
    query = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    base_context = contexts.base_context
    query_before = query.clone()
    context_before = base_context.clone()

    expanded = execution.expand(query, base_context, base_context)
    packed_output = expanded.query + expanded.context.mean(dim=1, keepdim=True)
    actual = execution.blend(packed_output)

    base_output = query + base_context.mean(dim=1, keepdim=True)
    region_outputs: list[torch.Tensor] = []
    for region in contexts.regions:
        entry_outputs = tuple(
            query + entry.context.mean(dim=1, keepdim=True) for entry in region.entries
        )
        region_outputs.append(
            REGIONAL_CONDITIONING_OUTPUT_COMBINER.combine(
                entry_outputs,
                strengths=tuple(entry.strengths for entry in region.entries),
            )
        )
    weights = RegionalAttentionWeightingPolicy().weights(
        masks,
        region_strengths=(1.0, 1.0),
    )
    expected = RegionalAttentionWeightingPolicy().blend(
        weights=weights,
        base_output=base_output,
        regional_outputs=torch.stack(region_outputs),
    )

    assert torch.allclose(actual, expected)
    assert torch.equal(query, query_before)
    assert torch.equal(base_context, context_before)
    assert execution.branches.packed_batch_size == 6


def test_unet_attn2_execution_prunes_zero_strength_and_zero_coverage_rows() -> None:
    """Avoid native attention rows that cannot contribute to the final output."""

    contexts = _contexts()
    masks = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]
    )
    execution = UnetAttn2Execution(contexts, masks, (1.0, 0.0))

    assert tuple(
        (segment.key.region_index, segment.key.entry_index)
        for segment in execution.branches.segments
    ) == ((None, None),)
    assert execution.branches.packed_batch_size == 2


@pytest.mark.parametrize(
    ("masks", "region_strengths", "expected"),
    [
        (torch.tensor([[[0.0]], [[0.0]]]), (1.0, 1.0), 10.0),
        (torch.tensor([[[1.0]], [[0.0]]]), (1.0, 1.0), 30.0),
        (torch.tensor([[[0.25]], [[0.0]]]), (1.0, 1.0), 15.0),
        (torch.tensor([[[0.75]], [[0.75]]]), (1.0, 1.0), 50.0),
        (torch.tensor([[[1.0]], [[1.0]]]), (0.0, 0.0), 10.0),
    ],
)
def test_unet_attn2_execution_matches_base_region_and_overlap_closed_forms(
    masks: torch.Tensor,
    region_strengths: tuple[float, ...],
    expected: float,
) -> None:
    """Match all-base, all-region, uncovered, overlap, and zero-strength cases."""

    contexts = _single_entry_contexts()
    execution = UnetAttn2Execution(contexts, masks, region_strengths)
    query = torch.zeros(1, 1, 1)
    expanded = execution.expand(query, contexts.base_context, contexts.base_context)
    packed_output = expanded.context.clone()

    actual = execution.blend(packed_output)

    assert torch.allclose(actual, torch.tensor([[[expected]]]))


@pytest.mark.parametrize(
    ("query_masks", "message"),
    [
        (torch.ones(2, 1, 2), "batch must match"),
        (torch.ones(1, 2, 2), "region count must match"),
        (torch.full((2, 2, 2), float("nan")), "finite"),
    ],
)
def test_unet_attn2_execution_rejects_misaligned_query_masks(
    query_masks: torch.Tensor,
    message: str,
) -> None:
    """Fail before branch construction on malformed preprojected geometry."""

    with pytest.raises(ValueError, match=message):
        UnetAttn2Execution(_contexts(), query_masks, (1.0,) * int(query_masks.shape[0]))


def _contexts() -> BatchedRegionalAttentionContexts:
    """Return two source rows with two regions and one multi-entry region."""

    base = torch.tensor([[[10.0]], [[20.0]]])
    return BatchedRegionalAttentionContexts(
        latent_batch_size=2,
        chunks=(
            RegionalAttentionChunkBatch(
                0,
                RegionalAttentionBranch.POSITIVE,
                0,
                2,
            ),
        ),
        base_context=base,
        regions=(
            BatchedRegionalAttentionRegion(
                0,
                (
                    BatchedRegionalAttentionEntry(
                        0,
                        torch.tensor([[[30.0]], [[40.0]]]),
                        (1.0, 1.0),
                    ),
                    BatchedRegionalAttentionEntry(
                        1,
                        torch.tensor([[[50.0]], [[60.0]]]),
                        (0.5, 0.0),
                    ),
                ),
            ),
            BatchedRegionalAttentionRegion(
                1,
                (
                    BatchedRegionalAttentionEntry(
                        0,
                        torch.tensor([[[70.0]], [[80.0]]]),
                        (1.0, 1.0),
                    ),
                ),
            ),
        ),
    )


def _single_entry_contexts() -> BatchedRegionalAttentionContexts:
    """Return one source row with recognizable base and regional outputs."""

    return BatchedRegionalAttentionContexts(
        latent_batch_size=1,
        chunks=(
            RegionalAttentionChunkBatch(
                0,
                RegionalAttentionBranch.POSITIVE,
                0,
                1,
            ),
        ),
        base_context=torch.tensor([[[10.0]]]),
        regions=(
            BatchedRegionalAttentionRegion(
                0,
                (
                    BatchedRegionalAttentionEntry(
                        0,
                        torch.tensor([[[30.0]]]),
                        (1.0,),
                    ),
                ),
            ),
            BatchedRegionalAttentionRegion(
                1,
                (
                    BatchedRegionalAttentionEntry(
                        0,
                        torch.tensor([[[70.0]]]),
                        (1.0,),
                    ),
                ),
            ),
        ),
    )
