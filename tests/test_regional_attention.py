"""Prove immutable raw and processed Attention Coupling plan contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.conditioning_schedule import UNBOUNDED_CONDITIONING_SCHEDULE
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.raw_regional_attention import (
    RawRegionalAttentionBranch,
    RawRegionalAttentionContext,
    build_raw_regional_attention_plan,
)
from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_selection import (
    REGIONAL_ATTENTION_SELECTION_SERVICE,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank


def test_raw_plan_reuses_global_first_pairing_and_shared_authorities() -> None:
    """Preserve distinct branches, original identities, and canonical owners."""

    mask_bank = _mask_bank(3)
    lora_plan = _lora_plan(region_index=2)
    positive = tuple(object() for _ in range(3))
    negative = tuple(object() for _ in range(2))

    plan = build_raw_regional_attention_plan(
        positive=ConditioningBatch(positive),
        negative=ConditioningBatch(negative),
        mask_bank=mask_bank,
        lora_plan=lora_plan,
    )

    assert plan.positive.base_conditioning is positive[0]
    assert [context.conditioning for context in plan.positive.regional_contexts] == [
        positive[1],
        positive[2],
    ]
    assert plan.negative.base_conditioning is negative[0]
    assert [context.conditioning for context in plan.negative.regional_contexts] == [
        negative[1]
    ]
    assert [
        (context.conditioning_index, context.region_index)
        for context in plan.positive.regional_contexts
    ] == [(1, 0), (2, 1)]
    assert plan.mask_bank is mask_bank
    assert plan.lora_plan is lora_plan


def test_raw_plan_supports_base_only_and_all_authored_regions() -> None:
    """Represent zero active regional contexts and complete regional coverage."""

    mask_bank = _mask_bank(2)
    base_positive = object()
    full_negative = tuple(object() for _ in range(3))

    plan = build_raw_regional_attention_plan(
        positive=base_positive,
        negative=ConditioningBatch(full_negative),
        mask_bank=mask_bank,
    )

    assert plan.positive.base_conditioning is base_positive
    assert plan.positive.regional_contexts == ()
    assert [context.region_index for context in plan.negative.regional_contexts] == [
        0,
        1,
    ]


def test_raw_plan_reuses_pairing_policy_mismatch_diagnostic() -> None:
    """Reject excess contexts through the existing authoritative policy."""

    with pytest.raises(
        ValueError,
        match="positive conditioning contains 2 regional entries but only 1",
    ):
        build_raw_regional_attention_plan(
            positive=ConditioningBatch((object(), object(), object())),
            negative=object(),
            mask_bank=_mask_bank(1),
        )


def test_processed_plan_validates_model_ready_contexts_and_owner_references() -> None:
    """Retain separate processed branches without copying masks or LoRA state."""

    mask_bank = _mask_bank(2)
    lora_plan = _lora_plan(region_index=1)
    positive_base = _processed(0, None, value=1.0)
    negative_base = _processed(0, None, value=-1.0)
    positive_region = _processed(1, 0, value=2.0)
    negative_region = _processed(1, 0, value=-2.0)
    positive = ProcessedRegionalAttentionBranch(positive_base, (positive_region,))
    negative = ProcessedRegionalAttentionBranch(negative_base, (negative_region,))

    plan = ProcessedRegionalAttentionPlan(
        positive=positive,
        negative=negative,
        mask_bank=mask_bank,
        lora_plan=lora_plan,
    )

    assert plan.positive.base_context.entries[0].cross_attention[0, 0, 0].item() == 1.0
    assert plan.negative.base_context.entries[0].cross_attention[0, 0, 0].item() == -1.0
    assert plan.mask_bank is mask_bank
    assert plan.lora_plan is lora_plan


@pytest.mark.parametrize(
    ("tensor", "error", "message"),
    [
        (torch.ones((2, 3)), ValueError, "BxSxD"),
        (torch.ones((1, 0, 3)), ValueError, "dimensions must be positive"),
        (torch.ones((1, 2, 3), dtype=torch.int64), TypeError, "floating point"),
        (
            torch.tensor([[[float("nan")]]]),
            ValueError,
            "finite values",
        ),
    ],
)
def test_processed_context_rejects_malformed_model_ready_tensor(
    tensor: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    """Reject every malformed processed cross-attention tensor boundary."""

    with pytest.raises(error, match=message):
        _entry(0, tensor, 1.0)


def test_processed_entry_requires_explicit_comfy_identity_and_schedule() -> None:
    """Reject identity or schedule state not owned by the exact Comfy entry."""

    tensor = torch.ones((1, 2, 3))
    with pytest.raises(TypeError, match="UUID"):
        ProcessedRegionalAttentionEntry(
            0,
            object(),  # type: ignore[arg-type]
            UNBOUNDED_CONDITIONING_SCHEDULE,
            tensor,
            1.0,
        )
    with pytest.raises(TypeError, match="schedule"):
        ProcessedRegionalAttentionEntry(
            0,
            uuid4(),
            object(),  # type: ignore[arg-type]
            tensor,
            1.0,
        )


def test_contexts_and_branches_reject_noncanonical_indices() -> None:
    """Prevent processed or raw plans from silently reconstructing pairing order."""

    with pytest.raises(ValueError, match=r"region_index \+ 1"):
        RawRegionalAttentionContext(2, 0, object())
    with pytest.raises(ValueError, match="conditioning index 0"):
        ProcessedRegionalAttentionContext(
            1,
            None,
            (_entry(0, torch.ones((1, 1, 1)), 1.0),),
        )
    with pytest.raises(ValueError, match="canonical order"):
        RawRegionalAttentionBranch(
            object(),
            (RawRegionalAttentionContext(2, 1, object()),),
        )
    with pytest.raises(ValueError, match="canonical order"):
        ProcessedRegionalAttentionBranch(
            _processed(0, None, value=0.0),
            (_processed(2, 1, value=1.0),),
        )


def test_plan_rejects_out_of_bounds_lora_and_context_regions() -> None:
    """Bind all regional plan references to the canonical mask-bank count."""

    mask_bank = _mask_bank(1)
    with pytest.raises(ValueError, match="LoRA region indices exceed"):
        build_raw_regional_attention_plan(
            positive=object(),
            negative=object(),
            mask_bank=mask_bank,
            lora_plan=_lora_plan(region_index=1),
        )
    two_regions = ProcessedRegionalAttentionBranch(
        _processed(0, None, value=0.0),
        (
            _processed(1, 0, value=1.0),
            _processed(2, 1, value=2.0),
        ),
    )
    with pytest.raises(ValueError, match="positive branch exceeds"):
        ProcessedRegionalAttentionPlan(
            two_regions,
            ProcessedRegionalAttentionBranch(
                _processed(0, None, value=0.0),
                (),
            ),
            mask_bank,
            RegionalLoraPlan(()),
        )


def test_attention_plan_values_are_frozen() -> None:
    """Prevent callers from rewriting raw and processed plan ownership."""

    raw = build_raw_regional_attention_plan(
        positive=object(),
        negative=object(),
        mask_bank=_mask_bank(1),
    )
    processed = _processed(0, None, value=0.0)

    with pytest.raises(FrozenInstanceError):
        raw.positive = raw.negative  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        processed.region_index = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("selectors", "expected"),
    [
        ([], []),
        ([0], [RegionalAttentionBranch.POSITIVE]),
        ([1], [RegionalAttentionBranch.NEGATIVE]),
        ([0, 1], [RegionalAttentionBranch.POSITIVE, RegionalAttentionBranch.NEGATIVE]),
        ([1, 0], [RegionalAttentionBranch.NEGATIVE, RegionalAttentionBranch.POSITIVE]),
        (
            [0, 0, 1, 0, 1, 1],
            [
                RegionalAttentionBranch.POSITIVE,
                RegionalAttentionBranch.POSITIVE,
                RegionalAttentionBranch.NEGATIVE,
                RegionalAttentionBranch.POSITIVE,
                RegionalAttentionBranch.NEGATIVE,
                RegionalAttentionBranch.NEGATIVE,
            ],
        ),
    ],
)
def test_processed_chunk_selection_follows_exact_comfy_order(
    selectors: list[int],
    expected: list[RegionalAttentionBranch],
) -> None:
    """Support CFG-disabled, reversed, repeated, and interleaved chunk sequences."""

    plan = _processed_plan()

    chunks = REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
        plan,
        cond_or_uncond=selectors,
        conditioning_uuids=_uuids_for_selectors(plan, selectors),
        sigma=0.5,
    )

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(selectors)))
    assert [chunk.branch for chunk in chunks] == expected
    assert all(
        len(chunk.regional_entries) == plan.mask_bank.region_count for chunk in chunks
    )


@pytest.mark.parametrize(
    ("selectors", "error", "message"),
    [
        (object(), TypeError, "list or tuple"),
        ([True], TypeError, "integer 0 or 1"),
        (["0"], TypeError, "integer 0 or 1"),
        ([2], ValueError, "observed 2"),
        ([-1], ValueError, "observed -1"),
    ],
)
def test_processed_chunk_selection_rejects_invalid_selectors(
    selectors: object,
    error: type[Exception],
    message: str,
) -> None:
    """Fail closed instead of guessing conditional branch order."""

    with pytest.raises(error, match=message):
        REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
            _processed_plan(),
            cond_or_uncond=selectors,
            conditioning_uuids=(
                [uuid4()] * len(selectors) if isinstance(selectors, list) else []
            ),
            sigma=0.5,
        )


def test_processed_chunk_selection_preserves_all_active_base_entries() -> None:
    """Map repeated branch chunks to simultaneous base entries in declared order."""

    plan = _processed_plan()
    positive_base = ProcessedRegionalAttentionContext(
        0,
        None,
        (
            _entry(0, torch.full((1, 2, 3), 1.0), 0.25),
            _entry(1, torch.full((1, 2, 3), 5.0), 0.75),
        ),
    )
    plan = ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            positive_base,
            plan.positive.regional_contexts,
        ),
        plan.negative,
        plan.mask_bank,
        plan.lora_plan,
    )

    chunks = REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
        plan,
        cond_or_uncond=[0, 0, 1],
        conditioning_uuids=[
            positive_base.entries[0].uuid,
            positive_base.entries[1].uuid,
            plan.negative.base_context.entries[0].uuid,
        ],
        sigma=0.5,
    )

    assert [chunk.base_entry.entry_index for chunk in chunks] == [0, 1, 0]
    assert [chunk.base_entry.strength for chunk in chunks] == [0.25, 0.75, 1.0]
    with pytest.raises(ValueError, match="does not identify exactly one"):
        REGIONAL_ATTENTION_SELECTION_SERVICE.select_chunks(
            plan,
            cond_or_uncond=[0, 0, 0],
            conditioning_uuids=[
                positive_base.entries[0].uuid,
                positive_base.entries[1].uuid,
                uuid4(),
            ],
            sigma=0.5,
        )


def _processed(
    conditioning_index: int,
    region_index: int | None,
    *,
    value: float,
) -> ProcessedRegionalAttentionContext:
    """Build one small valid processed context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (_entry(0, torch.full((1, 2, 3), value), 1.0),),
    )


def _entry(
    entry_index: int,
    tensor: torch.Tensor,
    strength: float,
) -> ProcessedRegionalAttentionEntry:
    """Build one unscheduled processed entry with an explicit Comfy identity."""

    return ProcessedRegionalAttentionEntry(
        entry_index,
        uuid4(),
        UNBOUNDED_CONDITIONING_SCHEDULE,
        tensor,
        strength,
    )


def _processed_plan() -> ProcessedRegionalAttentionPlan:
    """Build distinguishable positive and negative processed banks."""

    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _processed(0, None, value=1.0),
            (_processed(1, 0, value=2.0),),
        ),
        ProcessedRegionalAttentionBranch(
            _processed(0, None, value=-1.0),
            (_processed(1, 0, value=-2.0),),
        ),
        _mask_bank(1),
        RegionalLoraPlan(()),
    )


def _uuids_for_selectors(
    plan: ProcessedRegionalAttentionPlan,
    selectors: list[int],
) -> list[object]:
    """Return matching static UUIDs for arbitrary characterized chunk order."""

    return [
        (
            plan.positive.base_context.entries[0].uuid
            if selector == 0
            else plan.negative.base_context.entries[0].uuid
        )
        for selector in selectors
    ]


def _mask_bank(region_count: int) -> RegionalMaskBank:
    """Build one canonical bank with distinct tensor storage."""

    planning = torch.zeros((region_count, 4, 5), dtype=torch.float32)
    conditioning = torch.zeros_like(planning)
    return RegionalMaskBank(planning, conditioning, 5, 4)


def _lora_plan(region_index: int) -> RegionalLoraPlan:
    """Build one immutable regional adapter use."""

    return RegionalLoraPlan(
        (
            RegionalLoraAdapterPlan(
                RegionalLoraAdapterIdentity("adapter.safetensors"),
                0,
                region_index,
                RegionalLoraBranch.POSITIVE,
                1.0,
                (RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
            ),
        )
    )
