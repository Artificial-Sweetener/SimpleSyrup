"""Prove regional contexts use installed Comfy conditioning processing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import comfy.conds
import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.raw_regional_attention import (
    build_raw_regional_attention_plan,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.attention_coupling.anima_context import (
    ANIMA_CONTEXT_FEATURE_WIDTH,
    ANIMA_CONTEXT_SEQUENCE_LENGTH,
    ANIMA_REGIONAL_CONTEXT_VALIDATOR,
)
from simple_syrup.runtime.attention_coupling.unet_context import (
    STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR,
)
from simple_syrup.runtime.comfy_conditioning_processing import (
    COMFY_REGIONAL_CONDITIONING_PROCESSOR,
)
from simple_syrup.services.attention_coupling_preparation_service import (
    ATTENTION_COUPLING_PREPARATION_SERVICE,
    AttentionCouplingPreparation,
)


class _RecordingAnimaModel:
    """Simulate Anima weighting and semantic padding in `extra_conds`."""

    def __init__(
        self,
        *,
        sequence_length: int = ANIMA_CONTEXT_SEQUENCE_LENGTH,
        feature_width: int = ANIMA_CONTEXT_FEATURE_WIDTH,
        include_cross_attention: bool = True,
    ) -> None:
        """Configure output shape and initialize the call record."""

        self.sequence_length = sequence_length
        self.feature_width = feature_width
        self.include_cross_attention = include_cross_attention
        self.model_sampling = _LinearModelSampling()
        self.calls: list[dict[str, Any]] = []
        self.outputs: list[torch.Tensor] = []

    def extra_conds(self, **kwargs: Any) -> dict[str, object]:
        """Apply a recognizable weight and zero-pad without repeating tokens."""

        self.calls.append(kwargs)
        if not self.include_cross_attention:
            return {}
        source = kwargs["cross_attn"]
        assert isinstance(source, torch.Tensor)
        weight = float(kwargs.get("semantic_weight", 1.0))
        output = torch.zeros(
            (int(source.shape[0]), self.sequence_length, self.feature_width),
            dtype=source.dtype,
            device=source.device,
        )
        copy_tokens = min(int(source.shape[1]), self.sequence_length)
        copy_width = min(int(source.shape[2]), self.feature_width)
        output[:, :copy_tokens, :copy_width] = (
            source[:, :copy_tokens, :copy_width] * weight
        )
        self.outputs.append(output)
        return {"c_crossattn": comfy.conds.CONDRegular(output)}


class _LinearModelSampling:
    """Expose deterministic descending sigma conversion like a diffusion model."""

    @staticmethod
    def percent_to_sigma(percent: float) -> float:
        """Convert one authored percentage into a recognizable sigma value."""

        return 100.0 * (1.0 - float(percent))


def test_processor_uses_comfy_conversion_and_model_post_adapter_contexts() -> None:
    """Retain weighted padded model outputs in exact positive/negative order."""

    model = _RecordingAnimaModel()
    positive_base = _conditioning(1.0, semantic_weight=2.0)
    positive_region = _conditioning(3.0, semantic_weight=4.0)
    negative_base = _conditioning(-1.0, semantic_weight=0.5)
    negative_region = _conditioning(-3.0, semantic_weight=0.25)
    preparation = _preparation(
        positive=(positive_base, positive_region),
        negative=(negative_base, negative_region),
    )
    noise = torch.zeros((2, 16, 8, 8))

    processed = COMFY_REGIONAL_CONDITIONING_PROCESSOR.process(
        preparation,
        model=SimpleNamespace(model=model),
        noise=noise,
        device=torch.device("cpu"),
        context_validator=ANIMA_REGIONAL_CONTEXT_VALIDATOR,
    )

    contexts = (
        processed.positive.base_context,
        processed.positive.regional_contexts[0],
        processed.negative.base_context,
        processed.negative.regional_contexts[0],
    )
    assert all(
        context.entries[0].cross_attention is output
        for context, output in zip(contexts, model.outputs, strict=True)
    )
    assert [context.entries[0].cross_attention.shape for context in contexts] == [
        (1, 512, 1024),
    ] * 4
    assert [
        context.entries[0].cross_attention[0, 0, 0].item() for context in contexts
    ] == [
        2.0,
        12.0,
        -0.5,
        -0.75,
    ]
    assert [call["prompt_type"] for call in model.calls] == [
        "positive",
        "positive",
        "negative",
        "negative",
    ]
    assert all(call["noise"] is noise for call in model.calls)
    assert processed.mask_bank is preparation.plan.mask_bank
    assert processed.lora_plan is preparation.plan.lora_plan
    source_positive_context = positive_base[0][0]
    assert isinstance(source_positive_context, torch.Tensor)
    assert source_positive_context.shape == (1, 3, 1024)


@pytest.mark.parametrize(
    ("sequence_length", "feature_width", "message"),
    [
        (511, 1024, "exactly 512 semantic tokens"),
        (513, 1024, "exactly 512 semantic tokens"),
        (512, 2048, "feature width 1024"),
    ],
)
def test_processor_rejects_incompatible_post_adapter_context_shape(
    sequence_length: int,
    feature_width: int,
    message: str,
) -> None:
    """Reject incompatible Anima context output without token repetition."""

    model = _RecordingAnimaModel(
        sequence_length=sequence_length,
        feature_width=feature_width,
    )

    with pytest.raises(ValueError, match=message):
        COMFY_REGIONAL_CONDITIONING_PROCESSOR.process(
            _preparation(),
            model=SimpleNamespace(model=model),
            noise=torch.zeros((1, 16, 8, 8)),
            device=torch.device("cpu"),
            context_validator=ANIMA_REGIONAL_CONTEXT_VALIDATOR,
        )


def test_processor_rejects_missing_model_cross_attention_output() -> None:
    """Require the exact model-produced `c_crossattn` condition."""

    model = _RecordingAnimaModel(include_cross_attention=False)

    with pytest.raises(TypeError, match="did not produce a tensor c_crossattn"):
        COMFY_REGIONAL_CONDITIONING_PROCESSOR.process(
            _preparation(),
            model=SimpleNamespace(model=model),
            noise=torch.zeros((1, 16, 8, 8)),
            device=torch.device("cpu"),
            context_validator=ANIMA_REGIONAL_CONTEXT_VALIDATOR,
        )


def test_processor_preserves_every_static_entry_and_comfy_strength() -> None:
    """Retain simultaneous entries in order without selecting the first."""

    model = _RecordingAnimaModel()
    raw = build_raw_regional_attention_plan(
        positive=_conditioning(1.0, strength=0.25) + _conditioning(2.0, strength=0.75),
        negative=_conditioning(-1.0),
        mask_bank=_mask_bank(),
    )
    preparation = ATTENTION_COUPLING_PREPARATION_SERVICE.prepare(raw)

    processed = COMFY_REGIONAL_CONDITIONING_PROCESSOR.process(
        preparation,
        model=SimpleNamespace(model=model),
        noise=torch.zeros((1, 16, 8, 8)),
        device=torch.device("cpu"),
        context_validator=ANIMA_REGIONAL_CONTEXT_VALIDATOR,
    )

    entries = processed.positive.base_context.entries
    assert [entry.entry_index for entry in entries] == [0, 1]
    assert [entry.strength for entry in entries] == [0.25, 0.75]
    assert [entry.cross_attention[0, 0, 0].item() for entry in entries] == [1.0, 2.0]
    assert len(model.calls) == 3


def test_processor_delegates_variable_standard_unet_context_geometry() -> None:
    """Preserve model-consumed SDXL geometry without applying Anima constants."""

    model = _RecordingAnimaModel(sequence_length=77, feature_width=2048)

    processed = COMFY_REGIONAL_CONDITIONING_PROCESSOR.process(
        _preparation(),
        model=SimpleNamespace(model=model),
        noise=torch.zeros((1, 4, 8, 8)),
        device=torch.device("cpu"),
        context_validator=STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR,
    )

    assert processed.positive.base_context.entries[0].cross_attention.shape == (
        1,
        77,
        2048,
    )


def test_processor_preserves_exact_comfy_uuid_and_schedule_ranges_per_branch() -> None:
    """Keep UUID identity and converted ranges for both conditioning banks."""

    positive = (
        _conditioning(1.0, start_percent=0.0, end_percent=1.0)
        + _conditioning(2.0, start_percent=0.0, end_percent=0.5)
        + _conditioning(3.0, start_percent=0.25, end_percent=0.75)
        + _conditioning(4.0, start_percent=0.5, end_percent=0.5)
    )
    negative = (
        _conditioning(-1.0, start_percent=0.5, end_percent=0.5)
        + _conditioning(-2.0, start_percent=0.25, end_percent=0.75)
        + _conditioning(-3.0, start_percent=0.5, end_percent=1.0)
        + _conditioning(-4.0, start_percent=0.0, end_percent=1.0)
    )
    model = _RecordingAnimaModel()

    processed = COMFY_REGIONAL_CONDITIONING_PROCESSOR.process(
        _preparation(positive=(positive,), negative=(negative,)),
        model=SimpleNamespace(model=model),
        noise=torch.zeros((1, 16, 8, 8)),
        device=torch.device("cpu"),
        context_validator=ANIMA_REGIONAL_CONTEXT_VALIDATOR,
    )

    positive_entries = processed.positive.base_context.entries
    negative_entries = processed.negative.base_context.entries
    assert [
        (
            entry.schedule.start_percent,
            entry.schedule.end_percent,
            entry.schedule.timestep_start,
            entry.schedule.timestep_end,
        )
        for entry in positive_entries
    ] == [
        (0.0, 1.0, 100.0, 0.0),
        (0.0, 0.5, 100.0, 50.0),
        (0.25, 0.75, 75.0, 25.0),
        (0.5, 0.5, 50.0, 50.0),
    ]
    assert [
        (
            entry.schedule.start_percent,
            entry.schedule.end_percent,
            entry.schedule.timestep_start,
            entry.schedule.timestep_end,
        )
        for entry in negative_entries
    ] == [
        (0.5, 0.5, 50.0, 50.0),
        (0.25, 0.75, 75.0, 25.0),
        (0.5, 1.0, 50.0, 0.0),
        (0.0, 1.0, 100.0, 0.0),
    ]
    entries = (*positive_entries, *negative_entries)
    assert all(isinstance(entry.uuid, UUID) for entry in entries)
    assert len({entry.uuid for entry in entries}) == len(entries)
    assert all(
        entry.uuid is call["uuid"]
        for entry, call in zip(entries, model.calls, strict=True)
    )


def test_processing_owner_contains_no_semantic_token_repeat_path() -> None:
    """Keep semantic-length rejection explicit in the production source."""

    source_path = (
        Path(__file__).resolve().parents[1]
        / "simple_syrup"
        / "runtime"
        / "attention_coupling"
        / "anima_context.py"
    )

    assert ".repeat(" not in source_path.read_text(encoding="utf-8")


def _preparation(
    *,
    positive: tuple[list[list[object]], ...] | None = None,
    negative: tuple[list[list[object]], ...] | None = None,
) -> AttentionCouplingPreparation:
    """Build and validate one raw static preparation."""

    positive_values = positive or (_conditioning(1.0),)
    negative_values = negative or (_conditioning(-1.0),)
    raw = build_raw_regional_attention_plan(
        positive=(
            ConditioningBatch(positive_values)
            if len(positive_values) > 1
            else positive_values[0]
        ),
        negative=(
            ConditioningBatch(negative_values)
            if len(negative_values) > 1
            else negative_values[0]
        ),
        mask_bank=_mask_bank(),
    )
    return ATTENTION_COUPLING_PREPARATION_SERVICE.prepare(raw)


def _conditioning(
    value: float,
    *,
    semantic_weight: float = 1.0,
    strength: float | None = None,
    start_percent: float | None = None,
    end_percent: float | None = None,
) -> list[list[object]]:
    """Build one small standard conditioning with model-consumed metadata."""

    metadata: dict[str, object] = {"semantic_weight": semantic_weight}
    if strength is not None:
        metadata["strength"] = strength
    if start_percent is not None:
        metadata["start_percent"] = start_percent
    if end_percent is not None:
        metadata["end_percent"] = end_percent
    return [
        [
            torch.full((1, 3, ANIMA_CONTEXT_FEATURE_WIDTH), value),
            metadata,
        ]
    ]


def _mask_bank() -> RegionalMaskBank:
    """Build one canonical single-region bank."""

    planning = torch.zeros((1, 4, 4))
    conditioning = torch.zeros_like(planning)
    return RegionalMaskBank(planning, conditioning, 4, 4)
