# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify focused common diagnostics for attention-only Anima execution."""

from __future__ import annotations

import logging
from uuid import uuid4

import pytest
import torch
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
from simple_syrup.runtime.regional_lora.anima_attention_diagnostics import (
    AnimaAttentionDiagnosticsBuilder,
)
from simple_syrup.runtime.regional_lora.anima_attention_diagnostics_wrapper import (
    ANIMA_ATTENTION_DIAGNOSTICS_WRAPPER_KEY,
    AnimaAttentionDiagnosticsDiffusionWrapper,
    anima_attention_diagnostics_wrapper_mutation,
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
from simple_syrup.runtime.regional_lora.anima_module_surface import (
    AnimaModuleSurface,
)


class _RecordingExecutor:
    """Record exact wrapper forwarding around one model owner."""

    def __init__(self, class_obj: object) -> None:
        """Retain the expected model owner."""

        self.class_obj = class_obj
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Record the call and return a stable downstream value."""

        self.calls.append((args, kwargs))
        return "prediction"


def test_attention_builder_reports_exact_common_model_call_evidence() -> None:
    """Expose geometry, sigma, UUID, and selected entry strengths without LoRA."""

    surface, attention = _authorities()
    identity = uuid4()
    snapshot = AnimaAttentionDiagnosticsBuilder(surface, attention).build(
        _geometry(),
        transformer_options={
            "cond_or_uncond": [0],
            "sigmas": torch.tensor([0.5]),
            "uuids": [identity],
        },
    )
    fields = snapshot.to_log_fields()

    assert snapshot.backend.endswith("._TestDiffusionModel")
    assert snapshot.sampling_sigma == 0.5
    assert snapshot.conditioning_uuids == (str(identity),)
    assert [entry.to_log_fields() for entry in snapshot.regional_entries] == [
        {
            "region_index": 0,
            "entry_index": 0,
            "strengths": [0.75],
            "active": True,
        }
    ]
    assert fields["estimated_work"] == {
        "cross_attention_branch_multiplier": 2.0,
        "cross_attention_formula": "base_plus_region_count",
        "denoiser_call_multiplier": 1.0,
    }
    assert "adapter_uses" not in fields


def test_attention_wrapper_emits_once_and_preserves_downstream_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log one common snapshot and forward every supplied argument unchanged."""

    surface, attention = _authorities()
    context = AnimaActivationContext()
    phase_context = AnimaCompositionPhaseContext()
    logger = logging.getLogger("test.anima.attention.diagnostics")
    wrapper = AnimaAttentionDiagnosticsDiffusionWrapper(
        surface,
        context,
        AnimaAttentionDiagnosticsBuilder(surface, attention),
        phase_context,
        logger=logger,
    )
    executor = _RecordingExecutor(surface.diffusion_model)
    options = {"cond_or_uncond": [0], "sigmas": torch.tensor([0.25])}

    with caplog.at_level(logging.INFO, logger=logger.name):
        with (
            context.activate(_geometry()),
            phase_context.activate(_composition_phase()),
        ):
            result = wrapper(executor, "model-input", transformer_options=options)

    records = [record for record in caplog.records if record.name == logger.name]
    assert result == "prediction"
    assert executor.calls == [(("model-input",), {"transformer_options": options})]
    assert len(records) == 1
    assert records[0].__dict__["operation"] == "anima_attention_coupling.execute"
    diagnostics = records[0].__dict__["regional_diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["model_call"] == {
        "sampling_sigma": 0.25,
        "conditioning_uuids": [],
    }
    assert diagnostics["composition_phase"] == {
        "stage": "composition",
        "denoising_progress": 0.05,
        "restrict_self_attention": False,
        "regional_lora_scale": 0.125,
    }


def test_attention_wrapper_validates_when_info_logging_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip payload construction while retaining exact dynamic validation."""

    surface, attention = _authorities()
    context = AnimaActivationContext()
    phase_context = AnimaCompositionPhaseContext()
    builder = AnimaAttentionDiagnosticsBuilder(surface, attention)
    logger = logging.getLogger("test.anima.attention.diagnostics.disabled")
    logger.setLevel(logging.WARNING)
    wrapper = AnimaAttentionDiagnosticsDiffusionWrapper(
        surface,
        context,
        builder,
        phase_context,
        logger=logger,
    )

    def unexpected_build(*args: object, **kwargs: object) -> object:
        """Reject unused payload construction."""

        del args, kwargs
        raise AssertionError("disabled diagnostics payload was built")

    monkeypatch.setattr(builder, "build", unexpected_build)
    executor = _RecordingExecutor(surface.diffusion_model)
    with context.activate(_geometry()):
        result = wrapper(
            executor,
            transformer_options={"cond_or_uncond": [0]},
        )

    assert result == "prediction"
    assert len(executor.calls) == 1


def test_attention_diagnostic_mutation_uses_a_distinct_collision_safe_key() -> None:
    """Keep attention-only logging separate from combined regional-LoRA logging."""

    surface, attention = _authorities()
    mutation = anima_attention_diagnostics_wrapper_mutation(
        surface,
        AnimaActivationContext(),
        AnimaAttentionDiagnosticsBuilder(surface, attention),
        AnimaCompositionPhaseContext(),
    )

    assert mutation.key == ANIMA_ATTENTION_DIAGNOSTICS_WRAPPER_KEY
    assert isinstance(mutation.wrapper, AnimaAttentionDiagnosticsDiffusionWrapper)


class _TestDiffusionModel(nn.Module):
    """Provide an allocation-free model identity for wrapper ownership."""


def _authorities() -> tuple[AnimaModuleSurface, AnimaRegionalAttentionExecution]:
    """Return one minimal valid Anima-shaped diagnostic authority set."""

    context = torch.zeros((1, 1, 2))
    contexts = BatchedRegionalAttentionContexts(
        1,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),),
        context,
        (
            BatchedRegionalAttentionRegion(
                0,
                (BatchedRegionalAttentionEntry(0, context.clone(), (0.75,)),),
            ),
        ),
    )
    masks = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    attention = AnimaRegionalAttentionExecution(
        contexts,
        RegionalMaskBank(masks, masks.clone(), 2, 2),
        (1.0,),
    )
    return AnimaModuleSurface(_TestDiffusionModel(), (), ()), attention


def _geometry() -> AnimaActivationGeometry:
    """Return one full-canvas single-image query geometry."""

    return AnimaActivationGeometry(1, 1, 2, 2, 1, 1, 1, 2, 2, None)


def _composition_phase() -> AnimaCompositionPhase:
    """Return stable scene-composition facts for wrapper diagnostics."""

    return AnimaCompositionPhase(
        AnimaCompositionStage.COMPOSITION,
        0.05,
        False,
        0.125,
    )
