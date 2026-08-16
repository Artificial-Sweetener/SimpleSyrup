# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify standard-UNet regional-operation phase boundaries."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.runtime.attention_coupling.unet_attention_phase import (
    StandardUnetAttentionStage,
)
from simple_syrup.runtime.attention_coupling.unet_attention_phase_schedule import (
    StandardUnetAttentionPhaseSchedule,
)


def test_phase_resolves_composition_specialization_and_consolidation() -> None:
    """Publish exact operation stages throughout one trajectory."""

    schedule = StandardUnetAttentionPhaseSchedule()
    sigmas = torch.linspace(10.0, 0.0, 101)

    initial = schedule.resolve(_options(sigmas, 10.0))
    specialization = schedule.resolve(_options(sigmas, 9.0))
    consolidation = schedule.resolve(_options(sigmas, 3.5))
    terminal = schedule.resolve(_options(sigmas, 0.0))

    assert initial.stage is StandardUnetAttentionStage.COMPOSITION
    assert initial.stage_progress == pytest.approx(0.0)
    assert specialization.stage is StandardUnetAttentionStage.SPECIALIZATION
    assert specialization.stage_progress == pytest.approx(0.0)
    assert consolidation.stage is StandardUnetAttentionStage.CONSOLIDATION
    assert consolidation.stage_progress == pytest.approx(0.0)
    assert terminal.stage is StandardUnetAttentionStage.CONSOLIDATION
    assert terminal.stage_progress == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("composition", "specialization", "error"),
    [
        (0.0, 0.5, ValueError),
        (0.6, 0.4, ValueError),
        (True, 0.5, TypeError),
        (0.1, float("nan"), TypeError),
    ],
)
def test_phase_schedule_rejects_invalid_stage_configuration(
    composition: Any,
    specialization: Any,
    error: type[Exception],
) -> None:
    """Fail closed before accepting invalid phase boundaries."""

    with pytest.raises(error):
        StandardUnetAttentionPhaseSchedule(
            composition_fraction=composition,
            specialization_fraction=specialization,
        )


def _options(sample_sigmas: torch.Tensor, sigma: float) -> dict[str, object]:
    """Return exact uniform Comfy phase metadata."""

    return {"sample_sigmas": sample_sigmas, "sigmas": torch.tensor([sigma, sigma])}
