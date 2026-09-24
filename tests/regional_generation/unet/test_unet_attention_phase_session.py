# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify standard-UNet attention phase publication and cleanup."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.attention_coupling.unet_attention_phase import (
    StandardUnetAttentionStage,
)
from simple_syrup.runtime.attention_coupling.unet_attention_phase_session import (
    StandardUnetAttentionPhaseSession,
)


def test_phase_session_restores_nested_state_after_success() -> None:
    """Restore the outer phase after a nested model call completes."""

    session = StandardUnetAttentionPhaseSession()
    with session.activate(_options(1.0)):
        outer = session.require_current()
        assert outer.stage is StandardUnetAttentionStage.COMPOSITION
        with session.activate(_options(0.5)):
            assert (
                session.require_current().stage
                is StandardUnetAttentionStage.SPECIALIZATION
            )
        assert session.require_current() is outer

    with pytest.raises(RuntimeError, match="inactive"):
        session.require_current()


def test_phase_session_clears_state_after_nested_failure() -> None:
    """Prevent a failed model call from leaking attention phase state."""

    session = StandardUnetAttentionPhaseSession()
    with pytest.raises(RuntimeError, match="model failure"):
        with session.activate(_options(0.1)):
            assert (
                session.require_current().stage
                is StandardUnetAttentionStage.CONSOLIDATION
            )
            raise RuntimeError("model failure")

    with pytest.raises(RuntimeError, match="inactive"):
        session.require_current()


def _options(sigma: float) -> dict[str, object]:
    """Return one complete normalized phase call."""

    return {
        "sample_sigmas": torch.linspace(1.0, 0.0, 11),
        "sigmas": torch.tensor([sigma]),
    }
