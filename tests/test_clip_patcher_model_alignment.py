# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify derived CLIP and patcher model coherence."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from simple_syrup.runtime.clip_patcher_model_alignment import (
    align_clip_text_encoder_with_patcher,
)


@dataclass
class _Patcher:
    """Expose one patcher-owned text encoder."""

    model: object


@dataclass
class _Clip:
    """Expose the host CLIP surface required by the alignment owner."""

    patcher: _Patcher
    cond_stage_model: object


def test_alignment_replaces_a_stale_source_encoder() -> None:
    """Execute the independently reloaded model owned by a static patcher."""

    patcher_model = object()
    clip = _Clip(_Patcher(patcher_model), object())

    align_clip_text_encoder_with_patcher(clip)

    assert clip.cond_stage_model is patcher_model


def test_alignment_preserves_an_already_coherent_encoder() -> None:
    """Leave the identity fast path unchanged."""

    encoder = object()
    clip = _Clip(_Patcher(encoder), encoder)

    align_clip_text_encoder_with_patcher(clip)

    assert clip.cond_stage_model is encoder


@pytest.mark.parametrize(
    ("clip", "message"),
    (
        (object(), "does not expose a patcher"),
        (
            type("Clip", (), {"patcher": object(), "cond_stage_model": object()})(),
            "patcher does not expose a model",
        ),
        (_Clip(_Patcher(object()), None), "does not expose a text encoder"),
    ),
)
def test_alignment_rejects_missing_host_surfaces(
    clip: object,
    message: str,
) -> None:
    """Fail closed when Comfy does not expose the required relationship."""

    with pytest.raises(TypeError, match=message):
        align_clip_text_encoder_with_patcher(clip)


def test_alignment_rejects_a_read_only_text_encoder_surface() -> None:
    """Report host values that cannot retain the patcher-owned encoder."""

    patcher_model = object()

    class ReadOnlyClip:
        """Expose a derived text encoder without a writable host boundary."""

        patcher = _Patcher(patcher_model)

        @property
        def cond_stage_model(self) -> object:
            """Return a stale source encoder."""

            return self._source_encoder

        def __init__(self) -> None:
            """Create the stale encoder identity."""

            self._source_encoder = object()

    with pytest.raises(TypeError, match="cannot be aligned"):
        align_clip_text_encoder_with_patcher(ReadOnlyClip())
