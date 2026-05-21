# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the batched WD14 tagger runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from simple_syrup.runtime.loaded_models import LoadedWD14Tagger
from simple_syrup.runtime.wd14_tagger import (
    WD14TagFormattingControls,
    WD14Tagger,
    WD14TagRecord,
)

FloatArray = NDArray[np.float32]


def test_wd14_tagger_batches_images_and_preserves_order() -> None:
    """One inference receives the full batch and returns tags by image order."""

    progress = _ProgressRecorder()
    session = _FakeSession(
        np.array(
            [
                [0.0, 0.9, 0.1, 0.2],
                [0.0, 0.2, 0.8, 0.7],
            ],
            dtype=np.float32,
        )
    )
    loaded_tagger = _loaded_tagger(session)
    tagger = WD14Tagger()

    tags = tagger.tag_images(
        loaded_tagger,
        (_image(0.1), _image(0.9)),
        _controls(threshold=0.5, character_threshold=0.5),
        progress=progress,
    )

    assert tags == ("blue hair", "smile, cirno")
    assert session.batch_shapes == [(2, 448, 448, 3)]
    assert progress.updates == [2]


def test_wd14_tagger_chunks_without_reordering() -> None:
    """Internal chunks are concatenated in original tile order."""

    progress = _ProgressRecorder()
    session = _FakeSession(
        np.array(
            [
                [0.0, 0.9, 0.1, 0.2],
                [0.0, 0.2, 0.8, 0.7],
                [0.0, 0.6, 0.6, 0.1],
            ],
            dtype=np.float32,
        )
    )
    loaded_tagger = _loaded_tagger(session)
    tagger = WD14Tagger(chunk_size=2)

    tags = tagger.tag_images(
        loaded_tagger,
        (_image(0.1), _image(0.9), _image(0.5)),
        _controls(threshold=0.5, character_threshold=0.5),
        progress=progress,
    )

    assert tags == ("blue hair", "smile, cirno", "blue hair, smile")
    assert session.batch_shapes == [(2, 448, 448, 3), (1, 448, 448, 3)]
    assert progress.updates == [2, 1]


def test_wd14_tagger_applies_formatting_and_exclusions() -> None:
    """Tag formatting follows WD14 controls."""

    session = _FakeSession(np.array([[0.0, 0.9, 0.8, 0.0]], dtype=np.float32))
    loaded_tagger = _loaded_tagger(session)
    tagger = WD14Tagger()

    tags = tagger.tag_images(
        loaded_tagger,
        (_image(0.1),),
        _controls(
            threshold=0.5,
            character_threshold=0.5,
            replace_underscore=False,
            trailing_comma=True,
            exclude_tags="smile",
        ),
    )

    assert tags == ("blue_hair, ",)


def test_wd14_tagger_rejects_invalid_output_shape() -> None:
    """Bad ONNX output shapes fail before tags can be misaligned."""

    loaded_tagger = _loaded_tagger(_FakeSession(np.array([0.1, 0.2], dtype=np.float32)))
    tagger = WD14Tagger()

    with pytest.raises(ValueError, match="WD14 output shape"):
        tagger.tag_images(loaded_tagger, (_image(0.1),), _controls())


def test_wd14_tagger_rejects_output_tag_count_mismatch() -> None:
    """Model outputs must match the loaded selected tag records."""

    loaded_tagger = _loaded_tagger(
        _FakeSession(np.array([[0.9, 0.2]], dtype=np.float32))
    )

    with pytest.raises(ValueError, match="selected_tags.csv"):
        WD14Tagger().tag_images(loaded_tagger, (_image(0.1),), _controls())


def test_wd14_tagger_returns_empty_tuple_for_no_images() -> None:
    """Empty image batches return no tags without touching the loaded session."""

    loaded_tagger = _loaded_tagger(
        _FakeSession(np.array([[0.9, 0.2, 0.1, 0.0]], dtype=np.float32))
    )

    assert WD14Tagger().tag_images(loaded_tagger, (), _controls()) == ()


class _FakeSession:
    """Fake ONNX session with visible batched inputs."""

    def __init__(self, outputs: FloatArray) -> None:
        """Store fixed output rows."""

        self.outputs = outputs
        self.offset = 0
        self.batch_shapes: list[tuple[int, ...]] = []

    def get_inputs(self) -> list[Any]:
        """Return one fake model input."""

        return [_FakeIO("input", [None, 448, 448, 3])]

    def get_outputs(self) -> list[Any]:
        """Return one fake model output."""

        return [_FakeIO("output", [None, 4])]

    def run(
        self, output_names: list[str], feeds: dict[str, FloatArray]
    ) -> list[FloatArray]:
        """Return rows matching the provided batch size."""

        _ = output_names
        batch = feeds["input"]
        self.batch_shapes.append(tuple(batch.shape))
        count = int(batch.shape[0])
        rows = self.outputs[self.offset : self.offset + count]
        self.offset += count
        return [rows]


class _FakeIO:
    """Simple ONNX IO metadata object."""

    def __init__(self, name: str, shape: list[int | None]) -> None:
        """Store IO metadata."""

        self.name = name
        self.shape = shape


class _ProgressRecorder:
    """Record WD14 progress updates."""

    def __init__(self) -> None:
        """Initialize captured updates."""

        self.updates: list[int] = []

    def update(self, value: int) -> None:
        """Record one progress update."""

        self.updates.append(value)


def _loaded_tagger(session: _FakeSession) -> LoadedWD14Tagger:
    """Create loaded WD14 tagger metadata around a fake session."""

    return LoadedWD14Tagger(
        model_id="wd-eva02-large-tagger-v3",
        source="test",
        onnx_path=Path("wd-eva02-large-tagger-v3.onnx"),
        csv_path=Path("wd-eva02-large-tagger-v3.csv"),
        providers=("CPUExecutionProvider",),
        session=session,
        tags=(
            WD14TagRecord("rating:safe", "9"),
            WD14TagRecord("blue_hair", "0"),
            WD14TagRecord("smile", "0"),
            WD14TagRecord("cirno", "4"),
        ),
    )


def _controls(
    *,
    threshold: float = 0.35,
    character_threshold: float = 1.0,
    replace_underscore: bool = True,
    trailing_comma: bool = False,
    exclude_tags: str = "",
) -> WD14TagFormattingControls:
    """Return WD14 controls for runtime tests."""

    return WD14TagFormattingControls(
        threshold=threshold,
        character_threshold=character_threshold,
        replace_underscore=replace_underscore,
        trailing_comma=trailing_comma,
        exclude_tags=exclude_tags,
    )


def _image(value: float) -> torch.Tensor:
    """Return a small single-image BHWC tensor."""

    return torch.full((1, 8, 8, 3), value, dtype=torch.float32)
