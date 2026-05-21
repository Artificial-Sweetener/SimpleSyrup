# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Batched WD14 tagger runtime for tile prompt generation."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from ..shared.logging import get_logger
from .loaded_models import LoadedWD14Tagger
from .progress import NullProgressReporter, ProgressReporter

LOGGER = get_logger(__name__)
FloatArray = NDArray[np.float32]


class WD14Session(Protocol):
    """Minimal ONNX session surface used by the tagger."""

    def get_inputs(self) -> list[Any]:
        """Return model input metadata."""

    def get_outputs(self) -> list[Any]:
        """Return model output metadata."""

    def run(self, output_names: list[str], feeds: dict[str, FloatArray]) -> list[Any]:
        """Run inference and return model outputs."""


@dataclass(frozen=True)
class WD14TagFormattingControls:
    """Store validated WD14 tag formatting controls."""

    threshold: float = 0.35
    character_threshold: float = 1.0
    replace_underscore: bool = True
    trailing_comma: bool = False
    exclude_tags: str = ""

    def __post_init__(self) -> None:
        """Reject impossible threshold values."""

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")
        if not 0.0 <= self.character_threshold <= 1.0:
            raise ValueError("character_threshold must be between 0.0 and 1.0.")


@dataclass(frozen=True)
class WD14TagRecord:
    """Represent one WD14 tag CSV row."""

    name: str
    category: str


class WD14Tagger:
    """Tag ordered image crops through a loaded batched WD14 ONNX session."""

    def __init__(self, chunk_size: int | None = None) -> None:
        """Create the runtime with optional fixed-size batch chunks."""

        self._chunk_size = chunk_size

    def tag_images(
        self,
        loaded_tagger: LoadedWD14Tagger,
        images: Sequence[torch.Tensor],
        controls: WD14TagFormattingControls,
        progress: ProgressReporter | None = None,
    ) -> tuple[str, ...]:
        """Return one WD14 tag prompt for each input image in order."""

        if not images:
            return ()
        reporter = progress or NullProgressReporter()
        session = loaded_tagger.session
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        input_size = _input_size(input_meta)
        batch = cast(
            FloatArray,
            np.concatenate(
                tuple(_preprocess_image(image, input_size) for image in images),
                axis=0,
            ),
        )
        probabilities = self._run_batches(
            session,
            output_meta.name,
            input_meta.name,
            batch,
            reporter,
        )
        if probabilities.ndim != 2 or int(probabilities.shape[0]) != len(images):
            raise ValueError(
                "WD14 output shape must be [batch, tags]; "
                f"received {probabilities.shape} for {len(images)} images."
            )
        if int(probabilities.shape[1]) != len(loaded_tagger.tags):
            raise ValueError(
                "WD14 output tag count does not match selected_tags.csv; "
                f"received {probabilities.shape[1]} probabilities for "
                f"{len(loaded_tagger.tags)} tags."
            )
        return tuple(
            _format_tags(row, loaded_tagger.tags, controls) for row in probabilities
        )

    def _run_batches(
        self,
        session: WD14Session,
        output_name: str,
        input_name: str,
        batch: FloatArray,
        progress: ProgressReporter,
    ) -> FloatArray:
        """Run a full input batch, optionally in fixed-size chunks."""

        if self._chunk_size is None or self._chunk_size >= int(batch.shape[0]):
            output = session.run([output_name], {input_name: batch})[0]
            progress.update(int(batch.shape[0]))
            return cast(FloatArray, np.asarray(output, dtype=np.float32))
        chunks: list[FloatArray] = []
        for start in range(0, int(batch.shape[0]), self._chunk_size):
            chunk = batch[start : start + self._chunk_size]
            output = session.run([output_name], {input_name: chunk})[0]
            chunks.append(cast(FloatArray, np.asarray(output, dtype=np.float32)))
            progress.update(int(chunk.shape[0]))
        return cast(FloatArray, np.concatenate(tuple(chunks), axis=0))


def load_wd14_tags(csv_path: Path) -> tuple[WD14TagRecord, ...]:
    """Load WD14 tag records from selected_tags.csv."""

    records: list[WD14TagRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"WD14 tag CSV is empty: {csv_path}.") from exc
        for row in reader:
            if len(row) < 3:
                raise ValueError(f"WD14 tag CSV has an invalid row: {row}.")
            records.append(WD14TagRecord(name=row[1], category=row[2]))
    if not records:
        raise ValueError(f"WD14 tag CSV has no tag rows: {csv_path}.")
    return tuple(records)


def _input_size(input_meta: Any) -> int:
    """Return the square WD14 input size from ONNX metadata."""

    shape = cast(Sequence[Any], input_meta.shape)
    if len(shape) >= 3:
        try:
            height = int(shape[1])
            if height > 0:
                return height
        except (TypeError, ValueError):
            pass
    return 448


def _preprocess_image(image: torch.Tensor, input_size: int) -> FloatArray:
    """Resize and pad a BHWC image into one WD14 BGR batch row."""

    if image.ndim != 4 or int(image.shape[0]) != 1:
        raise ValueError("WD14 tile crops must be single-image BHWC tensors.")
    tensor = image.detach().cpu().float().clamp(0.0, 1.0)
    array = np.asarray(tensor[0].numpy() * 255.0, dtype=np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] > 3:
        array = array[:, :, :3]
    pil_image = Image.fromarray(array, mode="RGB")
    ratio = float(input_size) / max(pil_image.size)
    resized_size = (
        max(1, int(pil_image.size[0] * ratio)),
        max(1, int(pil_image.size[1] * ratio)),
    )
    resized = pil_image.resize(resized_size, Image.Resampling.LANCZOS)
    square = Image.new("RGB", (input_size, input_size), (255, 255, 255))
    square.paste(
        resized,
        ((input_size - resized_size[0]) // 2, (input_size - resized_size[1]) // 2),
    )
    processed = np.asarray(square, dtype=np.float32)[:, :, ::-1]
    return cast(FloatArray, np.expand_dims(processed, axis=0))


def _format_tags(
    probabilities: FloatArray,
    records: tuple[WD14TagRecord, ...],
    controls: WD14TagFormattingControls,
) -> str:
    """Format thresholded WD14 probabilities into one prompt string."""

    selected: list[str] = []
    excluded = _excluded_tags(controls)
    for probability, record in zip(probabilities, records, strict=True):
        threshold = _threshold_for_category(record.category, controls)
        if threshold is None or float(probability) <= threshold:
            continue
        name = _display_name(record.name, controls.replace_underscore)
        if name.lower() in excluded:
            continue
        selected.append(_escape_prompt_tag(name))
    if controls.trailing_comma:
        return "".join(f"{tag}, " for tag in selected)
    return ", ".join(selected)


def _threshold_for_category(
    category: str,
    controls: WD14TagFormattingControls,
) -> float | None:
    """Return the threshold for an emitted tag category."""

    if category == "0":
        return controls.threshold
    if category == "4":
        return controls.character_threshold
    return None


def _display_name(name: str, replace_underscore: bool) -> str:
    """Return the prompt-facing form of a WD14 tag."""

    if replace_underscore:
        return name.replace("_", " ")
    return name


def _excluded_tags(controls: WD14TagFormattingControls) -> set[str]:
    """Return normalized excluded tag names."""

    return {
        _display_name(tag.strip(), controls.replace_underscore).lower()
        for tag in controls.exclude_tags.split(",")
        if tag.strip()
    }


def _escape_prompt_tag(name: str) -> str:
    """Escape prompt syntax characters emitted by WD14 tags."""

    return name.replace("(", "\\(").replace(")", "\\)")
