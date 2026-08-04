# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI filesystem adapter for one authored image file."""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path

import torch
from PIL import Image


class ImageFileLoader:
    """Load one static image through ComfyUI's native image implementation."""

    def available_files(self) -> tuple[str, ...]:
        """Return choices declared by ComfyUI's native image loader."""

        declaration = import_module("nodes").LoadImage.INPUT_TYPES()
        image_input = declaration["required"]["image"]
        choices = image_input[0]
        if not isinstance(choices, (list, tuple)):
            raise TypeError("ComfyUI LoadImage returned invalid image choices.")
        return tuple(str(value) for value in choices)

    def validate(self, annotated_path: str) -> None:
        """Validate one path with ComfyUI's native image-loader contract."""

        result = import_module("nodes").LoadImage.VALIDATE_INPUTS(annotated_path)
        if result is not True:
            raise ValueError(str(result))

    def load(self, annotated_path: str) -> torch.Tensor:
        """Return exactly one BHWC image from a validated annotated path."""

        self.validate(annotated_path)
        path = self._resolve_path(annotated_path)
        with Image.open(path) as image:
            frame_count = int(getattr(image, "n_frames", 1))
        if frame_count != 1:
            raise ValueError(
                f"Load Image List requires one image per file; {annotated_path!r} "
                f"contains {frame_count} frames."
            )

        load_image = import_module("nodes").LoadImage()
        result = load_image.load_image(annotated_path)
        image = result[0]
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"ComfyUI did not return an IMAGE for {annotated_path!r}.")
        if image.ndim != 4 or int(image.shape[0]) != 1:
            raise ValueError(
                f"Load Image List requires one image per file; {annotated_path!r} "
                f"returned shape {tuple(image.shape)}."
            )
        return image

    def fingerprint(self, annotated_path: str) -> str:
        """Return the content fingerprint for one validated image file."""

        path = self._resolve_path(annotated_path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _resolve_path(self, annotated_path: str) -> Path:
        """Resolve an existing Comfy annotated path without arbitrary file IO."""

        if not isinstance(annotated_path, str) or not annotated_path:
            raise ValueError("Load Image List requires a non-empty image file path.")
        folder_paths = import_module("folder_paths")
        if not folder_paths.exists_annotated_filepath(annotated_path):
            raise ValueError(f"Image file does not exist: {annotated_path!r}.")
        return Path(folder_paths.get_annotated_filepath(annotated_path))
