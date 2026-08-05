# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI filesystem adapter for authored mask files."""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path

import torch
from PIL import Image, ImageOps

MASK_CHANNELS: tuple[str, ...] = ("alpha", "red", "green", "blue")


class MaskFileLoader:
    """Load one static mask while preserving authored image geometry."""

    def available_files(self) -> tuple[str, ...]:
        """Return choices declared by ComfyUI's native mask loader."""

        declaration = import_module("nodes").LoadImageMask.INPUT_TYPES()
        image_input = declaration["required"]["image"]
        choices = image_input[0]
        if not isinstance(choices, (list, tuple)):
            raise TypeError("ComfyUI LoadImageMask returned invalid image choices.")
        return tuple(str(value) for value in choices)

    def validate(self, annotated_path: str, channel: str) -> None:
        """Validate one path with ComfyUI's native mask-loader contract."""

        self._validate_channel(channel)
        result = import_module("nodes").LoadImageMask.VALIDATE_INPUTS(annotated_path)
        if result is not True:
            raise ValueError(str(result))

    def load(self, annotated_path: str, channel: str) -> torch.Tensor:
        """Return one BHW mask with zero source-sized coverage when alpha is absent."""

        self.validate(annotated_path, channel)
        path = self._resolve_path(annotated_path)
        with Image.open(path) as image:
            frame_count = int(getattr(image, "n_frames", 1))
            normalized_image = ImageOps.exif_transpose(image)
            source_width, source_height = normalized_image.size
            has_alpha = "A" in normalized_image.getbands()
        if frame_count != 1:
            raise ValueError(
                f"Load Mask Batch requires one mask per file; {annotated_path!r} "
                f"contains {frame_count} frames."
            )

        load_image_mask = import_module("nodes").LoadImageMask()
        result = load_image_mask.load_image_mask(annotated_path, channel)
        mask = result[0]
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"ComfyUI did not return a MASK for {annotated_path!r}.")
        if channel == "alpha" and not has_alpha:
            mask = mask.new_zeros((1, source_height, source_width))
        if mask.ndim != 3 or int(mask.shape[0]) != 1:
            raise ValueError(
                f"Load Mask Batch requires one mask per file; {annotated_path!r} "
                f"returned shape {tuple(mask.shape)}."
            )
        return mask.float().clamp(0.0, 1.0)

    def fingerprint(self, annotated_path: str) -> str:
        """Return the content fingerprint for one validated mask file."""

        path = self._resolve_path(annotated_path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _resolve_path(self, annotated_path: str) -> Path:
        """Resolve an existing Comfy annotated path without accepting arbitrary IO."""

        if not isinstance(annotated_path, str) or not annotated_path:
            raise ValueError("Load Mask Batch requires a non-empty mask file path.")
        folder_paths = import_module("folder_paths")
        if not folder_paths.exists_annotated_filepath(annotated_path):
            raise ValueError(f"Mask file does not exist: {annotated_path!r}.")
        return Path(folder_paths.get_annotated_filepath(annotated_path))

    def _validate_channel(self, channel: str) -> None:
        """Reject channels outside ComfyUI's native mask choices."""

        if channel not in MASK_CHANNELS:
            raise ValueError(
                f"mask channel must be one of {MASK_CHANNELS}; received {channel!r}."
            )
