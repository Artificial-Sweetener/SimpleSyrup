# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for ordered authored-mask loading."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import ClassVar

import torch

from ..runtime.mask_file_loader import MaskFileLoader
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


class LoadMaskBatchService:
    """Load ordered files into one dimensionally consistent MASK batch."""

    loader_class: ClassVar[type[MaskFileLoader]] = MaskFileLoader

    def validate(self, files: Sequence[str], channel: str) -> None:
        """Validate every selected file through the native Comfy mask loader."""

        ordered_files = self._validate_files(files)
        loader = self.loader_class()
        for path in ordered_files:
            loader.validate(path, channel)

    def load(self, files: Sequence[str], channel: str) -> torch.Tensor:
        """Load one or many authored masks without changing their order."""

        masks = self.load_each(files, channel)
        expected_shape = tuple(masks[0].shape[1:])
        for index, mask in enumerate(masks[1:], start=1):
            if tuple(mask.shape[1:]) != expected_shape:
                raise ValueError(
                    "Load Mask Batch requires every mask to have identical "
                    f"dimensions; mask 0 is {expected_shape[0]}x{expected_shape[1]} "
                    f"but mask {index} is {mask.shape[1]}x{mask.shape[2]}."
                )
        batch = torch.cat(masks, dim=0)
        LOGGER.info(
            "Authored mask batch loaded",
            extra={
                "operation": "load_mask_batch",
                "mask_count": int(batch.shape[0]),
                "mask_height": int(batch.shape[1]),
                "mask_width": int(batch.shape[2]),
                "channel": channel,
            },
        )
        return batch

    def load_each(self, files: Sequence[str], channel: str) -> tuple[torch.Tensor, ...]:
        """Load ordered masks without requiring batch-compatible dimensions."""

        ordered_files = self._validate_files(files)
        loader = self.loader_class()
        return tuple(loader.load(path, channel) for path in ordered_files)

    def fingerprint(self, files: Sequence[str], channel: str) -> str:
        """Return an order-sensitive fingerprint for files and channel."""

        ordered_files = self._validate_files(files)
        loader = self.loader_class()
        digest = hashlib.sha256()
        digest.update(channel.encode("utf-8"))
        for path in ordered_files:
            encoded_path = path.encode("utf-8")
            digest.update(len(encoded_path).to_bytes(8, "big"))
            digest.update(encoded_path)
            digest.update(loader.fingerprint(path).encode("ascii"))
        return digest.hexdigest()

    def available_files(self) -> tuple[str, ...]:
        """Return Comfy input images eligible for selection."""

        return self.loader_class().available_files()

    def _validate_files(self, files: Sequence[str]) -> tuple[str, ...]:
        """Return a non-empty ordered immutable file sequence."""

        ordered: tuple[str, ...]
        if isinstance(files, str):
            ordered = (files,)
        else:
            ordered = tuple(files)
        if not ordered:
            raise ValueError("Load Mask Batch requires at least one mask file.")
        if any(not isinstance(path, str) or not path for path in ordered):
            raise TypeError("Load Mask Batch mask files must be non-empty strings.")
        return ordered
