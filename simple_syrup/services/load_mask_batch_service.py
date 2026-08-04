# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for ordered authored-mask loading."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch

from ..domain.ordered_files import OrderedFileSelection
from ..runtime.mask_file_loader import MaskFileLoader
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


class LoadMaskBatchService:
    """Load ordered files into one dimensionally consistent MASK batch."""

    loader_class: ClassVar[type[MaskFileLoader]] = MaskFileLoader

    def validate(self, files: Sequence[str], channel: str) -> None:
        """Validate every selected file through the native Comfy mask loader."""

        ordered_files = self._selection(files).paths
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

        ordered_files = self._selection(files).paths
        loader = self.loader_class()
        return tuple(loader.load(path, channel) for path in ordered_files)

    def fingerprint(self, files: Sequence[str], channel: str) -> str:
        """Return an order-sensitive fingerprint for files and channel."""

        loader = self.loader_class()
        return self._selection(files).fingerprint(
            loader.fingerprint,
            context=(channel,),
        )

    def available_files(self) -> tuple[str, ...]:
        """Return Comfy input images eligible for selection."""

        return self.loader_class().available_files()

    def _selection(self, files: Sequence[str]) -> OrderedFileSelection:
        """Return validated positional mask-file state."""

        return OrderedFileSelection.require(
            files,
            node_name="Load Mask Batch",
            item_name="mask",
        )
