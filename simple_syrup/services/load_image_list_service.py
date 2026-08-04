# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for ordered authored-image list loading."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch

from ..domain.ordered_files import OrderedFileSelection
from ..runtime.image_file_loader import ImageFileLoader
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


class LoadImageListService:
    """Load ordered files as independently sized singleton IMAGE tensors."""

    loader_class: ClassVar[type[ImageFileLoader]] = ImageFileLoader

    def validate(self, files: Sequence[str]) -> None:
        """Validate every selected file through Comfy's native image loader."""

        ordered_files = self._selection(files).paths
        loader = self.loader_class()
        for path in ordered_files:
            loader.validate(path)

    def load(self, files: Sequence[str]) -> list[torch.Tensor]:
        """Load every authored position without batching or resizing images."""

        ordered_files = self._selection(files).paths
        loader = self.loader_class()
        images = [loader.load(path) for path in ordered_files]
        LOGGER.info(
            "Authored image list loaded",
            extra={
                "operation": "load_image_list",
                "image_count": len(images),
                "image_shapes": [tuple(image.shape) for image in images],
            },
        )
        return images

    def fingerprint(self, files: Sequence[str]) -> str:
        """Return an order-sensitive fingerprint for all selected images."""

        loader = self.loader_class()
        return self._selection(files).fingerprint(loader.fingerprint)

    def available_files(self) -> tuple[str, ...]:
        """Return Comfy input images eligible for selection."""

        return self.loader_class().available_files()

    def _selection(self, files: Sequence[str]) -> OrderedFileSelection:
        """Return validated positional image-file state."""

        return OrderedFileSelection.require(
            files,
            node_name="Load Image List",
            item_name="image",
        )
