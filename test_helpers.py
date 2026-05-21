# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared typed helpers for SimpleSyrup tests."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import torch


def make_image_tensor(
    batch_size: int = 1,
    height: int = 4,
    width: int = 6,
    channels: int = 3,
) -> torch.Tensor:
    """Create a deterministic BHWC image tensor for tests."""

    values = torch.arange(
        batch_size * height * width * channels,
        dtype=torch.float32,
    )
    return values.reshape(batch_size, height, width, channels) / values.numel()


def make_mask_tensor(
    batch_size: int = 1,
    height: int = 4,
    width: int = 6,
) -> torch.Tensor:
    """Create a deterministic BHW mask tensor for tests."""

    values = torch.arange(batch_size * height * width, dtype=torch.float32)
    return values.reshape(batch_size, height, width) / values.numel()


class FakeFolderPaths(ModuleType):
    """Small fake for the ComfyUI folder_paths module."""

    def __init__(self, models_dir: Path) -> None:
        """Create fake folder path state."""

        super().__init__("folder_paths")
        self.models_dir = str(models_dir)
        self.folder_names_and_paths: dict[str, tuple[list[str], set[str]]] = {}

    def add_model_folder_path(self, folder_name: str, full_folder_path: str) -> None:
        """Record a registered folder."""

        self.folder_names_and_paths[folder_name] = ([full_folder_path], set())
