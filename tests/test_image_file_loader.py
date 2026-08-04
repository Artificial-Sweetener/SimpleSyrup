# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Comfy authored-image file adapter."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
import torch
from PIL import Image

from simple_syrup.runtime.image_file_loader import ImageFileLoader


def patch_path(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    *,
    exists: bool = True,
) -> None:
    """Route Comfy annotated-path helpers to one temporary file."""

    folder_paths = import_module("folder_paths")
    monkeypatch.setattr(import_module("nodes"), "folder_paths", folder_paths)
    monkeypatch.setattr(folder_paths, "exists_annotated_filepath", lambda value: exists)
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda value: str(path))


def test_load_uses_native_image_loader_and_returns_one_singleton(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static files are decoded through Comfy's LoadImage behavior."""

    path = tmp_path / "image.png"
    Image.new("RGB", (3, 2), color=(255, 0, 0)).save(path)
    patch_path(monkeypatch, path)
    calls: list[str] = []

    class FakeNativeLoader:
        """Return one recognizable native image."""

        @classmethod
        def VALIDATE_INPUTS(cls, value: str) -> bool:
            """Accept the annotated path through the native contract."""

            del cls, value
            return True

        def load_image(self, value: str) -> tuple[torch.Tensor, torch.Tensor]:
            """Record the native loader argument."""

            calls.append(value)
            return torch.full((1, 2, 3, 3), 0.75), torch.zeros((1, 2, 3))

    monkeypatch.setattr(import_module("nodes"), "LoadImage", FakeNativeLoader)

    result = ImageFileLoader().load("image.png")

    assert calls == ["image.png"]
    assert tuple(result.shape) == (1, 2, 3, 3)
    assert torch.all(result == 0.75)


def test_available_files_uses_native_image_loader_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node's file combo is sourced from native LoadImage inputs."""

    class FakeNativeLoader:
        """Expose recognizable native widget choices."""

        @classmethod
        def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[list[str], dict[str, bool]]]]:
            """Return the native image-upload declaration shape."""

            del cls
            return {"required": {"image": (["z.png", "a.png"], {"image_upload": True})}}

    monkeypatch.setattr(import_module("nodes"), "LoadImage", FakeNativeLoader)

    assert ImageFileLoader().available_files() == ("z.png", "a.png")


def test_rejects_multiframe_and_missing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One selected file cannot silently expand into several list items."""

    path = tmp_path / "animated.gif"
    first = Image.new("RGB", (2, 2), color=0)
    second = Image.new("RGB", (2, 2), color=255)
    first.save(path, save_all=True, append_images=[second])
    patch_path(monkeypatch, path)
    loader = ImageFileLoader()

    with pytest.raises(ValueError, match="contains 2 frames"):
        loader.load("animated.gif")

    patch_path(monkeypatch, path, exists=False)
    with pytest.raises(ValueError, match="does not exist"):
        loader.fingerprint("missing.png")


def test_fingerprint_reads_validated_file_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Content changes invalidate the image-list node cache."""

    path = tmp_path / "image.png"
    path.write_bytes(b"first")
    patch_path(monkeypatch, path)
    loader = ImageFileLoader()
    first = loader.fingerprint("image.png")
    path.write_bytes(b"second")

    assert loader.fingerprint("image.png") != first
