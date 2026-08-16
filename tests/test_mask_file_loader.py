# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Comfy authored-mask file adapter."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
import torch
from PIL import Image

from simple_syrup.runtime.mask_file_loader import MaskFileLoader


def _patch_path(
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


def test_load_uses_native_mask_loader_and_validated_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static files are decoded through Comfy's LoadImageMask behavior."""

    path = tmp_path / "mask.png"
    Image.new("RGB", (3, 2), color=(128, 0, 0)).save(path)
    _patch_path(monkeypatch, path)
    calls: list[tuple[str, str]] = []

    class FakeNativeLoader:
        """Return one recognizable native mask."""

        @classmethod
        def VALIDATE_INPUTS(cls, value: str) -> bool:
            """Accept the annotated path through the native contract."""

            del cls, value
            return True

        def load_image_mask(self, value: str, channel: str) -> tuple[torch.Tensor]:
            """Record native loader arguments."""

            calls.append((value, channel))
            return (torch.full((1, 2, 3), 0.75),)

    nodes = import_module("nodes")

    monkeypatch.setattr(nodes, "LoadImageMask", FakeNativeLoader)

    result = MaskFileLoader().load("mask.png", "red")

    assert calls == [("mask.png", "red")]
    assert torch.all(result == 0.75)


def test_missing_alpha_returns_zero_coverage_at_source_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RGB masks retain their authored geometry instead of Comfy's 64px fallback."""

    path = tmp_path / "rgb-mask.png"
    Image.new("RGB", (3, 2), color=(255, 255, 255)).save(path)
    _patch_path(monkeypatch, path)

    class FakeNativeLoader:
        """Reproduce Comfy's missing-alpha fallback mask."""

        @classmethod
        def VALIDATE_INPUTS(cls, value: str) -> bool:
            """Accept the annotated path through the native contract."""

            del cls, value
            return True

        def load_image_mask(self, value: str, channel: str) -> tuple[torch.Tensor]:
            """Return the native 64x64 zero mask for missing alpha."""

            del self, value, channel
            return (torch.zeros((1, 64, 64)),)

    monkeypatch.setattr(import_module("nodes"), "LoadImageMask", FakeNativeLoader)

    result = MaskFileLoader().load("rgb-mask.png", "alpha")

    assert result.shape == (1, 2, 3)
    assert torch.count_nonzero(result) == 0


def test_native_white_endpoint_is_restored_without_changing_lower_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat the current decoder's 255 endpoint as exact authored coverage."""

    path = tmp_path / "hard-mask.png"
    image = Image.new("RGB", (3, 1))
    image.putdata(((0, 0, 0), (254, 254, 254), (255, 255, 255)))
    image.save(path)
    _patch_path(monkeypatch, path)

    class FakeNativeLoader:
        """Expose the installed decoder's endpoint and two lower values."""

        @classmethod
        def VALIDATE_INPUTS(cls, value: str) -> bool:
            """Accept the annotated path through the native contract."""

            del cls, value
            return True

        def load_image_mask(self, value: str, channel: str) -> tuple[torch.Tensor]:
            """Return current-host normalized channel values."""

            del self, value, channel
            return (torch.tensor([[[0.0, 0.9922, 0.996108949]]]),)

    monkeypatch.setattr(import_module("nodes"), "LoadImageMask", FakeNativeLoader)

    result = MaskFileLoader().load("hard-mask.png", "red")

    assert torch.equal(result, torch.tensor([[[0.0, 0.9922, 1.0]]]))


def test_available_files_uses_native_mask_loader_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node's file combo is sourced from native LoadImageMask inputs."""

    class FakeNativeLoader:
        """Expose recognizable native widget choices."""

        @classmethod
        def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[list[str], dict[str, bool]]]]:
            """Return the native image-upload declaration shape."""

            del cls
            return {"required": {"image": (["z.png", "a.png"], {"image_upload": True})}}

    monkeypatch.setattr(import_module("nodes"), "LoadImageMask", FakeNativeLoader)

    assert MaskFileLoader().available_files() == ("z.png", "a.png")


def test_rejects_missing_multiframe_and_invalid_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Files cannot silently add regions or escape native channel semantics."""

    path = tmp_path / "animated.gif"
    first = Image.new("L", (2, 2), color=0)
    second = Image.new("L", (2, 2), color=255)
    first.save(path, save_all=True, append_images=[second])
    _patch_path(monkeypatch, path)
    loader = MaskFileLoader()

    with pytest.raises(ValueError, match="contains 2 frames"):
        loader.load("animated.gif", "red")
    with pytest.raises(ValueError, match="mask channel must be one of"):
        loader.load("animated.gif", "luminance")

    _patch_path(monkeypatch, path, exists=False)
    with pytest.raises(ValueError, match="does not exist"):
        loader.fingerprint("missing.png")


def test_fingerprint_reads_validated_file_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Content changes invalidate the loader node cache."""

    path = tmp_path / "mask.png"
    path.write_bytes(b"first")
    _patch_path(monkeypatch, path)
    loader = MaskFileLoader()
    first = loader.fingerprint("mask.png")
    path.write_bytes(b"second")

    assert loader.fingerprint("mask.png") != first
