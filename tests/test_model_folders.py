# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ComfyUI model folder registration helpers."""

from __future__ import annotations

from pathlib import Path

from simple_syrup.runtime.model_folders import (
    get_model_folder_paths,
    nonrecursive_model_files,
    register_required_model_folders,
    resolve_model_file,
)
from test_helpers import FakeFolderPaths


def test_register_required_model_folders_adds_missing_folders(tmp_path: Path) -> None:
    """SimpleSyrup registers required masking model folders when absent."""

    fake = FakeFolderPaths(tmp_path)

    register_required_model_folders(fake)

    assert fake.folder_names_and_paths["sams"][0] == [str(tmp_path / "sams")]
    assert fake.folder_names_and_paths["grounding-dino"][0] == [
        str(tmp_path / "grounding-dino")
    ]
    assert fake.folder_names_and_paths["vitmatte"][0] == [str(tmp_path / "vitmatte")]
    assert fake.folder_names_and_paths["wd14_tagger"][0] == [
        str(tmp_path / "wd14_tagger")
    ]
    assert fake.folder_names_and_paths["wd14_tagger"][1] == {".onnx", ".csv"}


def test_get_model_folder_paths_includes_registered_and_fallback(
    tmp_path: Path,
) -> None:
    """Resolver preserves registered paths and appends the conventional fallback."""

    fake = FakeFolderPaths(tmp_path)
    fake.folder_names_and_paths["sams"] = ([str(tmp_path / "custom_sams")], {".pth"})

    paths = get_model_folder_paths("sams", fake)

    assert paths == [tmp_path / "custom_sams", tmp_path / "sams"]


def test_resolve_model_file_finds_registered_file(tmp_path: Path) -> None:
    """Model file lookup searches registered folders."""

    fake = FakeFolderPaths(tmp_path)
    folder = tmp_path / "sams"
    folder.mkdir()
    model_path = folder / "sam_vit_b_01ec64.pth"
    model_path.write_bytes(b"model")
    register_required_model_folders(fake)

    assert resolve_model_file("sams", "sam_vit_b_01ec64.pth", fake) == model_path


def test_nonrecursive_model_files_does_not_walk_subdirectories(tmp_path: Path) -> None:
    """Bounded discovery scans only direct children."""

    fake = FakeFolderPaths(tmp_path)
    folder = tmp_path / "sams"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    (folder / "sam_vit_b_01ec64.pth").write_bytes(b"model")
    (nested / "sam_vit_h_4b8939.pth").write_bytes(b"model")
    register_required_model_folders(fake)

    assert nonrecursive_model_files("sams", fake) == ["sam_vit_b_01ec64.pth"]


def test_nonrecursive_model_files_uses_wd14_extensions(tmp_path: Path) -> None:
    """WD14 discovery includes ONNX and CSV files without accepting unrelated files."""

    fake = FakeFolderPaths(tmp_path)
    folder = tmp_path / "wd14_tagger"
    folder.mkdir()
    (folder / "wd-eva02-large-tagger-v3.onnx").write_bytes(b"onnx")
    (folder / "wd-eva02-large-tagger-v3.csv").write_text("", encoding="utf-8")
    (folder / "ignored.pth").write_bytes(b"model")
    register_required_model_folders(fake)

    assert nonrecursive_model_files("wd14_tagger", fake) == [
        "wd-eva02-large-tagger-v3.csv",
        "wd-eva02-large-tagger-v3.onnx",
    ]
