# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact cleanup-owned Comfy model visibility for U11."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.comfy_integration.managed_model_links import (
    ManagedComfyModelLinks,
    ManagedModelLink,
)


def test_links_are_hard_links_and_cleanup_removes_only_owned_targets(
    tmp_path: Path,
) -> None:
    """Expose two categories transactionally without copying model bytes."""

    model_root = tmp_path / "models"
    checkpoints = model_root / "checkpoints"
    loras = model_root / "loras"
    checkpoints.mkdir(parents=True)
    loras.mkdir()
    checkpoint = tmp_path / "checkpoint.safetensors"
    lora = tmp_path / "adapter.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    lora.write_bytes(b"lora")
    owner = ManagedComfyModelLinks(
        model_root=model_root,
        links=(
            ManagedModelLink(
                checkpoint,
                "checkpoints",
                r"simple_syrup_u11\checkpoint.safetensors",
            ),
            ManagedModelLink(
                lora,
                "loras",
                r"simple_syrup_u11\adapter.safetensors",
            ),
        ),
    )

    with owner:
        checkpoint_link = checkpoints / "simple_syrup_u11" / checkpoint.name
        lora_link = loras / "simple_syrup_u11" / lora.name
        assert checkpoint_link.read_bytes() == b"checkpoint"
        assert lora_link.read_bytes() == b"lora"
        assert os.path.samefile(checkpoint, checkpoint_link)
        assert os.path.samefile(lora, lora_link)

    assert owner.cleaned
    assert checkpoint.is_file()
    assert lora.is_file()
    assert not (checkpoints / "simple_syrup_u11").exists()
    assert not (loras / "simple_syrup_u11").exists()


def test_links_support_split_model_categories(tmp_path: Path) -> None:
    """Expose split diffusion, text-encoder, and VAE fixtures transactionally."""

    model_root = tmp_path / "models"
    categories = ("diffusion_models", "text_encoders", "vae")
    sources: list[Path] = []
    for category in categories:
        (model_root / category).mkdir(parents=True)
        source = tmp_path / f"{category}.safetensors"
        source.write_bytes(category.encode())
        sources.append(source)
    owner = ManagedComfyModelLinks(
        model_root=model_root,
        links=tuple(
            ManagedModelLink(
                source,
                category,
                f"managed/{category}.safetensors",
            )
            for category, source in zip(categories, sources, strict=True)
        ),
    )

    with owner:
        assert all(
            (model_root / category / "managed" / f"{category}.safetensors").is_file()
            for category in categories
        )

    assert owner.cleaned
    assert all(
        not (model_root / category / "managed").exists() for category in categories
    )


def test_collision_fails_before_creating_any_link(tmp_path: Path) -> None:
    """Refuse ambiguous ownership without touching the existing target."""

    model_root = tmp_path / "models"
    directory = model_root / "loras" / "simple_syrup_u11"
    (model_root / "checkpoints").mkdir(parents=True)
    directory.mkdir(parents=True)
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"source")
    collision = directory / "adapter.safetensors"
    collision.write_bytes(b"existing")
    owner = ManagedComfyModelLinks(
        model_root=model_root,
        links=(
            ManagedModelLink(
                source,
                "loras",
                r"simple_syrup_u11\adapter.safetensors",
            ),
        ),
    )

    with pytest.raises(FileExistsError):
        owner.__enter__()

    assert collision.read_bytes() == b"existing"


def test_exception_cleanup_removes_nested_owned_directories(tmp_path: Path) -> None:
    """Clean nested targets even when the protected operation raises."""

    model_root = tmp_path / "models"
    (model_root / "loras").mkdir(parents=True)
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"source")
    owner = ManagedComfyModelLinks(
        model_root=model_root,
        links=(
            ManagedModelLink(
                source,
                "loras",
                r"family\style\adapter.safetensors",
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="operation failed"):
        with owner:
            raise RuntimeError("operation failed")

    assert owner.cleaned
    assert not (model_root / "loras" / "family").exists()
