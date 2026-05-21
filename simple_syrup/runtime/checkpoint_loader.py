# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load checkpoints with optional external VAE replacement."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType
from typing import Any, Protocol, cast, runtime_checkable

from .vae_loader import VaeLoaderService

USE_CHECKPOINT_VAE_CHOICE = "Use Checkpoint VAE"
CLIP_SKIP_DEFAULT = False
CLIP_SKIP_LAYER = -2


class CheckpointLoaderService:
    """Load a checkpoint and optionally adjust its CLIP and VAE outputs."""

    def __init__(
        self,
        folder_paths_module: ModuleType | None = None,
        vae_loader: VaeLoaderBoundary | None = None,
    ) -> None:
        """Create a checkpoint loader with injectable runtime boundaries."""

        self._folder_paths_module = folder_paths_module
        self._vae_loader = vae_loader or VaeLoaderService(folder_paths_module)

    def load_checkpoint(
        self,
        ckpt_name: str,
        vae_name: str,
        clip_skip: bool = CLIP_SKIP_DEFAULT,
    ) -> tuple[object, object, object]:
        """Return MODEL, CLIP, and selected VAE objects."""

        _validate_clip_skip(clip_skip)
        folder_paths = self._folder_paths()
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        comfy_sd = _comfy_sd()
        loaded = tuple(
            cast(
                Iterable[object],
                comfy_sd.load_checkpoint_guess_config(
                    ckpt_path,
                    output_vae=True,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                ),
            )
        )
        model = loaded[0]
        clip = loaded[1]
        checkpoint_vae = loaded[2]
        selected_clip = _selected_clip(clip, clip_skip)

        if vae_name == USE_CHECKPOINT_VAE_CHOICE:
            return model, selected_clip, checkpoint_vae

        return model, selected_clip, self._vae_loader.load_vae(vae_name)

    def _folder_paths(self) -> ModuleType:
        """Return the ComfyUI folder_paths module."""

        if self._folder_paths_module is not None:
            return self._folder_paths_module
        module: Any = importlib.import_module("folder_paths")
        if not isinstance(module, ModuleType):
            raise TypeError("folder_paths import did not return a module.")
        self._folder_paths_module = module
        return module


class VaeLoaderBoundary(Protocol):
    """VAE loader interface required by the checkpoint loader."""

    def load_vae(self, vae_name: str) -> object:
        """Load the named external VAE."""


@runtime_checkable
class ClipLayerBoundary(Protocol):
    """CLIP interface required to apply the ComfyUI clip-skip layer."""

    def clone(self) -> ClipLayerBoundary:
        """Return an independent CLIP object."""

    def clip_layer(self, layer_idx: int) -> None:
        """Set the CLIP layer index used during prompt encoding."""


def _validate_clip_skip(clip_skip: object) -> None:
    """Reject non-boolean clip-skip selections from runtime callers."""

    if not isinstance(clip_skip, bool):
        raise TypeError("clip_skip must be a boolean.")


def _selected_clip(clip: object, clip_skip: bool) -> object:
    """Return the loaded CLIP or a cloned CLIP with clip skip applied."""

    if not clip_skip:
        return clip

    if not isinstance(clip, ClipLayerBoundary):
        raise TypeError(
            "clip_skip requires a CLIP object with clone() and clip_layer()."
        )

    selected_clip = clip.clone()
    selected_clip.clip_layer(CLIP_SKIP_LAYER)
    return selected_clip


def _comfy_sd() -> Any:
    """Import ComfyUI's stable diffusion loading module lazily."""

    return importlib.import_module("comfy.sd")
