# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load checkpoints with optional external VAE replacement."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType
from typing import Any, Protocol, cast

from .clip_patcher_mutations import ClipLayerMutation
from .patcher_lifecycle import PATCHER_LIFECYCLE, ComfyPatcherLifecycle
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
        patcher_lifecycle: ComfyPatcherLifecycle | None = None,
    ) -> None:
        """Create a checkpoint loader with injectable runtime boundaries."""

        self._folder_paths_module = folder_paths_module
        self._vae_loader = vae_loader or VaeLoaderService(folder_paths_module)
        self._patcher_lifecycle = patcher_lifecycle or PATCHER_LIFECYCLE

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
        selected_clip = self._selected_clip(clip, clip_skip)

        if vae_name == USE_CHECKPOINT_VAE_CHOICE:
            selected_vae = checkpoint_vae
        else:
            selected_vae = self._vae_loader.load_vae(vae_name)

        return (
            model,
            selected_clip,
            self._patcher_lifecycle.preserve_vae(
                selected_vae,
                operation="SimpleSyrup checkpoint loading",
            ),
        )

    def _selected_clip(self, clip: object, clip_skip: bool) -> object:
        """Return the loaded CLIP or a lifecycle-owned clip-skip derivation."""

        if not clip_skip:
            return clip
        return self._patcher_lifecycle.derive_clip(
            clip,
            (ClipLayerMutation(CLIP_SKIP_LAYER),),
            operation="SimpleSyrup checkpoint clip skip",
        )

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


def _validate_clip_skip(clip_skip: object) -> None:
    """Reject non-boolean clip-skip selections from runtime callers."""

    if not isinstance(clip_skip, bool):
        raise TypeError("clip_skip must be a boolean.")


def _comfy_sd() -> Any:
    """Import ComfyUI's stable diffusion loading module lazily."""

    return importlib.import_module("comfy.sd")
