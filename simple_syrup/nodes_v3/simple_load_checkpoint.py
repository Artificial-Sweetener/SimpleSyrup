# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node wrapper for checkpoint loading with optional VAE override."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..nodes.simple_load_checkpoint import SimpleLoadCheckpoint

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class SimpleLoadCheckpointV3(_ComfyNodeBase):
    """Expose Simple Load Checkpoint through Comfy's v3 extension API."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Simple Load Checkpoint v3 schema."""

        required = SimpleLoadCheckpoint.INPUT_TYPES()["required"]
        checkpoint_choices = list(required["ckpt_name"][0])
        vae_options = required["vae_name"][1]
        vae_choices = list(required["vae_name"][0])
        vae_default = str(vae_options["default"])
        clip_skip_options = required["clip_skip"][1]
        clip_skip_default = clip_skip_options["default"]
        if not isinstance(clip_skip_default, bool):
            raise TypeError("clip_skip default must be a boolean.")

        return _comfy_io.Schema(
            node_id="SimpleSyrup.SimpleLoadCheckpoint",
            display_name="Simple Load Checkpoint",
            category="SimpleSyrup/Loaders",
            description="Loads a checkpoint and optionally replaces its VAE.",
            search_aliases=["load checkpoint", "checkpoint", "ckpt", "vae"],
            inputs=[
                _comfy_io.Combo.Input(
                    "ckpt_name",
                    options=checkpoint_choices,
                    tooltip=tooltips.CHECKPOINT_MODEL_INPUT,
                ),
                _comfy_io.Combo.Input(
                    "vae_name",
                    options=vae_choices,
                    default=vae_default,
                    tooltip=tooltips.CHECKPOINT_VAE_INPUT,
                ),
                _comfy_io.Boolean.Input(
                    "clip_skip",
                    default=clip_skip_default,
                    tooltip=tooltips.CLIP_SKIP_INPUT,
                ),
            ],
            outputs=[
                _comfy_io.Model.Output(
                    "model",
                    tooltip=tooltips.MODEL_OUTPUT,
                ),
                _comfy_io.Clip.Output(
                    "clip",
                    tooltip=tooltips.CLIP_OUTPUT,
                ),
                _comfy_io.Vae.Output(
                    "vae",
                    tooltip=tooltips.VAE_OUTPUT,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        ckpt_name: str,
        vae_name: str,
        clip_skip: bool,
    ) -> tuple[object, object, object]:
        """Run the legacy loader implementation behind the v3 schema."""

        return SimpleLoadCheckpoint().load_checkpoint(
            ckpt_name=ckpt_name,
            vae_name=vae_name,
            clip_skip=clip_skip,
        )
