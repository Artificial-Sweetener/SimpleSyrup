# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize atomic Comfy attn1 replacement installation."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from comfy.model_patcher import ModelPatcher
from simple_syrup.runtime.model_attention_replacement_mutations import (
    ModelAttn1ReplacementsMutation,
)
from torch import nn


def test_attn1_replacements_install_every_declared_key_atomically() -> None:
    """Install one callback at each exact standard-UNet self-attention key."""

    model = _patcher()

    def replacement(*args: object) -> object:
        """Return the first callback argument for one inert fixture."""

        return args[0]

    ModelAttn1ReplacementsMutation(
        (("input", 4, 0), ("middle", 0, 0)),
        replacement,
    ).apply(model)

    installed = model.model_options["transformer_options"]["patches_replace"]["attn1"]
    assert installed == {
        ("input", 4, 0): replacement,
        ("middle", 0, 0): replacement,
    }


def test_attn1_replacements_reject_broad_or_exact_collisions_before_mutation() -> None:
    """Preserve existing replacement ownership without partial installation."""

    for existing_key in (("input", 4), ("input", 4, 0)):
        model = _patcher()

        def existing(*args: object) -> object:
            """Return the first argument through the foreign owner."""

            return args[0]

        replacements: dict[str, Any] = {"attn1": {existing_key: existing}}
        model.model_options["transformer_options"]["patches_replace"] = replacements
        before = replacements["attn1"].copy()

        with pytest.raises(ValueError, match="already installed"):
            ModelAttn1ReplacementsMutation(
                (("input", 4, 0), ("middle", 0, 0)),
                lambda *args: args[0],
            ).apply(model)

        assert replacements["attn1"] == before


def _patcher() -> ModelPatcher:
    """Return one real CPU patcher with mutable Comfy transformer options."""

    root = nn.Module()
    root.diffusion_model = nn.Identity()
    device = torch.device("cpu")
    return ModelPatcher(root, load_device=device, offload_device=device)
