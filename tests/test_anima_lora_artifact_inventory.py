# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify fail-closed safetensors inventory for Anima LoRA pairs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools.anima_lora_characterization.artifact_inventory import inspect_adapter


def test_inventory_records_matched_rank_decomposed_targets(tmp_path: Path) -> None:
    """Read target identity and dimensions without loading a model."""

    artifact = tmp_path / "fixture.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.cross_attn.q_proj.lora_A.weight": torch.zeros(
                (4, 8)
            ),
            "diffusion_model.blocks.0.cross_attn.q_proj.lora_B.weight": torch.zeros(
                (16, 4)
            ),
        },
        artifact,
        metadata={"modelspec.architecture": "anima/lora"},
    )

    inventory = inspect_adapter(artifact)

    assert inventory.metadata["modelspec.architecture"] == "anima/lora"
    assert len(inventory.sha256) == 64
    assert inventory.pairs[0].target == "diffusion_model.blocks.0.cross_attn.q_proj"
    assert inventory.pairs[0].rank == 4
    assert inventory.pairs[0].input_features == 8
    assert inventory.pairs[0].output_features == 16


def test_inventory_rejects_incomplete_or_foreign_targets(tmp_path: Path) -> None:
    """Do not characterize a partial or non-Anima adapter as supported."""

    incomplete = tmp_path / "incomplete.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.cross_attn.q_proj.lora_A.weight": torch.zeros(
                (4, 8)
            )
        },
        incomplete,
    )
    with pytest.raises(ValueError, match="incomplete"):
        inspect_adapter(incomplete)

    foreign = tmp_path / "foreign.safetensors"
    save_file(
        {
            "foreign.lora_A.weight": torch.zeros((4, 8)),
            "foreign.lora_B.weight": torch.zeros((16, 4)),
        },
        foreign,
    )
    with pytest.raises(ValueError, match="Anima block"):
        inspect_adapter(foreign)
