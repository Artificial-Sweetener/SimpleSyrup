# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize GroundingDINO text-token attention mask behavior."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_TOKEN_MASKS_PATH = (
    REPO_ROOT
    / "simple_syrup"
    / "third_party"
    / "groundingdino_runtime"
    / "models"
    / "GroundingDINO"
    / "text_token_masks.py"
)
MaskGenerator = Callable[
    [dict[str, torch.Tensor], list[int]],
    tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]],
]


def _load_module(path: Path) -> ModuleType:
    """Load one vendored module without importing the full model package."""

    spec = importlib.util.spec_from_file_location(
        "grounding_dino_text_token_masks_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_text_token_masks = cast(
    MaskGenerator,
    _load_module(TEXT_TOKEN_MASKS_PATH).generate_text_token_masks,
)


def test_text_token_masks_preserve_special_token_delimited_attention_blocks() -> None:
    """Keep tokens isolated within the sub-sentence ending at each delimiter."""

    tokenized = {"input_ids": torch.tensor([[101, 2001, 1012, 102]], dtype=torch.long)}

    attention_mask, position_ids, category_masks = generate_text_token_masks(
        tokenized,
        [101, 102, 1012, 1029],
    )

    expected_attention_mask = torch.tensor(
        [
            [
                [True, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, True],
            ]
        ]
    )
    expected_category_masks = torch.tensor(
        [
            [
                [False, True, False, False],
            ]
        ]
    )

    assert attention_mask.dtype is torch.bool
    assert torch.equal(attention_mask, expected_attention_mask)
    assert torch.equal(position_ids, torch.tensor([[0, 0, 1, 0]]))
    assert torch.equal(torch.stack(category_masks), expected_category_masks)
