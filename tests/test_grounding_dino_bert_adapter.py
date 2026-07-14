# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test GroundingDINO adaptation to supported Transformers BERT APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch
import transformers
from torch import nn
from transformers import BertConfig, BertModel

from simple_syrup.runtime.grounding_dino_bert_adapter import (
    prepare_grounding_dino_bert_attention_mask,
)


@pytest.mark.parametrize("transformers_version", ["4.50.3", "4.57.6"])
def test_transformers_v4_preserves_grounding_dino_attention_mask(
    transformers_version: str,
) -> None:
    """Let supported v4 BERT normalize its native three-dimensional mask."""

    text_encoder = nn.Linear(2, 2)
    attention_mask = _block_attention_mask()

    prepared = prepare_grounding_dino_bert_attention_mask(
        attention_mask,
        text_encoder=text_encoder,
        transformers_version=transformers_version,
    )

    assert prepared is attention_mask


@pytest.mark.parametrize("transformers_version", ["5.0.0", "5.13.1"])
def test_transformers_v5_receives_prepared_additive_attention_mask(
    transformers_version: str,
) -> None:
    """Give v5 BERT the four-dimensional additive mask its public API accepts."""

    text_encoder = nn.Linear(2, 2, dtype=torch.float16)
    attention_mask = _block_attention_mask()

    prepared = prepare_grounding_dino_bert_attention_mask(
        attention_mask,
        text_encoder=text_encoder,
        transformers_version=transformers_version,
    )

    assert prepared.shape == (1, 1, 4, 4)
    assert prepared.dtype is torch.float16
    assert prepared.device == attention_mask.device
    assert torch.equal(
        prepared[0, 0].eq(0),
        attention_mask[0],
    )
    assert prepared[0, 0, 0, 2] == torch.finfo(torch.float16).min


@pytest.mark.parametrize("transformers_version", ["4.50.3", "5.13.1"])
def test_padding_attention_mask_passes_through_for_supported_versions(
    transformers_version: str,
) -> None:
    """Leave ordinary tokenizer padding masks to native BERT handling."""

    text_encoder = nn.Linear(2, 2)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    prepared = prepare_grounding_dino_bert_attention_mask(
        attention_mask,
        text_encoder=text_encoder,
        transformers_version=transformers_version,
    )

    assert prepared is attention_mask


@pytest.mark.parametrize(
    "transformers_version",
    ["4.50.2", "6.0.0", "not-a-version"],
)
def test_unsupported_transformers_versions_fail_with_actionable_error(
    transformers_version: str,
) -> None:
    """Reject unverified API generations before BERT fails inside inference."""

    with pytest.raises(RuntimeError, match=r"Transformers >=4\.50\.3,<6"):
        prepare_grounding_dino_bert_attention_mask(
            _block_attention_mask(),
            text_encoder=nn.Linear(2, 2),
            transformers_version=transformers_version,
        )


def test_sub_sentence_attention_mask_must_be_boolean_and_square() -> None:
    """Reject malformed GroundingDINO block masks at their owning boundary."""

    text_encoder = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="boolean"):
        prepare_grounding_dino_bert_attention_mask(
            torch.ones((1, 4, 4)),
            text_encoder=text_encoder,
            transformers_version="5.13.1",
        )
    with pytest.raises(ValueError, match="square"):
        prepare_grounding_dino_bert_attention_mask(
            torch.ones((1, 4, 3), dtype=torch.bool),
            text_encoder=text_encoder,
            transformers_version="5.13.1",
        )


def test_installed_transformers_runs_native_bert_with_adapted_block_mask() -> None:
    """Exercise the installed v4 or v5 BERT implementation through the adapter."""

    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    bert_model_factory = cast(Callable[[BertConfig], nn.Module], BertModel)
    text_encoder = bert_model_factory(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = prepare_grounding_dino_bert_attention_mask(
        _block_attention_mask(),
        text_encoder=text_encoder,
        transformers_version=transformers.__version__,
    )

    with torch.inference_mode():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
            return_dict=True,
        )

    assert output.last_hidden_state.shape == (1, 4, 16)
    assert torch.isfinite(output.last_hidden_state).all()


def _block_attention_mask() -> torch.Tensor:
    """Return two isolated two-token attention blocks."""

    return torch.tensor(
        [
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, True],
                [False, False, True, True],
            ]
        ]
    )
