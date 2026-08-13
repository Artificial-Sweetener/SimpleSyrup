# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove standard adapter decoding and exhaustive Anima target admission."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from simple_syrup.runtime.regional_lora.anima_targets import (
    ANIMA_LORA_TARGET_CLASSIFIER,
    AnimaLoraAdmissionError,
    AnimaLoraTargetFamily,
)
from simple_syrup.runtime.regional_lora.standard_adapter import (
    STANDARD_LORA_ADAPTER_DECODER,
)

PINNED_ADAPTER_A_PATH = Path(r"<MODEL_ROOT>\Loras\Anima\style\adapter-a.safetensors")
PINNED_TURBO_PATH = Path(
    r"<MODEL_ROOT>\Loras\Anima\anima-turbo-lora-v0.2.safetensors"
)
PINNED_character_a_PATH = Path(
    r"<MODEL_ROOT>\Loras\Anima\character"
    r"\Sakura Kyouko anima v2.safetensors"
)


def test_pinned_adapter_a_admits_all_448_targets_and_16_families() -> None:
    """Require complete full-surface classification of the acceptance fixture."""

    weights = load_file(PINNED_ADAPTER_A_PATH, device="cpu")
    source_ids = {key: id(value) for key, value in weights.items()}

    admission = ANIMA_LORA_TARGET_CLASSIFIER.admit(weights)

    assert len(admission.targets) == 448
    assert {target.block_index for target in admission.targets} == set(range(28))
    assert {target.family for target in admission.targets} == set(AnimaLoraTargetFamily)
    assert all(target.adapter.rank == 32 for target in admission.targets)
    assert all(target.adapter.down.device.type == "cpu" for target in admission.targets)
    assert all(target.adapter.up.device.type == "cpu" for target in admission.targets)
    assert {key: id(value) for key, value in weights.items()} == source_ids


def test_pinned_turbo_rejects_all_60_llm_adapter_targets_atomically() -> None:
    """Reject the mixed diffusion/LLM payload without returning 448 partial targets."""

    weights = load_file(PINNED_TURBO_PATH, device="cpu")

    with pytest.raises(AnimaLoraAdmissionError) as captured:
        ANIMA_LORA_TARGET_CLASSIFIER.admit(weights)

    assert len(captured.value.issues) == 60
    assert all(
        issue.key.startswith("diffusion_model.llm_adapter.blocks.")
        for issue in captured.value.issues
    )
    assert {issue.reason for issue in captured.value.issues} == {
        "Anima LLM adapter target is outside the approved regional "
        "diffusion-LoRA contract"
    }


def test_pinned_character_a_admits_sd_scripts_layout_and_intrinsic_alpha_scale() -> None:
    """Admit the native Anima character fixture without losing alpha/rank scale."""

    weights = load_file(PINNED_character_a_PATH, device="cpu")

    admission = ANIMA_LORA_TARGET_CLASSIFIER.admit(weights)

    assert len(admission.targets) == 280
    assert {target.block_index for target in admission.targets} == set(range(28))
    assert {target.family for target in admission.targets} == {
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        AnimaLoraTargetFamily.SELF_ATTN_K,
        AnimaLoraTargetFamily.SELF_ATTN_V,
        AnimaLoraTargetFamily.SELF_ATTN_OUTPUT,
        AnimaLoraTargetFamily.CROSS_ATTN_Q,
        AnimaLoraTargetFamily.CROSS_ATTN_K,
        AnimaLoraTargetFamily.CROSS_ATTN_V,
        AnimaLoraTargetFamily.CROSS_ATTN_OUTPUT,
        AnimaLoraTargetFamily.MLP_LAYER1,
        AnimaLoraTargetFamily.MLP_LAYER2,
    }
    assert all(target.adapter.rank == 8 for target in admission.targets)
    assert all(target.adapter.intrinsic_scale == 0.5 for target in admission.targets)


def test_admission_reports_every_format_target_block_and_shape_failure() -> None:
    """Aggregate every unsupported key before any partial adapter can execute."""

    weights = {
        "unsupported.lora_down.weight": torch.ones((2, 2)),
        "incomplete.lora_A.weight": torch.ones((2, 2)),
        "foreign.layer.lora_A.weight": torch.ones((2, 2)),
        "foreign.layer.lora_B.weight": torch.ones((2, 2)),
        "diffusion_model.blocks.28.self_attn.q_proj.lora_A.weight": torch.ones(
            (2, 2048)
        ),
        "diffusion_model.blocks.28.self_attn.q_proj.lora_B.weight": torch.ones(
            (2048, 2)
        ),
        "diffusion_model.blocks.0.unknown.lora_A.weight": torch.ones((2, 2048)),
        "diffusion_model.blocks.0.unknown.lora_B.weight": torch.ones((2048, 2)),
        "diffusion_model.blocks.0.mlp.layer1.lora_A.weight": torch.ones((2, 4)),
        "diffusion_model.blocks.0.mlp.layer1.lora_B.weight": torch.ones((8, 2)),
    }

    with pytest.raises(AnimaLoraAdmissionError) as captured:
        ANIMA_LORA_TARGET_CLASSIFIER.admit(weights)

    message = str(captured.value)
    assert "before device work" in message
    assert "unsupported.lora_down.weight" in message
    assert "incomplete" in message
    assert "foreign.layer" in message
    assert "blocks.28.self_attn.q_proj" in message
    assert "blocks.0.unknown" in message
    assert "blocks.0.mlp.layer1" in message
    assert len(captured.value.issues) == 6


@pytest.mark.parametrize(
    ("weights", "expected_fragments"),
    [
        (
            object(),
            ("adapter weights must be a mapping",),
        ),
        (
            {
                "target.lora_A.weight": object(),
                "target.lora_B.weight": torch.ones((2, 2)),
            },
            ("target.lora_A.weight", "must be a tensor"),
        ),
        (
            {
                "target.lora_A.weight": torch.ones((2, 2, 1)),
                "target.lora_B.weight": torch.ones((2, 2)),
            },
            ("target.lora_A.weight", "rank 2"),
        ),
        (
            {
                "target.lora_A.weight": torch.ones((2, 2), dtype=torch.int64),
                "target.lora_B.weight": torch.ones((2, 2), dtype=torch.int64),
            },
            ("floating point",),
        ),
        (
            {
                "target.lora_A.weight": torch.tensor([[float("nan"), 0.0]]),
                "target.lora_B.weight": torch.ones((2, 1)),
            },
            ("finite values",),
        ),
        (
            {
                "target.lora_A.weight": torch.ones((3, 2)),
                "target.lora_B.weight": torch.ones((2, 2)),
            },
            ("rank dimensions do not match",),
        ),
    ],
)
def test_standard_decoder_rejects_every_malformed_pair_field(
    weights: object,
    expected_fragments: tuple[str, ...],
) -> None:
    """Return complete typed format issues instead of a partial decoded target."""

    result = STANDARD_LORA_ADAPTER_DECODER.decode(weights)

    assert result.targets == ()
    message = "\n".join(f"{issue.key}: {issue.reason}" for issue in result.issues)
    assert all(fragment in message for fragment in expected_fragments)


def test_standard_decoder_preserves_deterministic_target_order() -> None:
    """Normalize mapping order while retaining original CPU tensor identities."""

    second_down = torch.ones((2, 3))
    second_up = torch.ones((4, 2))
    first_down = torch.ones((1, 2))
    first_up = torch.ones((3, 1))
    weights = {
        "z.lora_B.weight": second_up,
        "a.lora_B.weight": first_up,
        "z.lora_A.weight": second_down,
        "a.lora_A.weight": first_down,
    }

    decoded = STANDARD_LORA_ADAPTER_DECODER.decode(weights)

    assert decoded.issues == ()
    assert [target.target for target in decoded.targets] == ["a", "z"]
    assert decoded.targets[0].down is first_down
    assert decoded.targets[0].up is first_up
    assert decoded.targets[1].down is second_down
    assert decoded.targets[1].up is second_up


def test_standard_decoder_preserves_sd_scripts_alpha_as_intrinsic_scale() -> None:
    """Normalize down/up pairs while retaining their scalar alpha/rank meaning."""

    down = torch.ones((8, 16), dtype=torch.bfloat16)
    up = torch.ones((32, 8), dtype=torch.bfloat16)
    alpha = torch.tensor(4.0, dtype=torch.bfloat16)

    decoded = STANDARD_LORA_ADAPTER_DECODER.decode(
        {
            "target.lora_down.weight": down,
            "target.lora_up.weight": up,
            "target.alpha": alpha,
        }
    )

    assert decoded.issues == ()
    assert len(decoded.targets) == 1
    assert decoded.targets[0].down is down
    assert decoded.targets[0].up is up
    assert decoded.targets[0].rank == 8
    assert decoded.targets[0].intrinsic_scale == 0.5
