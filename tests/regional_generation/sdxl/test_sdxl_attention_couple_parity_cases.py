# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify external model-neutral prompt-case loading for parity diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.sdxl_attention_couple_parity.cases import (
    SdxlAttentionCoupleParityCase,
    load_parity_case,
)


def test_loader_requires_and_preserves_every_prompt_field(tmp_path: Path) -> None:
    """Keep external diagnostic prompts complete and ordered by the case type."""

    fields = tuple(SdxlAttentionCoupleParityCase.__dataclass_fields__)
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps({field: f"generic {field}" for field in fields}),
        encoding="utf-8",
    )

    loaded = load_parity_case(path)

    assert tuple(getattr(loaded, field) for field in fields) == tuple(
        f"generic {field}" for field in fields
    )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"base_positive_g": "incomplete"},
        {
            field: "" if index == 0 else "value"
            for index, field in enumerate(
                SdxlAttentionCoupleParityCase.__dataclass_fields__
            )
        },
    ],
)
def test_loader_rejects_incomplete_or_empty_contracts(
    tmp_path: Path,
    payload: object,
) -> None:
    """Reject diagnostic cases that cannot establish all parity controls."""

    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        load_parity_case(path)
