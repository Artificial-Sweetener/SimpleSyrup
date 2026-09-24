# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify shared external prompt adaptation for SDXL visual proofs."""

from __future__ import annotations

import json
from pathlib import Path

from tools.sdxl_attention_couple_parity.cases import (
    SdxlAttentionCoupleParityCase,
)
from tools.sdxl_attention_coupling_integration.visual_prompt_fixture import (
    load_visual_prompt_set,
)


def test_visual_prompt_fixture_preserves_every_authored_field(tmp_path: Path) -> None:
    """Map a complete external parity fixture without defaults or reordering."""

    fields = tuple(SdxlAttentionCoupleParityCase.__dataclass_fields__)
    path = tmp_path / "prompt-case.json"
    values = {field: f"anonymous {field}" for field in fields}
    path.write_text(json.dumps(values), encoding="utf-8")

    prompts = load_visual_prompt_set(path)

    assert tuple(getattr(prompts, field) for field in fields) == tuple(
        values[field] for field in fields
    )
