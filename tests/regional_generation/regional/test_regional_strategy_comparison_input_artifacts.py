# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact P10.2 Comfy input-artifact ownership."""

import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.regional_strategy_comparison.input_artifacts import (
    ComparisonInputArtifacts,
)
from tools.regional_strategy_comparison.matrix import MASK_CASE_ID


def _png(width: int, height: int) -> bytes:
    """Return one compact deterministic RGB PNG."""

    stream = BytesIO()
    Image.new("RGB", (width, height), color=(20, 40, 60)).save(stream, format="PNG")
    return stream.getvalue()


def test_input_artifacts_publish_evidence_and_remove_only_comfy_inputs(
    tmp_path: Path,
) -> None:
    """Retain durable evidence while cleaning exact run-owned input files."""

    root = tmp_path
    input_root = root / "input"
    evidence_root = root / "evidence"
    input_root.mkdir()
    evidence_root.mkdir()
    case = next(case for case in load_manifest().cases if case.case_id == MASK_CASE_ID)
    lifecycle = ComparisonInputArtifacts(input_root, evidence_root, "RUN-1")
    source = _png(1024, 1024)

    with lifecycle:
        masks = lifecycle.materialize_masks(case)
        assert len(masks.full) == len(masks.refinement) == 2
        assert all(
            (input_root / name).is_file() for name in (*masks.full, *masks.refinement)
        )
        assert lifecycle.publish_source(source) == hashlib.sha256(source).hexdigest()
        assert (input_root / lifecycle.source_name).is_file()

    assert lifecycle.cleanup_verified is True
    assert (evidence_root / "shared-source.png").read_bytes() == source
    assert len(tuple((evidence_root / "masks").rglob("*.png"))) == 4
