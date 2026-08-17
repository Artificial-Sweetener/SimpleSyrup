# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify fail-closed materialization parity history decoding."""

from __future__ import annotations

import pytest

from tools.sdxl_materialization_parity.results import (
    decode_materialization_parity,
)


def _payload() -> dict[str, object]:
    """Return one model-identity-neutral parity result fixture."""

    return {
        "run_id": "run-1",
        "variant_count": 1,
        "element_count": 8,
        "differing_element_count": 0,
        "peak_vram_bytes": 1024,
        "exact": True,
        "variants": [{"region_index": 0}],
    }


def test_decoder_returns_one_complete_matching_result() -> None:
    """Preserve the terminal object after validating its required evidence."""

    payload = _payload()

    result = decode_materialization_parity(
        {"outputs": {"9": {"materialization_parity": [payload]}}},
        terminal_node_id="9",
        expected_run_id="run-1",
    )

    assert result == payload


@pytest.mark.parametrize(
    "payload",
    [
        {**_payload(), "run_id": "wrong"},
        {**_payload(), "exact": "yes"},
        {**_payload(), "variant_count": 2},
        {**_payload(), "differing_element_count": 9},
        {**_payload(), "element_count": 0},
    ],
)
def test_decoder_rejects_incomplete_or_inconsistent_evidence(
    payload: dict[str, object],
) -> None:
    """Fail before publishing malformed benchmark evidence."""

    with pytest.raises(ValueError):
        decode_materialization_parity(
            {"outputs": {"9": {"materialization_parity": [payload]}}},
            terminal_node_id="9",
            expected_run_id="run-1",
        )
