# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify accepted Anima artifact validation fails closed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.anima_regression_oracle.artifact_validation import (
    AcceptedArtifactValidator,
)
from tools.anima_regression_oracle.manifest import (
    AcceptedImage,
    AnimaRegressionManifest,
    CompletedJsonEvidence,
    OracleCommand,
    PerformanceEvidence,
)


def test_validator_requires_hashes_cleanup_counts_targets_and_performance(
    tmp_path: Path,
) -> None:
    """Accept only a complete baseline across every evidence authority."""

    manifest = _manifest(tmp_path)

    observations = AcceptedArtifactValidator().validate(manifest)

    assert [item.identity for item in observations] == [
        "accepted-image",
        "managed-result",
        "accepted-performance-gate",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("image", "Accepted image hash changed"),
        ("cleanup", "Managed cleanup failed"),
        ("target", "target_count=448"),
        ("performance", "performance result does not pass"),
    ],
)
def test_validator_rejects_changed_accepted_evidence(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Report the exact blocking evidence dimension that regressed."""

    manifest = _manifest(tmp_path)
    if mutation == "image":
        manifest.images[0].path.write_bytes(b"changed")
    elif mutation == "cleanup":
        _write_json(
            manifest.managed_results[0].path,
            {"status": "completed", "cleanup_verified": False},
        )
    elif mutation == "target":
        _write_json(
            manifest.managed_results[0].detail_paths[0],
            {"model_call_count": 30, "target_count": 447},
        )
    else:
        payload = json.loads(manifest.performance.path.read_text(encoding="utf-8"))
        payload["passed"] = False
        _write_json(manifest.performance.path, payload)

    with pytest.raises(ValueError, match=message):
        AcceptedArtifactValidator().validate(manifest)


def _manifest(root: Path) -> AnimaRegressionManifest:
    """Write the smallest complete accepted-evidence fixture."""

    image = root / "accepted.png"
    image.write_bytes(b"accepted-image")
    result = root / "result.json"
    detail = root / "history.json"
    performance = root / "performance.json"
    _write_json(
        result,
        {
            "status": "completed",
            "cleanup_verified": True,
            "observations": [{"case_id": "one"}],
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        },
    )
    _write_json(detail, {"model_call_count": 30, "target_count": 448})
    _write_json(
        performance,
        {
            "passed": True,
            "environment": {"adapter_target_count": 448},
            "profiles": [
                {
                    "profile_id": "attention-only",
                    "maximum_overhead_percent": 0.0,
                    "passed": True,
                },
                {
                    "profile_id": "regional-lora-1",
                    "maximum_overhead_percent": 15.0,
                    "passed": True,
                },
                {
                    "profile_id": "regional-lora-4",
                    "maximum_overhead_percent": 35.0,
                    "passed": True,
                },
            ],
        },
    )
    empty = OracleCommand("empty", ("empty",))
    return AnimaRegressionManifest(
        images=(
            AcceptedImage(
                "accepted-image", image, hashlib.sha256(image.read_bytes()).hexdigest()
            ),
        ),
        managed_results=(
            CompletedJsonEvidence(
                "managed-result",
                result,
                detail_paths=(detail,),
                observation_count=1,
                required_integer_occurrences=(
                    ("model_call_count", 30, 1),
                    ("target_count", 448, 1),
                ),
                required_image_hashes=(hashlib.sha256(image.read_bytes()).hexdigest(),),
            ),
        ),
        performance=PerformanceEvidence(
            performance,
            448,
            (
                ("attention-only", 0.0),
                ("regional-lora-1", 15.0),
                ("regional-lora-4", 35.0),
            ),
        ),
        model_visibility_inventory=root / "model-visibility.json",
        focused_command=empty,
        repository_commands=(),
        managed_rerun_commands=(),
    )


def _write_json(path: Path, payload: object) -> None:
    """Write one deterministic fixture object."""

    path.write_text(json.dumps(payload), encoding="utf-8")
