# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify P10.3 runner graph coverage and scoring CLI lifecycle."""

import json
from pathlib import Path

import pytest

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.regional_visual_benchmark.execution import required_node_ids
from tools.regional_visual_benchmark.matrix import source_positions, visual_positions
from tools.regional_visual_benchmark.workflow import VisualBenchmarkWorkflowBuilder
from tools.score_regional_visual_benchmark import main as score_main


def test_required_nodes_cover_every_public_strategy_profile_sampler() -> None:
    """Verify managed readiness against every graph path before a long run."""

    manifest = load_manifest()
    required = required_node_ids(
        manifest,
        VisualBenchmarkWorkflowBuilder(),
        source_positions(manifest),
        visual_positions(manifest),
        "run-1",
    )

    assert {
        "SimpleSyrup.KSamplerPromptByRegion",
        "SimpleSyrup.KSamplerPromptByTiledRegion",
        "SimpleSyrup.KSamplerContextualDiffusion",
        "SimpleSyrup.KSamplerAttentionCoupling",
        "SimpleSyrup.KSamplerAttentionCouplingTiled",
        "SimpleSyrup.KSamplerAttentionCouplingContextual",
        "KSampler",
        "LoadImage",
        "ImageScale",
        "VAEEncode",
    } <= required


def test_scoring_cli_records_and_seals_one_complete_packet(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Map all nine CLI fields into one validated sealed score record."""

    packet = tmp_path / "packet.json"
    packet.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_id": "benchmark",
                "strategy_blind": True,
                "entries": [{"opaque_id": "opaque01"}],
            }
        ),
        encoding="utf-8",
    )
    assert score_main(["next", "--packet", str(packet)]) == 0
    assert "opaque01" in capsys.readouterr().out
    arguments = [
        "record",
        "--packet",
        str(packet),
        "--opaque-id",
        "opaque01",
        "--subject-count",
        "4",
        "--identity-separation",
        "4",
        "--attribute-isolation",
        "3",
        "--anatomical-integrity",
        "4",
        "--pose-continuity",
        "3",
        "--boundary-integrity",
        "4",
        "--global-composition",
        "4",
        "--style-consistency",
        "3",
        "--failure-severity",
        "0",
    ]
    assert score_main(arguments) == 0
    assert '"remaining": 0' in capsys.readouterr().out
    assert score_main(["seal", "--packet", str(packet)]) == 0
    sealed = Path(capsys.readouterr().out.strip())
    assert sealed.is_file()
