# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify P10.3 opaque staging, score sealing, and unblinding."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.manifest_types import (
    BenchmarkManifest,
    JsonObject,
)
from tools.regional_visual_benchmark.blind import BlindCorpusStager
from tools.regional_visual_benchmark.matrix import visual_positions
from tools.regional_visual_benchmark.scoring import BlindScoreLedger, VisualScore
from tools.regional_visual_benchmark.unblind import unblind_scores


def test_blind_packet_hides_strategy_until_complete_scores_seal(
    tmp_path: Path,
) -> None:
    """Keep source identity private, score all opaque IDs, then join exactly once."""

    manifest = _minimal_manifest()
    corpus_path = _write_corpus(tmp_path, manifest)
    identities = iter(f"opaque{index:02d}" for index in range(10))
    staged = BlindCorpusStager(
        manifest, identity_factory=lambda: next(identities)
    ).stage(corpus_path)
    packet = _read(staged.packet_path)
    packet_text = staged.packet_path.read_text(encoding="utf-8")

    assert packet["entry_count"] == 10
    assert "regional_conditioning" not in packet_text
    assert "attention_coupling" not in packet_text
    assert all(
        position.artifact_id not in packet_text
        for position in visual_positions(manifest)
    )
    assert hashlib.sha256(staged.mapping_path.read_bytes()).hexdigest() == (
        staged.mapping_sha256
    )

    ledger = BlindScoreLedger(staged.packet_path)
    entries = cast(list[object], packet["entries"])
    first_id = str(cast(JsonObject, entries[0])["opaque_id"])
    ledger.record(first_id, _score())
    with pytest.raises(ValueError, match="already scored"):
        ledger.record(first_id, _score())
    for value in entries[1:]:
        ledger.record(str(cast(JsonObject, value)["opaque_id"]), _score())
    assert ledger.remaining_count == 0
    sealed = ledger.seal()

    unblinded = unblind_scores(
        manifest,
        corpus_path=corpus_path,
        packet_path=staged.packet_path,
        mapping_path=staged.mapping_path,
        sealed_scores_path=sealed,
    )
    result = _read(unblinded)
    assert result["observation_count"] == 10
    observations = cast(list[object], result["observations"])
    assert {cast(JsonObject, value)["strategy"] for value in observations} == {
        "regional_conditioning",
        "attention_coupling",
    }


def test_severe_failure_definition_is_fixed_and_validated() -> None:
    """Classify named structural failures even when scalar severity is low."""

    assert not _score().severe_failure
    assert replace(_score(), failure_severity=3).severe_failure
    assert replace(_score(), anatomical_integrity=0).severe_failure
    assert replace(_score(), failure_classes=("duplicate_subject",)).severe_failure
    with pytest.raises(ValueError, match="unknown failure"):
        replace(_score(), failure_classes=("invented",))
    with pytest.raises(ValueError, match="0 to 4"):
        replace(_score(), style_consistency=5)


def test_incomplete_scores_cannot_seal(tmp_path: Path) -> None:
    """Keep every configured image mandatory before strategy identity is exposed."""

    manifest = _minimal_manifest()
    corpus_path = _write_corpus(tmp_path, manifest)
    identities = iter(f"opaque{index:02d}" for index in range(10))
    staged = BlindCorpusStager(
        manifest, identity_factory=lambda: next(identities)
    ).stage(corpus_path)

    with pytest.raises(ValueError, match="remaining=10"):
        BlindScoreLedger(staged.packet_path).seal()


def _minimal_manifest() -> BenchmarkManifest:
    """Return one case and one seed while preserving the five fixed profiles."""

    manifest = load_manifest()
    return replace(
        manifest,
        cases=(manifest.cases[0],),
        sampling=replace(manifest.sampling, seeds=(1,)),
    )


def _write_corpus(root: Path, manifest: BenchmarkManifest) -> Path:
    """Write one complete hash-valid synthetic corpus record."""

    images = root / "images"
    images.mkdir()
    observations: list[JsonObject] = []
    for index, position in enumerate(visual_positions(manifest)):
        value = f"image-{index}".encode()
        relative = Path("images") / f"image-{index}.png"
        (root / relative).write_bytes(value)
        observations.append(
            {
                "artifact_id": position.artifact_id,
                "case_id": position.case_id,
                "seed": position.seed,
                "strategy": position.strategy.value,
                "spatial_profile": position.spatial_profile.value,
                "image_file": relative.as_posix(),
                "image_sha256": hashlib.sha256(value).hexdigest(),
            }
        )
    path = root / "p10.3-corpus.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_id": manifest.benchmark_id,
                "status": "completed",
                "observations": observations,
            }
        ),
        encoding="utf-8",
    )
    return path


def _score() -> VisualScore:
    """Return one complete no-failure score."""

    return VisualScore(
        subject_count=4,
        identity_separation=4,
        attribute_isolation=4,
        anatomical_integrity=4,
        pose_continuity=4,
        boundary_integrity=4,
        global_composition=4,
        style_consistency=4,
        failure_severity=0,
    )


def _read(path: Path) -> JsonObject:
    """Read one test JSON object."""

    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(JsonObject, value)
