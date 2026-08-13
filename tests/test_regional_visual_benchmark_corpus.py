# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify complete P10.3 corpus persistence and source pairing."""

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.manifest_types import BenchmarkManifest
from tools.attention_coupling_benchmark.results import (
    CompletedOutputs,
    ProbeMetrics,
)
from tools.comfy_api import ImageReference
from tools.regional_visual_benchmark.corpus import VisualCorpusRecorder
from tools.regional_visual_benchmark.matrix import source_positions, visual_positions
from tools.regional_visual_benchmark.workflow import BuiltVisualWorkflow


def test_complete_minimal_matrix_publishes_with_paired_source_hashes(
    tmp_path: Path,
) -> None:
    """Publish all positions only after their exact neutral source exists."""

    manifest = _minimal_manifest()
    recorder = VisualCorpusRecorder(tmp_path, manifest)
    source = source_positions(manifest)[0]
    source_bytes = _png((1024, 1024), "purple")
    recorder.record_source(
        source,
        _workflow(),
        _outputs(),
        history={},
        prompt_id="source-prompt",
        image_bytes=source_bytes,
    )
    full_bytes = _png((1024, 1024), "red")
    refinement_bytes = _png((1536, 1536), "blue")
    for position in visual_positions(manifest):
        recorder.record_visual(
            position,
            _workflow(),
            _outputs(),
            history={},
            prompt_id=position.artifact_id,
            image_bytes=refinement_bytes if position.is_refinement else full_bytes,
            mask_sha256s=(),
        )

    result = recorder.finalize(system_stats={"gpu": "test"}, cleanup_verified=True)

    assert result.is_file()
    assert not (tmp_path / "p10.3-corpus.inprogress.json").exists()
    assert len(recorder.completed_artifact_ids) == 10
    resumed = VisualCorpusRecorder(tmp_path, manifest)
    assert resumed.completed_artifact_ids == frozenset()


def test_refinement_cannot_record_before_its_source(tmp_path: Path) -> None:
    """Reject an output whose paired neutral source is unavailable."""

    manifest = _minimal_manifest()
    position = next(
        value for value in visual_positions(manifest) if value.is_refinement
    )
    recorder = VisualCorpusRecorder(tmp_path, manifest)

    with pytest.raises(ValueError, match="source was not recorded"):
        recorder.record_visual(
            position,
            _workflow(),
            _outputs(),
            history={},
            prompt_id="prompt",
            image_bytes=_png((1536, 1536), "blue"),
            mask_sha256s=(),
        )


def test_finalize_rejects_incomplete_corpus(tmp_path: Path) -> None:
    """Keep missing images blocking even when process cleanup succeeded."""

    manifest = _minimal_manifest()
    recorder = VisualCorpusRecorder(tmp_path, manifest)

    with pytest.raises(ValueError, match="source corpus is incomplete"):
        recorder.finalize(system_stats={}, cleanup_verified=True)


def _minimal_manifest() -> BenchmarkManifest:
    """Return one global-only case and one seed through the typed manifest."""

    manifest = load_manifest()
    global_only = next(case for case in manifest.cases if not case.masks)
    return replace(
        manifest,
        cases=(global_only,),
        sampling=replace(manifest.sampling, seeds=(1,)),
    )


def _workflow() -> BuiltVisualWorkflow:
    """Return the smallest recorder-compatible workflow identity."""

    return BuiltVisualWorkflow({}, "1", "2")


def _outputs() -> CompletedOutputs:
    """Return deterministic parsed probe evidence."""

    return CompletedOutputs(
        ProbeMetrics(runtime_ms=1.0, peak_vram_bytes=2, model_call_count=3),
        ImageReference("image.png", "", "output"),
    )


def _png(size: tuple[int, int], color: str) -> bytes:
    """Encode one deterministic RGB PNG fixture."""

    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()
