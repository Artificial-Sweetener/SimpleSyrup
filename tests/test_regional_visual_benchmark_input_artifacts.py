"""Verify exact P10.3 Comfy input ownership and cleanup."""

from pathlib import Path

import pytest
from PIL import Image

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.regional_visual_benchmark.input_artifacts import VisualPositionInputs
from tools.regional_visual_benchmark.matrix import visual_positions


def test_refinement_inputs_publish_exact_source_and_masks(tmp_path: Path) -> None:
    """Publish 1536 masks and a byte-identical 1024 source, then remove both."""

    manifest = load_manifest()
    position = next(
        value
        for value in visual_positions(manifest)
        if value.is_refinement and value.case_id == "vertical-hard-50-50"
    )
    case = next(value for value in manifest.cases if value.case_id == position.case_id)
    source = tmp_path / "source.png"
    Image.new("RGB", (1024, 1024), "purple").save(source)

    owner = VisualPositionInputs(
        tmp_path,
        "run-1",
        position,
        case,
        source_path=source,
    )
    with owner as inputs:
        assert len(inputs.mask_names) == 2
        assert len(inputs.mask_sha256s) == 2
        assert inputs.source_name is not None
        assert inputs.source_sha256 is not None
        assert (tmp_path / inputs.source_name).read_bytes() == source.read_bytes()
        with Image.open(tmp_path / inputs.mask_names[0]) as mask:
            assert mask.size == (1536, 1536)
        assert not inputs.cleanup_verified

    assert owner.cleanup_verified


def test_full_inputs_reject_a_source_file(tmp_path: Path) -> None:
    """Keep full generation from noise by failing before publishing inputs."""

    manifest = load_manifest()
    position = next(
        value for value in visual_positions(manifest) if not value.is_refinement
    )
    case = next(value for value in manifest.cases if value.case_id == position.case_id)
    source = tmp_path / "source.png"
    Image.new("RGB", (1024, 1024), "black").save(source)

    with pytest.raises(ValueError, match="does not match"):
        with VisualPositionInputs(
            tmp_path,
            "run-1",
            position,
            case,
            source_path=source,
        ):
            pass


def test_refinement_rejects_wrong_source_dimensions(tmp_path: Path) -> None:
    """Fail closed before a malformed refinement source reaches Comfy."""

    manifest = load_manifest()
    position = next(
        value for value in visual_positions(manifest) if value.is_refinement
    )
    case = next(value for value in manifest.cases if value.case_id == position.case_id)
    source = tmp_path / "source.png"
    Image.new("RGB", (512, 512), "black").save(source)

    with pytest.raises(ValueError, match="exactly 1024x1024"):
        with VisualPositionInputs(
            tmp_path,
            "run-1",
            position,
            case,
            source_path=source,
        ):
            pass
