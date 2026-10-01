# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Safeguard the strict live-generation completion gate for inversion promotion."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from tools import check_inversion_experiment_parity as parity


def _image(
    path: Path,
    *,
    color: tuple[int, int, int, int] = (40, 60, 80, 255),
    size: tuple[int, int] = (3, 2),
    metadata: str = "",
) -> Path:
    """Write a bounded lossless image fixture with optional non-pixel metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("prompt", metadata)
    Image.new("RGBA", size, color).save(path, pnginfo=png_info)
    return path


def _matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Provide all six synthetic outputs while fingerprinting their references."""

    references, candidates = tmp_path / "references", tmp_path / "candidates"
    fingerprints: dict[str, str] = {}
    for case in ("anima", "krea", "klein"):
        reference = _image(references / case / "half2_full1" / "save_final.png")
        fingerprints[case] = hashlib.sha256(reference.read_bytes()).hexdigest()
        for interface in parity.INTERFACES:
            _image(candidates / interface / case / "save_final.png", metadata=interface)
    monkeypatch.setattr(parity, "REFERENCE_SHA256", fingerprints)
    return references, candidates


def test_metadata_differences_do_not_change_decoded_pixel_exactness(
    tmp_path: Path,
) -> None:
    """Compare generated content, not PNG compression or workflow metadata."""

    reference = _image(tmp_path / "reference.png", metadata="experimental workflow")
    candidate = _image(tmp_path / "candidate.png", metadata="production workflow")
    assert reference.read_bytes() != candidate.read_bytes()
    observation = parity.compare_images(reference, candidate)
    assert observation.exact
    assert observation.changed_pixels == 0
    assert observation.maximum_channel_difference == 0


@pytest.mark.parametrize("channel", range(4))
def test_one_changed_channel_in_one_pixel_fails_without_tolerance(
    tmp_path: Path, channel: int
) -> None:
    """Reject a single intensity-level change, including alpha-only changes."""

    reference = _image(tmp_path / "reference.png")
    candidate = _image(tmp_path / "candidate.png")
    with Image.open(candidate) as source:
        changed = source.copy()
    original_pixel = changed.getpixel((0, 0))
    assert isinstance(original_pixel, tuple)
    pixel = list(original_pixel)
    pixel[channel] -= 1
    changed.putpixel((0, 0), tuple(pixel))
    changed.save(candidate)
    observation = parity.compare_images(reference, candidate)
    assert not observation.exact
    assert observation.changed_pixels == 1
    assert observation.maximum_channel_difference == 1


def test_dimension_mismatch_is_not_resized_into_a_match(tmp_path: Path) -> None:
    """Reject equal colors at different dimensions rather than resampling them."""

    observation = parity.compare_images(
        _image(tmp_path / "reference.png"),
        _image(tmp_path / "candidate.png", size=(6, 4)),
    )
    assert not observation.exact
    assert observation.changed_pixels is None


def test_reference_file_cannot_be_submitted_as_production_output(
    tmp_path: Path,
) -> None:
    """Require a separate candidate file instead of comparing a reference to itself."""

    reference = _image(tmp_path / "reference.png")
    with pytest.raises(ValueError, match="must not be the experimental reference"):
        parity.compare_images(reference, reference)


@pytest.mark.parametrize("mode", ("L", "I;16"))
def test_other_bit_depths_are_not_silently_quantized_for_comparison(
    tmp_path: Path, mode: str
) -> None:
    """Reject formats whose conversion could hide differences in decoded pixels."""

    reference = _image(tmp_path / "reference.png")
    candidate = tmp_path / "candidate.png"
    Image.new(mode, (3, 2), 300 if mode == "I;16" else 40).save(candidate)
    with pytest.raises(ValueError, match="lossless 8-bit RGB or RGBA PNGs"):
        parity.compare_images(reference, candidate)


def test_gate_requires_every_case_through_both_interfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require a six-output matrix rather than one successful showcase generation."""

    references, candidates = _matrix(tmp_path, monkeypatch)
    observations = parity.check_experiment_parity(
        experimental_root=references, generated_root=candidates
    )
    assert [(case, interface) for case, interface, _ in observations] == [
        (case, interface)
        for case in ("anima", "krea", "klein")
        for interface in ("convenience", "options_stack")
    ]
    assert all(observation.exact for _, _, observation in observations)


@pytest.mark.parametrize("failure", ("missing", "changed", "reference"))
def test_cli_blocks_completion_on_missing_or_changed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Return failure for incomplete generation, pixel differences or baseline edits."""

    references, candidates = _matrix(tmp_path, monkeypatch)
    candidate = candidates / "options_stack" / "klein" / "save_final.png"
    if failure == "missing":
        candidate.unlink()
    elif failure == "changed":
        _image(candidate, color=(41, 60, 80, 255))
    else:
        _image(
            references / "anima" / "half2_full1" / "save_final.png", metadata="changed"
        )
    assert (
        parity.main(
            [
                "--experimental-root",
                str(references),
                "--generated-root",
                str(candidates),
            ]
        )
        == 1
    )


def test_cli_passes_only_the_complete_exact_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report zero changed pixels for all six required output comparisons."""

    references, candidates = _matrix(tmp_path, monkeypatch)
    assert (
        parity.main(
            [
                "--experimental-root",
                str(references),
                "--generated-root",
                str(candidates),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert output.count("PIXEL-EXACT:") == 6
    assert output.count("changed pixels=0;") == 6
