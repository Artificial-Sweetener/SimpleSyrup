# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for grounded SAM model catalog metadata."""

from __future__ import annotations

import pytest

from simple_syrup.runtime.model_catalog import (
    ANIMA_QWEN_TEXT_ENCODER,
    ANIMA_QWEN_VAE,
    BERT_ENTRY,
    GROUNDING_DINO_ENTRIES,
    SAM_ENTRIES,
    get_grounding_dino_entry,
    get_sam_entry,
    grounding_dino_choices,
    sam_choices,
)


def test_sam_catalog_exposes_layerstyle_compatible_models() -> None:
    """Known SAM choices match LayerStyle-compatible labels and filenames."""

    choices = sam_choices()

    assert "sam_hq_vit_b (379MB)" in choices
    entry = get_sam_entry("sam_hq_vit_b (379MB)")
    assert entry.entry_id == "sam_hq_vit_b"
    assert entry.artifacts[0].filename == "sam_hq_vit_b.pth"
    assert entry.artifacts[0].folder_name == "sams"
    assert entry.artifacts[0].source_url.startswith("https://huggingface.co/")


def test_grounding_dino_catalog_has_required_artifact_pairs() -> None:
    """GroundingDINO entries carry config and checkpoint artifacts."""

    entry = get_grounding_dino_entry("GroundingDINO_SwinT_OGC (694MB)")
    filenames = {artifact.filename for artifact in entry.artifacts}

    assert "GroundingDINO_SwinT_OGC.cfg.py" in filenames
    assert "groundingdino_swint_ogc.pth" in filenames
    assert all(artifact.folder_name == "grounding-dino" for artifact in entry.artifacts)


def test_catalog_choices_are_deterministic() -> None:
    """Dropdown choices preserve catalog declaration order."""

    assert sam_choices() == [entry.display_name for entry in SAM_ENTRIES]
    assert grounding_dino_choices() == [
        entry.display_name for entry in GROUNDING_DINO_ENTRIES
    ]


def test_catalog_lookup_rejects_unknown_selection() -> None:
    """Unknown model selections fail with actionable context."""

    with pytest.raises(ValueError, match="Unknown SAM model"):
        get_sam_entry("not a model")


def test_bert_catalog_has_huggingface_snapshot_artifacts() -> None:
    """BERT metadata points at the expected Hugging Face source."""

    filenames = {artifact.filename for artifact in BERT_ENTRY.artifacts}

    assert BERT_ENTRY.source_repo == "google-bert/bert-base-uncased"
    assert {
        "config.json",
        "tokenizer.json",
        "vocab.txt",
        "model.safetensors",
    } <= filenames


def test_anima_catalog_has_trusted_auto_artifacts() -> None:
    """Anima auto artifacts declare canonical folders and checksums."""

    assert ANIMA_QWEN_TEXT_ENCODER.folder_name == "text_encoders"
    assert ANIMA_QWEN_TEXT_ENCODER.filename == "qwen_3_06b_base.safetensors"
    assert ANIMA_QWEN_TEXT_ENCODER.canonical_subfolder == "qwen"
    assert ANIMA_QWEN_TEXT_ENCODER.source_url.endswith(
        "/split_files/text_encoders/qwen_3_06b_base.safetensors"
    )
    assert len(ANIMA_QWEN_TEXT_ENCODER.sha256) == 64

    assert ANIMA_QWEN_VAE.folder_name == "vae"
    assert ANIMA_QWEN_VAE.filename == "qwen_image_vae.safetensors"
    assert ANIMA_QWEN_VAE.canonical_subfolder == "qwen"
    assert ANIMA_QWEN_VAE.source_url.endswith(
        "/split_files/vae/qwen_image_vae.safetensors"
    )
    assert len(ANIMA_QWEN_VAE.sha256) == 64
