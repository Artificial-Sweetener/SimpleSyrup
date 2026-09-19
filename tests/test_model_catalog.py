# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for grounded SAM model catalog metadata."""

from __future__ import annotations

import pytest

from simple_syrup.runtime.anima_artifacts import (
    ANIMA_QWEN_TEXT_ENCODER,
    ANIMA_QWEN_VAE,
)
from simple_syrup.runtime.model_catalog import (
    BERT_ENTRY,
    GROUNDING_DINO_ENTRIES,
    SAM_ENTRIES,
    ULTRALYTICS_ENTRIES,
    get_grounding_dino_entry,
    get_sam_entry,
    get_ultralytics_entry,
    grounding_dino_choices,
    sam_choices,
    ultralytics_choices,
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
    assert ultralytics_choices() == [
        entry.display_name for entry in ULTRALYTICS_ENTRIES
    ]


def test_ultralytics_catalog_has_pinned_verified_anzhc_checkpoints() -> None:
    """Curated Anzhc models are revision-pinned, verified, and task-foldered."""

    anzhc_entries = tuple(
        entry
        for entry in ULTRALYTICS_ENTRIES
        if entry.source_repo == "Anzhc/Anzhcs_YOLOs"
    )

    assert len(anzhc_entries) == 15
    assert all(entry.source_repo == "Anzhc/Anzhcs_YOLOs" for entry in anzhc_entries)
    assert all(len(entry.artifacts) == 1 for entry in anzhc_entries)
    assert all(
        artifact.source_url.startswith(
            "https://huggingface.co/Anzhc/Anzhcs_YOLOs/resolve/"
            "f5a2306d7fed4f3cfc26c25ff1ab2e3f3cfce855/"
        )
        for entry in anzhc_entries
        for artifact in entry.artifacts
    )
    assert all(
        artifact.folder_name == "ultralytics_segm"
        and artifact.sha256 is not None
        and len(artifact.sha256) == 64
        for entry in anzhc_entries
        for artifact in entry.artifacts
    )
    assert all(
        "Drones" not in artifact.filename
        and "Score" not in artifact.filename
        and "Breast size" not in artifact.filename
        for entry in anzhc_entries
        for artifact in entry.artifacts
    )


def test_ultralytics_catalog_has_verified_adetailer_and_anime_models() -> None:
    """ADetailer and anime face checkpoints have compatible curated metadata."""

    assert len(ULTRALYTICS_ENTRIES) == 22

    face = get_ultralytics_entry("bingsu_face_yolov8n_v2")
    hand = get_ultralytics_entry("bingsu_hand_yolov8s")
    person = get_ultralytics_entry("bingsu_person_yolov8s_seg")
    anime_face = get_ultralytics_entry("fuyucchi_yolov8x6_animeface")

    assert face.artifacts[0].folder_name == "ultralytics_bbox"
    assert hand.artifacts[0].folder_name == "ultralytics_bbox"
    assert person.artifacts[0].folder_name == "ultralytics_segm"
    assert anime_face.artifacts[0].folder_name == "ultralytics_bbox"
    assert face.source_repo == "Bingsu/adetailer"
    assert face.license_note == "Apache-2.0"
    assert (
        "/resolve/53cc19de382014514d9d4038601d261a7faa9b7b/"
        in face.artifacts[0].source_url
    )
    assert anime_face.source_repo == "Fuyucchi/yolov8_animeface"
    assert anime_face.license_note == "AGPL-3.0"
    assert "/resolve/b0841ce930453c0f23ceb8086d6554c17de5fe4a/" in (
        anime_face.artifacts[0].source_url
    )
    assert all(
        artifact.sha256 is not None and len(artifact.sha256) == 64
        for entry in (face, hand, person, anime_face)
        for artifact in entry.artifacts
    )


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
