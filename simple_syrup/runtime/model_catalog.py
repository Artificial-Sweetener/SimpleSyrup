# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Known model metadata for grounded SAM masking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelFamily(StrEnum):
    """Catalog families used by grounded SAM model selection."""

    SAM = "sam"
    GROUNDING_DINO = "grounding_dino"
    TEXT_ENCODER = "text_encoder"
    VITMATTE = "vitmatte"
    WD14_TAGGER = "wd14_tagger"


@dataclass(frozen=True)
class ModelArtifact:
    """A downloadable file required by a known model entry."""

    artifact_id: str
    filename: str
    folder_name: str
    source_url: str
    description: str


@dataclass(frozen=True)
class AutoModelArtifact:
    """A trusted auto-resolved model artifact with a canonical destination."""

    cache_id: str
    filename: str
    folder_name: str
    canonical_subfolder: str
    source_url: str
    source_repo: str
    description: str
    sha256: str


@dataclass(frozen=True)
class ModelEntry:
    """A known model selection and its source metadata."""

    entry_id: str
    display_name: str
    family: ModelFamily
    model_type: str
    artifacts: tuple[ModelArtifact, ...]
    source_repo: str
    auto_download_allowed: bool = True
    license_note: str = ""


SAM_ENTRIES: tuple[ModelEntry, ...] = (
    ModelEntry(
        entry_id="sam_vit_h",
        display_name="sam_vit_h (2.56GB)",
        family=ModelFamily.SAM,
        model_type="vit_h",
        source_repo="facebookresearch/segment-anything",
        artifacts=(
            ModelArtifact(
                artifact_id="sam_vit_h_checkpoint",
                filename="sam_vit_h_4b8939.pth",
                folder_name="sams",
                source_url=(
                    "https://dl.fbaipublicfiles.com/segment_anything/"
                    "sam_vit_h_4b8939.pth"
                ),
                description="SAM ViT-H checkpoint",
            ),
        ),
    ),
    ModelEntry(
        entry_id="sam_vit_l",
        display_name="sam_vit_l (1.25GB)",
        family=ModelFamily.SAM,
        model_type="vit_l",
        source_repo="facebookresearch/segment-anything",
        artifacts=(
            ModelArtifact(
                artifact_id="sam_vit_l_checkpoint",
                filename="sam_vit_l_0b3195.pth",
                folder_name="sams",
                source_url=(
                    "https://dl.fbaipublicfiles.com/segment_anything/"
                    "sam_vit_l_0b3195.pth"
                ),
                description="SAM ViT-L checkpoint",
            ),
        ),
    ),
    ModelEntry(
        entry_id="sam_vit_b",
        display_name="sam_vit_b (375MB)",
        family=ModelFamily.SAM,
        model_type="vit_b",
        source_repo="facebookresearch/segment-anything",
        artifacts=(
            ModelArtifact(
                artifact_id="sam_vit_b_checkpoint",
                filename="sam_vit_b_01ec64.pth",
                folder_name="sams",
                source_url=(
                    "https://dl.fbaipublicfiles.com/segment_anything/"
                    "sam_vit_b_01ec64.pth"
                ),
                description="SAM ViT-B checkpoint",
            ),
        ),
    ),
    ModelEntry(
        entry_id="sam_hq_vit_h",
        display_name="sam_hq_vit_h (2.57GB)",
        family=ModelFamily.SAM,
        model_type="sam_hq_vit_h",
        source_repo="lkeab/hq-sam",
        artifacts=(
            ModelArtifact(
                artifact_id="sam_hq_vit_h_checkpoint",
                filename="sam_hq_vit_h.pth",
                folder_name="sams",
                source_url=(
                    "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
                ),
                description="SAM-HQ ViT-H checkpoint",
            ),
        ),
    ),
    ModelEntry(
        entry_id="sam_hq_vit_l",
        display_name="sam_hq_vit_l (1.25GB)",
        family=ModelFamily.SAM,
        model_type="sam_hq_vit_l",
        source_repo="lkeab/hq-sam",
        artifacts=(
            ModelArtifact(
                artifact_id="sam_hq_vit_l_checkpoint",
                filename="sam_hq_vit_l.pth",
                folder_name="sams",
                source_url=(
                    "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
                ),
                description="SAM-HQ ViT-L checkpoint",
            ),
        ),
    ),
    ModelEntry(
        entry_id="sam_hq_vit_b",
        display_name="sam_hq_vit_b (379MB)",
        family=ModelFamily.SAM,
        model_type="sam_hq_vit_b",
        source_repo="lkeab/hq-sam",
        artifacts=(
            ModelArtifact(
                artifact_id="sam_hq_vit_b_checkpoint",
                filename="sam_hq_vit_b.pth",
                folder_name="sams",
                source_url=(
                    "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"
                ),
                description="SAM-HQ ViT-B checkpoint",
            ),
        ),
    ),
    ModelEntry(
        entry_id="mobile_sam",
        display_name="mobile_sam (39MB)",
        family=ModelFamily.SAM,
        model_type="mobile_sam",
        source_repo="ChaoningZhang/MobileSAM",
        artifacts=(
            ModelArtifact(
                artifact_id="mobile_sam_checkpoint",
                filename="mobile_sam.pt",
                folder_name="sams",
                source_url=(
                    "https://github.com/ChaoningZhang/MobileSAM/raw/master/"
                    "weights/mobile_sam.pt"
                ),
                description="MobileSAM checkpoint",
            ),
        ),
    ),
)

GROUNDING_DINO_ENTRIES: tuple[ModelEntry, ...] = (
    ModelEntry(
        entry_id="groundingdino_swint_ogc",
        display_name="GroundingDINO_SwinT_OGC (694MB)",
        family=ModelFamily.GROUNDING_DINO,
        model_type="swin_t",
        source_repo="ShilongLiu/GroundingDINO",
        artifacts=(
            ModelArtifact(
                artifact_id="groundingdino_swint_ogc_config",
                filename="GroundingDINO_SwinT_OGC.cfg.py",
                folder_name="grounding-dino",
                source_url=(
                    "https://huggingface.co/ShilongLiu/GroundingDINO/"
                    "resolve/main/GroundingDINO_SwinT_OGC.cfg.py"
                ),
                description="GroundingDINO SwinT OGC config",
            ),
            ModelArtifact(
                artifact_id="groundingdino_swint_ogc_checkpoint",
                filename="groundingdino_swint_ogc.pth",
                folder_name="grounding-dino",
                source_url=(
                    "https://huggingface.co/ShilongLiu/GroundingDINO/"
                    "resolve/main/groundingdino_swint_ogc.pth"
                ),
                description="GroundingDINO SwinT OGC checkpoint",
            ),
        ),
    ),
    ModelEntry(
        entry_id="groundingdino_swinb",
        display_name="GroundingDINO_SwinB (938MB)",
        family=ModelFamily.GROUNDING_DINO,
        model_type="swin_b",
        source_repo="ShilongLiu/GroundingDINO",
        artifacts=(
            ModelArtifact(
                artifact_id="groundingdino_swinb_config",
                filename="GroundingDINO_SwinB.cfg.py",
                folder_name="grounding-dino",
                source_url=(
                    "https://huggingface.co/ShilongLiu/GroundingDINO/"
                    "resolve/main/GroundingDINO_SwinB.cfg.py"
                ),
                description="GroundingDINO SwinB config",
            ),
            ModelArtifact(
                artifact_id="groundingdino_swinb_checkpoint",
                filename="groundingdino_swinb_cogcoor.pth",
                folder_name="grounding-dino",
                source_url=(
                    "https://huggingface.co/ShilongLiu/GroundingDINO/"
                    "resolve/main/groundingdino_swinb_cogcoor.pth"
                ),
                description="GroundingDINO SwinB checkpoint",
            ),
        ),
    ),
)

BERT_ENTRY = ModelEntry(
    entry_id="bert_base_uncased",
    display_name="BERT base uncased",
    family=ModelFamily.TEXT_ENCODER,
    model_type="bert",
    source_repo="google-bert/bert-base-uncased",
    artifacts=(
        ModelArtifact(
            artifact_id="bert_config",
            filename="config.json",
            folder_name="text_encoders",
            source_url=(
                "https://huggingface.co/google-bert/bert-base-uncased/"
                "resolve/main/config.json"
            ),
            description="BERT config",
        ),
        ModelArtifact(
            artifact_id="bert_tokenizer",
            filename="tokenizer.json",
            folder_name="text_encoders",
            source_url=(
                "https://huggingface.co/google-bert/bert-base-uncased/"
                "resolve/main/tokenizer.json"
            ),
            description="BERT tokenizer",
        ),
        ModelArtifact(
            artifact_id="bert_tokenizer_config",
            filename="tokenizer_config.json",
            folder_name="text_encoders",
            source_url=(
                "https://huggingface.co/google-bert/bert-base-uncased/"
                "resolve/main/tokenizer_config.json"
            ),
            description="BERT tokenizer config",
        ),
        ModelArtifact(
            artifact_id="bert_vocab",
            filename="vocab.txt",
            folder_name="text_encoders",
            source_url=(
                "https://huggingface.co/google-bert/bert-base-uncased/"
                "resolve/main/vocab.txt"
            ),
            description="BERT vocabulary",
        ),
        ModelArtifact(
            artifact_id="bert_weights",
            filename="model.safetensors",
            folder_name="text_encoders",
            source_url=(
                "https://huggingface.co/google-bert/bert-base-uncased/"
                "resolve/main/model.safetensors"
            ),
            description="BERT weights",
        ),
    ),
)

VITMATTE_ENTRIES: tuple[ModelEntry, ...] = (
    ModelEntry(
        entry_id="vitmatte-small-composition-1k",
        display_name="vitmatte-small-composition-1k",
        family=ModelFamily.VITMATTE,
        model_type="vitmatte_small",
        source_repo="hustvl/vitmatte-small-composition-1k",
        artifacts=(),
    ),
    ModelEntry(
        entry_id="vitmatte-base-composition-1k",
        display_name="vitmatte-base-composition-1k",
        family=ModelFamily.VITMATTE,
        model_type="vitmatte_base",
        source_repo="hustvl/vitmatte-base-composition-1k",
        artifacts=(),
    ),
)

DEFAULT_WD14_TAGGER_MODEL = "wd-eva02-large-tagger-v3"


def _wd14_tagger_entry(model_id: str) -> ModelEntry:
    """Build a WD14 catalog entry from the canonical SmilingWolf repository."""

    repo = f"SmilingWolf/{model_id}"
    return ModelEntry(
        entry_id=model_id,
        display_name=model_id,
        family=ModelFamily.WD14_TAGGER,
        model_type="wd14",
        source_repo=repo,
        artifacts=(
            ModelArtifact(
                artifact_id="onnx",
                filename=f"{model_id}.onnx",
                folder_name="wd14_tagger",
                source_url=f"https://huggingface.co/{repo}/resolve/main/model.onnx",
                description=f"{model_id} ONNX model",
            ),
            ModelArtifact(
                artifact_id="tags",
                filename=f"{model_id}.csv",
                folder_name="wd14_tagger",
                source_url=(
                    f"https://huggingface.co/{repo}/resolve/main/selected_tags.csv"
                ),
                description=f"{model_id} selected tags CSV",
            ),
        ),
    )


WD14_TAGGER_ENTRIES: tuple[ModelEntry, ...] = (
    _wd14_tagger_entry("wd-eva02-large-tagger-v3"),
    _wd14_tagger_entry("wd-vit-tagger-v3"),
    _wd14_tagger_entry("wd-swinv2-tagger-v3"),
    _wd14_tagger_entry("wd-convnext-tagger-v3"),
    _wd14_tagger_entry("wd-v1-4-moat-tagger-v2"),
    _wd14_tagger_entry("wd-v1-4-convnextv2-tagger-v2"),
    _wd14_tagger_entry("wd-v1-4-convnext-tagger-v2"),
    _wd14_tagger_entry("wd-v1-4-convnext-tagger"),
    _wd14_tagger_entry("wd-v1-4-vit-tagger-v2"),
    _wd14_tagger_entry("wd-v1-4-swinv2-tagger-v2"),
    _wd14_tagger_entry("wd-v1-4-vit-tagger"),
)

ANIMA_QWEN_TEXT_ENCODER = AutoModelArtifact(
    cache_id="anima_qwen_text_encoder",
    filename="qwen_3_06b_base.safetensors",
    folder_name="text_encoders",
    canonical_subfolder="qwen",
    source_url=(
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/"
        "split_files/text_encoders/qwen_3_06b_base.safetensors"
    ),
    source_repo="circlestone-labs/Anima",
    description="Anima Qwen3 0.6B text encoder",
    sha256="cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
)

ANIMA_QWEN_VAE = AutoModelArtifact(
    cache_id="anima_qwen_vae",
    filename="qwen_image_vae.safetensors",
    folder_name="vae",
    canonical_subfolder="qwen",
    source_url=(
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/"
        "split_files/vae/qwen_image_vae.safetensors"
    ),
    source_repo="circlestone-labs/Anima",
    description="Anima Qwen Image VAE",
    sha256="a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
)

ANIMA_AUTO_ARTIFACTS = (ANIMA_QWEN_TEXT_ENCODER, ANIMA_QWEN_VAE)


def sam_choices() -> list[str]:
    """Return deterministic SAM dropdown choices."""

    return [entry.display_name for entry in SAM_ENTRIES]


def grounding_dino_choices() -> list[str]:
    """Return deterministic GroundingDINO dropdown choices."""

    return [entry.display_name for entry in GROUNDING_DINO_ENTRIES]


def vitmatte_choices() -> list[str]:
    """Return deterministic ViTMatte dropdown choices."""

    return [entry.display_name for entry in VITMATTE_ENTRIES]


def wd14_tagger_choices() -> list[str]:
    """Return deterministic WD14 tagger dropdown choices."""

    return [entry.display_name for entry in WD14_TAGGER_ENTRIES]


def get_sam_entry(selection: str) -> ModelEntry:
    """Return the SAM catalog entry matching an id or display name."""

    return _get_entry(selection, SAM_ENTRIES, "SAM")


def get_grounding_dino_entry(selection: str) -> ModelEntry:
    """Return the GroundingDINO catalog entry matching an id or display name."""

    return _get_entry(selection, GROUNDING_DINO_ENTRIES, "GroundingDINO")


def get_vitmatte_entry(selection: str) -> ModelEntry:
    """Return the ViTMatte catalog entry matching an id or display name."""

    return _get_entry(selection, VITMATTE_ENTRIES, "ViTMatte")


def get_wd14_tagger_entry(selection: str) -> ModelEntry:
    """Return the WD14 tagger catalog entry matching an id or display name."""

    return _get_entry(selection, WD14_TAGGER_ENTRIES, "WD14 tagger")


def _get_entry(
    selection: str,
    entries: tuple[ModelEntry, ...],
    model_label: str,
) -> ModelEntry:
    """Return a catalog entry or raise an actionable selection error."""

    for entry in entries:
        if selection in (entry.entry_id, entry.display_name):
            return entry
    valid = ", ".join(entry.display_name for entry in entries)
    raise ValueError(
        f"Unknown {model_label} model '{selection}'. Expected one of: {valid}."
    )
