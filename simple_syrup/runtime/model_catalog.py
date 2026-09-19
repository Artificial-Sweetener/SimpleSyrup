# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Known model metadata for downloadable SimpleSyrup model loaders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelFamily(StrEnum):
    """Catalog families used by SimpleSyrup model selection."""

    SAM = "sam"
    GROUNDING_DINO = "grounding_dino"
    TEXT_ENCODER = "text_encoder"
    VITMATTE = "vitmatte"
    WD14_TAGGER = "wd14_tagger"
    ULTRALYTICS = "ultralytics"


@dataclass(frozen=True)
class ModelArtifact:
    """A downloadable file required by a known model entry."""

    artifact_id: str
    filename: str
    folder_name: str
    source_url: str
    description: str
    sha256: str | None = None


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
    ModelEntry(
        entry_id="fast_sam_s",
        display_name="FastSAM-s (23MB)",
        family=ModelFamily.SAM,
        model_type="fast_sam",
        source_repo="ultralytics/assets",
        artifacts=(
            ModelArtifact(
                artifact_id="fast_sam_s_checkpoint",
                filename="FastSAM-s.pt",
                folder_name="sams",
                source_url=(
                    "https://github.com/ultralytics/assets/releases/latest/download/"
                    "FastSAM-s.pt"
                ),
                description="FastSAM-s checkpoint",
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


_ANZHCS_YOLOS_REVISION = "f5a2306d7fed4f3cfc26c25ff1ab2e3f3cfce855"
_ANZHCS_YOLOS_REPOSITORY = "Anzhc/Anzhcs_YOLOs"


def _huggingface_yolo_entry(
    *,
    entry_id: str,
    display_name: str,
    filename: str,
    folder_name: str,
    model_type: str,
    source_repo: str,
    revision: str,
    license_note: str,
    description: str,
    sha256: str,
) -> ModelEntry:
    """Build one revision-pinned Hugging Face Ultralytics catalog entry."""

    encoded_filename = filename.replace(" ", "%20")
    return ModelEntry(
        entry_id=entry_id,
        display_name=display_name,
        family=ModelFamily.ULTRALYTICS,
        model_type=model_type,
        source_repo=source_repo,
        license_note=license_note,
        artifacts=(
            ModelArtifact(
                artifact_id=f"{entry_id}_checkpoint",
                filename=filename,
                folder_name=folder_name,
                source_url=(
                    f"https://huggingface.co/{source_repo}/resolve/{revision}/"
                    f"{encoded_filename}"
                ),
                description=description,
                sha256=sha256,
            ),
        ),
    )


def _anzhc_yolo_entry(
    *,
    entry_id: str,
    display_name: str,
    filename: str,
    folder_name: str,
    model_type: str,
    description: str,
    sha256: str,
) -> ModelEntry:
    """Build one revision-pinned Anzhc Ultralytics catalog entry."""

    return _huggingface_yolo_entry(
        entry_id=entry_id,
        display_name=display_name,
        filename=filename,
        folder_name=folder_name,
        model_type=model_type,
        source_repo=_ANZHCS_YOLOS_REPOSITORY,
        revision=_ANZHCS_YOLOS_REVISION,
        license_note="AGPL-3.0",
        description=description,
        sha256=sha256,
    )


ULTRALYTICS_ENTRIES: tuple[ModelEntry, ...] = (
    _anzhc_yolo_entry(
        entry_id="anzhc_face_seg",
        display_name="Anzhc Face -seg (6.52MB)",
        filename="Anzhc Face -seg.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc face segmentation model",
        sha256="dbf083201298a495e332113de0612d1be1ae8307628628eb7972a31979cdbbb3",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_face_seg_640_v2_y8n",
        display_name="Anzhc Face seg 640 v2 y8n (6.56MB)",
        filename="Anzhc Face seg 640 v2 y8n.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc face segmentation model",
        sha256="d473e8bccc4c833d8eb36c95e566ce6460ffdc8b2899c859910e380c85def276",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_face_seg_768_v2_y8n",
        display_name="Anzhc Face seg 768 v2 y8n (6.58MB)",
        filename="Anzhc Face seg 768 v2 y8n.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc face segmentation model",
        sha256="9a1e5b154c1d190812447431bda6b8f260f132877812b4a2f163981f54558355",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_face_seg_768ms_v2_y8n",
        display_name="Anzhc Face seg 768MS v2 y8n (6.60MB)",
        filename="Anzhc Face seg 768MS v2 y8n.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc multi-scale face segmentation model",
        sha256="429e88d9aecb9fa4167ffd41a6ebc42c97b7fa785aa5468a7eb302ceb9837aae",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_face_seg_1024_v2_y8n",
        display_name="Anzhc Face seg 1024 v2 y8n (6.63MB)",
        filename="Anzhc Face seg 1024 v2 y8n.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc face segmentation model",
        sha256="1bbcfd7a9f407c6f6e4389a371dbcc392f9444421cf7f824152e92bf563dc6a3",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_face_seg_640_v3_y11n",
        display_name="Anzhc Face seg 640 v3 y11n (5.80MB)",
        filename="Anzhc Face seg 640 v3 y11n.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc YOLO11 face segmentation model",
        sha256="96437afc773bacd118e275e6cddc1fb7263c78dc11299989c7a00a26506c45bf",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_face_seg_640_v4_y11n",
        display_name="Anzhc Face seg 640 v4 y11n (5.74MB)",
        filename="Anzhc Face seg 640 v4 y11n.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc YOLO11 face segmentation model",
        sha256="1e77ad7bd349babd8a4a90478bfc965348642b63a8d95d3b43ee13db42fd0a64",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhcs_manface_v02_1024_y8n",
        display_name="Anzhcs ManFace v02 1024 y8n (6.06MB)",
        filename="Anzhcs ManFace v02 1024 y8n.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc male face segmentation model",
        sha256="184b9a680afb3c4a559e46e2fe692338fe7bdd6267979fa4ef10526fa96c1b31",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhcs_womanface_v05_1024_y8n",
        display_name="Anzhcs WomanFace v05 1024 y8n (6.07MB)",
        filename="Anzhcs WomanFace v05 1024 y8n.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc female face segmentation model",
        sha256="84db37616e1ca975c4e23fa5a300acf0edd9144ec287bbbdbd1ad0f4a3afa9c1",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_eyes_seg_hd",
        display_name="Anzhc Eyes -seg-hd (6.59MB)",
        filename="Anzhc Eyes -seg-hd.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc eye segmentation model",
        sha256="6be1c13ca7a51c2425e278e07e7ae3d4c94ee125b874a0104a142f4f5a35a308",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_headhair_seg_y8n",
        display_name="Anzhc HeadHair seg y8n (6.50MB)",
        filename="Anzhc HeadHair seg y8n.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc head and hair segmentation model",
        sha256="a6e99b1305f600c35e7f6400741c2322b198ae03755f91dc1c59d7a78d77f13c",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_headhair_seg_y8m",
        display_name="Anzhc HeadHair seg y8m (52.34MB)",
        filename="Anzhc HeadHair seg y8m.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc head and hair segmentation model",
        sha256="f63aa1cdb63a26c0025a4a984588248241a5838aff4edfeea93d9c155efe0b5e",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_breasts_seg_v1_1024n",
        display_name="Anzhc Breasts Seg v1 1024n (6.58MB)",
        filename="Anzhc Breasts Seg v1 1024n.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc breast segmentation model",
        sha256="d469bd7abdcbe32a946e0e342bc1fe96aa021987787d51245f97a29e114cb31b",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_breasts_seg_v1_1024s",
        display_name="Anzhc Breasts Seg v1 1024s (22.86MB)",
        filename="Anzhc Breasts Seg v1 1024s.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc breast segmentation model",
        sha256="413a9b948a40f96a83769a882816ef0dd2b91b49673c91bff75463660077b395",
    ),
    _anzhc_yolo_entry(
        entry_id="anzhc_breasts_seg_v1_1024m",
        display_name="Anzhc Breasts Seg v1 1024m (52.39MB)",
        filename="Anzhc Breasts Seg v1 1024m.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        description="Anzhc breast segmentation model",
        sha256="53d15e82a8308f8056f4929838e00e42c8da576b661e0c2b4fef5837d8b5b2b4",
    ),
    _huggingface_yolo_entry(
        entry_id="bingsu_face_yolov8n_v2",
        display_name="Bingsu Face YOLOv8n v2 (6.23MB)",
        filename="face_yolov8n_v2.pt",
        folder_name="ultralytics_bbox",
        model_type="detect",
        source_repo="Bingsu/adetailer",
        revision="53cc19de382014514d9d4038601d261a7faa9b7b",
        license_note="Apache-2.0",
        description="Bingsu ADetailer face detection model",
        sha256="8f5f2110f83c4e00712993fab48c771d26036e2e80ec62bd5b9cb37c29e36b36",
    ),
    _huggingface_yolo_entry(
        entry_id="bingsu_face_yolov8s",
        display_name="Bingsu Face YOLOv8s (22.5MB)",
        filename="face_yolov8s.pt",
        folder_name="ultralytics_bbox",
        model_type="detect",
        source_repo="Bingsu/adetailer",
        revision="53cc19de382014514d9d4038601d261a7faa9b7b",
        license_note="Apache-2.0",
        description="Bingsu ADetailer face detection model",
        sha256="c7237eff25787377de196961140ceaed324d859ee8de5a775d93d33a0e3fab78",
    ),
    _huggingface_yolo_entry(
        entry_id="bingsu_hand_yolov8n",
        display_name="Bingsu Hand YOLOv8n (6.23MB)",
        filename="hand_yolov8n.pt",
        folder_name="ultralytics_bbox",
        model_type="detect",
        source_repo="Bingsu/adetailer",
        revision="53cc19de382014514d9d4038601d261a7faa9b7b",
        license_note="Apache-2.0",
        description="Bingsu ADetailer hand detection model",
        sha256="3991202eb69e9ddcb3b9ba80cdeb41e734ffaf844403d6c9f47d515cd88c6f29",
    ),
    _huggingface_yolo_entry(
        entry_id="bingsu_hand_yolov8s",
        display_name="Bingsu Hand YOLOv8s (22.5MB)",
        filename="hand_yolov8s.pt",
        folder_name="ultralytics_bbox",
        model_type="detect",
        source_repo="Bingsu/adetailer",
        revision="53cc19de382014514d9d4038601d261a7faa9b7b",
        license_note="Apache-2.0",
        description="Bingsu ADetailer hand detection model",
        sha256="70b540063fbc385736d8258970744a4afbc4cbf7932134bae3b24cdadeadec06",
    ),
    _huggingface_yolo_entry(
        entry_id="bingsu_person_yolov8n_seg",
        display_name="Bingsu Person YOLOv8n-seg (6.78MB)",
        filename="person_yolov8n-seg.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        source_repo="Bingsu/adetailer",
        revision="53cc19de382014514d9d4038601d261a7faa9b7b",
        license_note="Apache-2.0",
        description="Bingsu ADetailer person segmentation model",
        sha256="38fc8aaae97cb6e70be4ec44770005b26ed473471362afcda62a0037d7ccf432",
    ),
    _huggingface_yolo_entry(
        entry_id="bingsu_person_yolov8s_seg",
        display_name="Bingsu Person YOLOv8s-seg (23.9MB)",
        filename="person_yolov8s-seg.pt",
        folder_name="ultralytics_segm",
        model_type="segment",
        source_repo="Bingsu/adetailer",
        revision="53cc19de382014514d9d4038601d261a7faa9b7b",
        license_note="Apache-2.0",
        description="Bingsu ADetailer person segmentation model",
        sha256="53c54aec2239355faffc6c5b70d0f3d05042f386f956cbec39cec46ad456f050",
    ),
    _huggingface_yolo_entry(
        entry_id="fuyucchi_yolov8x6_animeface",
        display_name="Fuyucchi YOLOv8x6 Anime Face (195MB)",
        filename="yolov8x6_animeface.pt",
        folder_name="ultralytics_bbox",
        model_type="detect",
        source_repo="Fuyucchi/yolov8_animeface",
        revision="b0841ce930453c0f23ceb8086d6554c17de5fe4a",
        license_note="AGPL-3.0",
        description="Fuyucchi high-resolution anime face detection model",
        sha256="f3cdc1a6266347322439fd9b3c8f5a1222668eb10c8adf00e17b28c48b95213c",
    ),
)


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


def ultralytics_choices() -> list[str]:
    """Return deterministic Ultralytics dropdown choices."""

    return [entry.display_name for entry in ULTRALYTICS_ENTRIES]


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


def get_ultralytics_entry(selection: str) -> ModelEntry:
    """Return the Ultralytics catalog entry matching an id or display name."""

    return _get_entry(selection, ULTRALYTICS_ENTRIES, "Ultralytics")


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
