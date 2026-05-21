# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Request model catalog and local installation metadata."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

from .grounding_dino_loader import TEXT_ENCODER_CHOICES
from .model_catalog import (
    VITMATTE_ENTRIES,
    ModelEntry,
    get_grounding_dino_entry,
    get_sam_entry,
)
from .model_folders import (
    expected_model_file,
    get_primary_model_folder,
    register_required_model_folders,
    resolve_model_file,
)
from .vitmatte_loader import is_valid_vitmatte_directory


class GroundedSAMModelMetadata:
    """Describe known SAM, GroundingDINO, and text encoder assets."""

    def __init__(self, folder_paths_module: ModuleType | None = None) -> None:
        """Create a metadata provider with injectable folder paths."""

        self._folder_paths_module = folder_paths_module

    def describe_selection(self, sam_model: str, grounding_dino_model: str) -> str:
        """Return JSON metadata for selected model catalog entries."""

        register_required_model_folders(self._folder_paths_module)
        sam = get_sam_entry(sam_model)
        dino = get_grounding_dino_entry(grounding_dino_model)
        payload = {
            "sam": self._entry_metadata(sam),
            "grounding_dino": self._entry_metadata(dino),
            "text_encoder_choices": list(TEXT_ENCODER_CHOICES),
            "text_encoder_paths": {
                "layerstyle": str(self._models_dir() / "bert-base-uncased"),
                "text_encoders_bert": str(
                    expected_model_file(
                        "text_encoders",
                        "bert",
                        self._folder_paths_module,
                    )
                ),
            },
            "vitmatte": [self._vitmatte_metadata(entry) for entry in VITMATTE_ENTRIES],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def _entry_metadata(self, entry: ModelEntry) -> dict[str, object]:
        """Return JSON-serializable metadata for one catalog entry."""

        artifacts: list[dict[str, object]] = []
        for artifact in entry.artifacts:
            local_path = resolve_model_file(
                artifact.folder_name,
                artifact.filename,
                self._folder_paths_module,
            )
            expected = expected_model_file(
                artifact.folder_name,
                artifact.filename,
                self._folder_paths_module,
            )
            artifacts.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "filename": artifact.filename,
                    "source_url": artifact.source_url,
                    "expected_path": str(expected),
                    "local_path": str(local_path) if local_path else None,
                    "installed": local_path is not None,
                }
            )
        return {
            "id": entry.entry_id,
            "display_name": entry.display_name,
            "model_type": entry.model_type,
            "source_repo": entry.source_repo,
            "auto_download_allowed": entry.auto_download_allowed,
            "artifacts": artifacts,
        }

    def _vitmatte_metadata(self, entry: ModelEntry) -> dict[str, object]:
        """Return metadata for one ViTMatte model directory."""

        canonical = (
            get_primary_model_folder(
                "vitmatte",
                self._folder_paths_module,
            )
            / entry.entry_id
        )
        layerstyle = self._vitmatte_layerstyle_path(entry)
        candidates = [canonical]
        if layerstyle is not None and layerstyle not in candidates:
            candidates.append(layerstyle)
        installed_path = next(
            (
                candidate
                for candidate in candidates
                if is_valid_vitmatte_directory(candidate)
            ),
            None,
        )
        return {
            "id": entry.entry_id,
            "display_name": entry.display_name,
            "model_type": entry.model_type,
            "source_repo": entry.source_repo,
            "canonical_path": str(canonical),
            "layerstyle_compatible_path": str(layerstyle) if layerstyle else None,
            "installed": installed_path is not None,
            "local_path": str(installed_path) if installed_path else None,
        }

    def _vitmatte_layerstyle_path(self, entry: ModelEntry) -> Path | None:
        """Return LayerStyle's compatible ViTMatte path for one entry."""

        if entry.entry_id == "vitmatte-small-composition-1k":
            return self._models_dir() / "vitmatte"
        if entry.entry_id == "vitmatte-base-composition-1k":
            return self._models_dir() / "vitmatte-base-composition-1k"
        return None

    def _models_dir(self) -> Path:
        """Return ComfyUI's model root directory."""

        import importlib

        folder_paths = self._folder_paths_module or importlib.import_module(
            "folder_paths"
        )
        return Path(str(folder_paths.models_dir))
