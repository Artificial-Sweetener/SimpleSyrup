# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist readable, globally shared quantized checkpoint cache artifacts."""

from __future__ import annotations

import importlib
import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from types import ModuleType
from typing import Any

from ..domain.model_quantization import ModelQuantizationRecipe, QuantizationProfile
from ..domain.quant_cache import QuantCacheIdentity, QuantCacheManifest
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
CACHE_DIRECTORY_NAME = "SyrupQuants"
ARTIFACT_FILENAME_SUFFIX = ".safetensors"
MANIFEST_FILENAME = "manifest.json"
README_FILENAME = "README.txt"
README_CONTENT = """SimpleSyrup Quantized Model Cache
===================================

This folder contains quantized copies generated from models selected in
SimpleSyrup loader nodes. Your original models remain in their normal folders
and are the authoritative models recorded in workflows.

SimpleSyrup manages this folder as one global least-recently-used cache. You can
change its size limit or clear inactive cached models in the SimpleSyrup section
of ComfyUI settings.

It is safe to delete this entire folder while ComfyUI is stopped. Missing
quantized copies will be generated again when requested.
"""
_SAFE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_MANIFEST_REPLACE_ATTEMPTS = 20
_MANIFEST_REPLACE_RETRY_SECONDS = 0.01


@dataclass(frozen=True)
class QuantCacheArtifact:
    """Pair a complete cached model path with its validated manifest."""

    path: Path
    manifest_path: Path
    manifest: QuantCacheManifest


class QuantCacheRepository:
    """Own filesystem layout and persistence for SimpleSyrup quant artifacts."""

    def __init__(
        self,
        cache_root: Path | None = None,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create a repository with injectable model-directory discovery."""

        self._cache_root = cache_root
        self._folder_paths_module = folder_paths_module

    @property
    def root(self) -> Path:
        """Return the unregistered cache directory below ComfyUI models."""

        if self._cache_root is not None:
            return self._cache_root
        folder_paths = self._folder_paths_module or _folder_paths()
        models_dir: Any = folder_paths.models_dir
        self._cache_root = Path(str(models_dir)) / CACHE_DIRECTORY_NAME
        return self._cache_root

    @property
    def lock_directory(self) -> Path:
        """Return the internal cross-process lock directory."""

        return self.root / ".locks"

    def ensure_root(self) -> None:
        """Create the cache root and its plain-language ownership notice."""

        self.root.mkdir(parents=True, exist_ok=True)
        readme_path = self.root / README_FILENAME
        if not readme_path.is_file():
            readme_path.write_text(README_CONTENT, encoding="utf-8")

    def find_current(
        self,
        *,
        source_model: str,
        source_path: Path,
        source_size_bytes: int,
        source_modified_ns: int,
        profile: QuantizationProfile,
        recipe: ModelQuantizationRecipe,
    ) -> QuantCacheArtifact | None:
        """Return a cached artifact matching the source's current file state."""

        if not self.root.is_dir():
            return None
        resolved_source = source_path.resolve()
        for artifact in self.list_artifacts():
            if not artifact.manifest.matches_current_source(
                source_model,
                resolved_source,
                source_size_bytes,
                source_modified_ns,
                profile,
                recipe.model_family,
                recipe.version,
            ):
                continue
            return self.touch(artifact)
        return None

    def find_identity(self, identity: QuantCacheIdentity) -> QuantCacheArtifact | None:
        """Return the completed artifact for an exact hashed identity."""

        directory = self.artifact_directory(identity)
        manifest_path = directory / MANIFEST_FILENAME
        artifact = self._load_artifact(manifest_path)
        if artifact is None or not artifact.manifest.matches_identity(identity):
            return None
        return self.touch(artifact)

    def create_build_directory(self, identity: QuantCacheIdentity) -> Path:
        """Create an isolated temporary directory for one atomic build."""

        self.ensure_root()
        building_root = self.root / ".building"
        building_root.mkdir(parents=True, exist_ok=True)
        directory = building_root / f"{identity.stable_key}-{uuid.uuid4().hex}"
        directory.mkdir()
        return directory

    def artifact_filename(self, identity: QuantCacheIdentity) -> str:
        """Return a recognizable filename within a cache artifact directory."""

        stem = _safe_component(Path(identity.source.display_name).stem)
        return f"{stem}--{identity.profile.profile_id}{ARTIFACT_FILENAME_SUFFIX}"

    def commit(
        self,
        identity: QuantCacheIdentity,
        build_directory: Path,
    ) -> QuantCacheArtifact:
        """Atomically publish a validated build directory and its manifest."""

        self._require_within(build_directory, self.root / ".building")
        artifact_file = self.artifact_filename(identity)
        artifact_path = build_directory / artifact_file
        if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
            raise ValueError(
                "Quantized checkpoint build produced no valid artifact file."
            )
        manifest = QuantCacheManifest.create(
            identity,
            artifact_file,
            artifact_path.stat().st_size,
        )
        self._write_manifest(build_directory / MANIFEST_FILENAME, manifest)

        final_directory = self.artifact_directory(identity)
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        if final_directory.exists():
            existing = self.find_identity(identity)
            if existing is not None:
                shutil.rmtree(build_directory)
                return existing
            self._require_within(final_directory, self.root)
            LOGGER.warning(
                "replacing invalid quant cache artifact",
                extra={"artifact_directory": str(final_directory)},
            )
            shutil.rmtree(final_directory)
        build_directory.replace(final_directory)
        committed = self._load_artifact(final_directory / MANIFEST_FILENAME)
        if committed is None:
            raise RuntimeError("Published quant cache artifact could not be read back.")
        return committed

    def discard_build(self, build_directory: Path) -> None:
        """Remove a failed temporary build without touching completed artifacts."""

        try:
            self._require_within(build_directory, self.root / ".building")
        except ValueError:
            return
        if build_directory.is_dir():
            shutil.rmtree(build_directory, ignore_errors=True)

    def artifact_directory(self, identity: QuantCacheIdentity) -> Path:
        """Return the readable directory for an exact derived artifact."""

        family = _safe_component(identity.model_family)
        source = _safe_component(Path(identity.source.display_name).stem)
        profile = _safe_component(identity.profile.profile_id)
        version = (
            f"{identity.source.sha256[:12]}-profile-{identity.profile.version}"
            f"-recipe-{identity.recipe_version}"
        )
        return self.root / family / source / profile / version

    def list_artifacts(self) -> tuple[QuantCacheArtifact, ...]:
        """Return every valid completed SimpleSyrup-managed artifact."""

        if not self.root.is_dir():
            return ()
        artifacts: list[QuantCacheArtifact] = []
        for manifest_path in self.root.rglob(MANIFEST_FILENAME):
            if ".building" in manifest_path.parts:
                continue
            artifact = self._load_artifact(manifest_path)
            if artifact is not None:
                artifacts.append(artifact)
        return tuple(artifacts)

    def touch(self, artifact: QuantCacheArtifact) -> QuantCacheArtifact:
        """Record explicit last use for portable LRU behavior."""

        touched_manifest = artifact.manifest.touched()
        self._write_manifest(artifact.manifest_path, touched_manifest)
        return QuantCacheArtifact(
            path=artifact.path,
            manifest_path=artifact.manifest_path,
            manifest=touched_manifest,
        )

    def remove(self, artifact: QuantCacheArtifact) -> bool:
        """Remove one validated managed artifact directory."""

        directory = artifact.manifest_path.parent
        self._require_within(directory, self.root)
        try:
            shutil.rmtree(directory)
        except OSError as error:
            LOGGER.warning(
                "quant cache artifact eviction deferred",
                extra={"artifact": str(artifact.path), "reason": str(error)},
            )
            return False
        self._remove_empty_parents(directory.parent)
        return True

    def relative_display_path(self) -> str:
        """Return the user-facing location below ComfyUI's model directory."""

        return f"models/{CACHE_DIRECTORY_NAME}"

    def _load_artifact(self, manifest_path: Path) -> QuantCacheArtifact | None:
        """Load one valid managed artifact, ignoring unrelated or corrupt files."""

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = QuantCacheManifest.from_payload(payload)
            if Path(manifest.artifact_file).name != manifest.artifact_file:
                raise ValueError(
                    "Quant cache artifact filename must not contain a path."
                )
            artifact_path = manifest_path.parent / manifest.artifact_file
            if not artifact_path.is_file():
                raise ValueError("Quant cache artifact file is missing.")
            if artifact_path.stat().st_size != manifest.artifact_size_bytes:
                raise ValueError(
                    "Quant cache artifact size does not match its manifest."
                )
            return QuantCacheArtifact(artifact_path, manifest_path, manifest)
        except (JSONDecodeError, OSError, ValueError) as error:
            LOGGER.warning(
                "ignoring invalid quant cache manifest",
                extra={"manifest": str(manifest_path), "reason": str(error)},
            )
            return None

    @staticmethod
    def _write_manifest(path: Path, manifest: QuantCacheManifest) -> None:
        """Persist a manifest through an atomic same-directory replacement."""

        temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(manifest.to_payload(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            QuantCacheRepository._replace_manifest(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _replace_manifest(temporary_path: Path, path: Path) -> None:
        """Retry only transient Windows sharing violations during atomic replace."""

        for attempt in range(_MANIFEST_REPLACE_ATTEMPTS):
            try:
                temporary_path.replace(path)
                return
            except PermissionError:
                if attempt + 1 == _MANIFEST_REPLACE_ATTEMPTS:
                    raise
                time.sleep(_MANIFEST_REPLACE_RETRY_SECONDS)

    def _remove_empty_parents(self, directory: Path) -> None:
        """Remove empty readable grouping folders without removing the cache root."""

        current = directory
        while current != self.root:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

    @staticmethod
    def _require_within(path: Path, expected_root: Path) -> None:
        """Reject destructive operations outside the intended cache subtree."""

        try:
            path.resolve().relative_to(expected_root.resolve())
        except ValueError as error:
            raise ValueError(
                f"Quant cache path '{path}' is outside '{expected_root}'."
            ) from error


def _safe_component(value: str) -> str:
    """Return a readable path component with unsafe characters replaced."""

    normalized = _SAFE_COMPONENT_PATTERN.sub("_", value).strip("._")
    return normalized[:120] or "model"


def _folder_paths() -> ModuleType:
    """Import ComfyUI folder paths lazily."""

    module: Any = importlib.import_module("folder_paths")
    if not isinstance(module, ModuleType):
        raise TypeError("folder_paths import did not return a module.")
    return module
