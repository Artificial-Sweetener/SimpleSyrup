# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define validated identities and records for quantized profile artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from .model_quantization import QuantizationProfile

MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SourceCheckpointIdentity:
    """Identify an authoritative source checkpoint and its current file state."""

    display_name: str
    path: Path
    size_bytes: int
    modified_ns: int
    sha256: str


@dataclass(frozen=True)
class QuantCacheIdentity:
    """Identify one profile and recipe derivative of a source checkpoint."""

    source: SourceCheckpointIdentity
    profile: QuantizationProfile
    model_family: str
    recipe_version: int

    @property
    def stable_key(self) -> str:
        """Return the collision-resistant cache key."""

        identity = "\0".join(
            (
                self.source.sha256,
                self.profile.profile_id,
                str(self.profile.version),
                self.model_family,
                str(self.recipe_version),
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QuantCacheManifest:
    """Describe one complete SimpleSyrup-managed cache artifact."""

    source_model: str
    source_path: str
    source_sha256: str
    source_size_bytes: int
    source_modified_ns: int
    profile_id: str
    profile_label: str
    profile_version: int
    quantization_formats: tuple[str, ...]
    model_family: str
    recipe_version: int
    artifact_file: str
    artifact_size_bytes: int
    created_at: str
    last_used_at: str
    schema_version: int = MANIFEST_SCHEMA_VERSION
    managed_by: str = "SimpleSyrup"

    @classmethod
    def create(
        cls,
        identity: QuantCacheIdentity,
        artifact_file: str,
        artifact_size_bytes: int,
    ) -> QuantCacheManifest:
        """Create a current manifest for a completed artifact."""

        now = datetime.now(UTC).isoformat()
        return cls(
            source_model=identity.source.display_name,
            source_path=str(identity.source.path),
            source_sha256=identity.source.sha256,
            source_size_bytes=identity.source.size_bytes,
            source_modified_ns=identity.source.modified_ns,
            profile_id=identity.profile.profile_id,
            profile_label=identity.profile.label,
            profile_version=identity.profile.version,
            quantization_formats=tuple(
                sorted(item.value for item in identity.profile.required_formats)
            ),
            model_family=identity.model_family,
            recipe_version=identity.recipe_version,
            artifact_file=artifact_file,
            artifact_size_bytes=artifact_size_bytes,
            created_at=now,
            last_used_at=now,
        )

    def matches_current_source(
        self,
        source_model: str,
        source_path: Path,
        source_size_bytes: int,
        source_modified_ns: int,
        profile: QuantizationProfile,
        model_family: str,
        recipe_version: int,
    ) -> bool:
        """Return whether this artifact derives from the unchanged source file."""

        return (
            self.source_model == source_model
            and Path(self.source_path) == source_path
            and self.source_size_bytes == source_size_bytes
            and self.source_modified_ns == source_modified_ns
            and self.profile_id == profile.profile_id
            and self.profile_version == profile.version
            and self.model_family == model_family
            and self.recipe_version == recipe_version
        )

    def matches_identity(self, identity: QuantCacheIdentity) -> bool:
        """Return whether this v2 manifest exactly describes an identity."""

        return (
            self.schema_version == MANIFEST_SCHEMA_VERSION
            and self.source_sha256 == identity.source.sha256
            and self.profile_id == identity.profile.profile_id
            and self.profile_version == identity.profile.version
            and self.model_family == identity.model_family
            and self.recipe_version == identity.recipe_version
        )

    def touched(self) -> QuantCacheManifest:
        """Return a copy with a current explicit LRU timestamp."""

        payload = self.to_payload()
        payload["last_used_at"] = datetime.now(UTC).isoformat()
        return QuantCacheManifest.from_payload(payload)

    def to_payload(self) -> dict[str, object]:
        """Return the human-readable JSON representation."""

        if self.schema_version == 1:
            return {
                "schema_version": 1,
                "managed_by": self.managed_by,
                "source_model": self.source_model,
                "source_path": self.source_path,
                "source_sha256": self.source_sha256,
                "source_size_bytes": self.source_size_bytes,
                "source_modified_ns": self.source_modified_ns,
                "quantization_format": self.quantization_formats[0],
                "model_family": self.model_family,
                "recipe_version": self.recipe_version,
                "artifact_file": self.artifact_file,
                "artifact_size_bytes": self.artifact_size_bytes,
                "created_at": self.created_at,
                "last_used_at": self.last_used_at,
            }
        return {
            "schema_version": self.schema_version,
            "managed_by": self.managed_by,
            "source_model": self.source_model,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "source_modified_ns": self.source_modified_ns,
            "profile_id": self.profile_id,
            "profile_label": self.profile_label,
            "profile_version": self.profile_version,
            "quantization_formats": list(self.quantization_formats),
            "model_family": self.model_family,
            "recipe_version": self.recipe_version,
            "artifact_file": self.artifact_file,
            "artifact_size_bytes": self.artifact_size_bytes,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_payload(cls, payload: object) -> QuantCacheManifest:
        """Validate managed manifests, retaining v1 only for cache cleanup."""

        if not isinstance(payload, dict):
            raise ValueError("Quant cache manifest must be a JSON object.")
        if payload.get("schema_version") == 1:
            return cls._from_legacy_payload(payload)
        required_strings = (
            "managed_by",
            "source_model",
            "source_path",
            "source_sha256",
            "profile_id",
            "profile_label",
            "model_family",
            "artifact_file",
            "created_at",
            "last_used_at",
        )
        for key in required_strings:
            if not isinstance(payload.get(key), str):
                raise ValueError(
                    f"Quant cache manifest field '{key}' must be a string."
                )
        required_integers = (
            "schema_version",
            "source_size_bytes",
            "source_modified_ns",
            "profile_version",
            "recipe_version",
            "artifact_size_bytes",
        )
        for key in required_integers:
            value = payload.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"Quant cache manifest field '{key}' must be an integer."
                )
        raw_formats = payload.get("quantization_formats")
        if not isinstance(raw_formats, list) or not all(
            isinstance(item, str) for item in raw_formats
        ):
            raise ValueError(
                "Quant cache manifest field 'quantization_formats' must be a "
                "string list."
            )
        if payload["managed_by"] != "SimpleSyrup":
            raise ValueError("Quant cache manifest is not managed by SimpleSyrup.")
        if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError("Quant cache manifest schema version is unsupported.")
        return cls(
            schema_version=payload["schema_version"],
            managed_by=payload["managed_by"],
            source_model=payload["source_model"],
            source_path=payload["source_path"],
            source_sha256=payload["source_sha256"],
            source_size_bytes=payload["source_size_bytes"],
            source_modified_ns=payload["source_modified_ns"],
            profile_id=payload["profile_id"],
            profile_label=payload["profile_label"],
            profile_version=payload["profile_version"],
            quantization_formats=tuple(raw_formats),
            model_family=payload["model_family"],
            recipe_version=payload["recipe_version"],
            artifact_file=payload["artifact_file"],
            artifact_size_bytes=payload["artifact_size_bytes"],
            created_at=payload["created_at"],
            last_used_at=payload["last_used_at"],
        )

    @classmethod
    def _from_legacy_payload(cls, payload: dict[object, object]) -> QuantCacheManifest:
        """Decode v1 solely so ordinary LRU and clearing can remove it."""

        required_strings = (
            "managed_by",
            "source_model",
            "source_path",
            "source_sha256",
            "quantization_format",
            "model_family",
            "artifact_file",
            "created_at",
            "last_used_at",
        )
        required_integers = (
            "source_size_bytes",
            "source_modified_ns",
            "recipe_version",
            "artifact_size_bytes",
        )
        if any(not isinstance(payload.get(key), str) for key in required_strings):
            raise ValueError("Legacy quant cache manifest has invalid string fields.")
        if any(
            not isinstance(payload.get(key), int) or isinstance(payload.get(key), bool)
            for key in required_integers
        ):
            raise ValueError("Legacy quant cache manifest has invalid integer fields.")
        if payload["managed_by"] != "SimpleSyrup":
            raise ValueError("Quant cache manifest is not managed by SimpleSyrup.")
        quantization_format = str(payload["quantization_format"])
        return cls(
            schema_version=1,
            managed_by=str(payload["managed_by"]),
            source_model=str(payload["source_model"]),
            source_path=str(payload["source_path"]),
            source_sha256=str(payload["source_sha256"]),
            source_size_bytes=cast(int, payload["source_size_bytes"]),
            source_modified_ns=cast(int, payload["source_modified_ns"]),
            profile_id=f"legacy-v1-{quantization_format}",
            profile_label=f"Legacy v1 {quantization_format}",
            profile_version=1,
            quantization_formats=(quantization_format,),
            model_family=str(payload["model_family"]),
            recipe_version=cast(int, payload["recipe_version"]),
            artifact_file=str(payload["artifact_file"]),
            artifact_size_bytes=cast(int, payload["artifact_size_bytes"]),
            created_at=str(payload["created_at"]),
            last_used_at=str(payload["last_used_at"]),
        )
