# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate immutable accepted Anima artifacts without regenerating them."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .manifest import AnimaRegressionManifest, CompletedJsonEvidence


@dataclass(frozen=True, slots=True)
class ArtifactObservation:
    """Record one successfully validated accepted evidence identity."""

    identity: str
    path: str
    sha256: str


class AcceptedArtifactValidator:
    """Require exact accepted hashes and structural managed-run evidence."""

    def validate(
        self, manifest: AnimaRegressionManifest
    ) -> tuple[ArtifactObservation, ...]:
        """Validate every artifact in manifest order or fail with its identity."""

        observations: list[ArtifactObservation] = []
        for image in manifest.images:
            digest = _file_sha256(image.path)
            if digest != image.sha256:
                raise ValueError(f"Accepted image hash changed for {image.identity}.")
            observations.append(
                ArtifactObservation(image.identity, str(image.path), digest)
            )
        for evidence in manifest.managed_results:
            payload = _json_object(evidence.path)
            details = tuple(_json_object(path) for path in evidence.detail_paths)
            self._validate_completed(evidence, payload, details)
            observations.append(
                ArtifactObservation(
                    evidence.identity, str(evidence.path), _file_sha256(evidence.path)
                )
            )
        self._validate_performance(manifest)
        observations.append(
            ArtifactObservation(
                "accepted-performance-gate",
                str(manifest.performance.path),
                _file_sha256(manifest.performance.path),
            )
        )
        return tuple(observations)

    @staticmethod
    def _validate_completed(
        evidence: CompletedJsonEvidence,
        payload: dict[str, object],
        details: tuple[dict[str, object], ...],
    ) -> None:
        """Require completion, cleanup, matrix shape, and pinned values."""

        if payload.get("status") != "completed":
            raise ValueError(f"Managed evidence is incomplete: {evidence.identity}.")
        if not _cleanup_passed(payload):
            raise ValueError(f"Managed cleanup failed: {evidence.identity}.")
        if evidence.observation_count is not None:
            observations = payload.get("observations")
            if (
                not isinstance(observations, list)
                or len(observations) != evidence.observation_count
            ):
                raise ValueError(f"Managed matrix changed: {evidence.identity}.")
        values = tuple(
            item
            for document in (payload, *details)
            for item in _walk_key_values(document)
        )
        for key, required, minimum_count in evidence.required_integer_occurrences:
            count = sum(
                found_key == key and value == required for found_key, value in values
            )
            if count < minimum_count:
                raise ValueError(
                    f"Managed evidence lacks {minimum_count} occurrences of "
                    f"{key}={required}: {evidence.identity}."
                )
        serialized = json.dumps((payload, *details), sort_keys=True)
        for digest in evidence.required_image_hashes:
            if digest not in serialized:
                raise ValueError(
                    f"Managed evidence lacks image hash {digest}: {evidence.identity}."
                )

    @staticmethod
    def _validate_performance(manifest: AnimaRegressionManifest) -> None:
        """Require the exact accepted workload, limits, and passing profiles."""

        evidence = manifest.performance
        payload = _json_object(evidence.path)
        if payload.get("passed") is not True:
            raise ValueError("Accepted Anima performance result does not pass.")
        environment = payload.get("environment")
        if (
            not isinstance(environment, dict)
            or environment.get("adapter_target_count") != evidence.adapter_target_count
        ):
            raise ValueError("Accepted Anima performance target surface changed.")
        profiles = payload.get("profiles")
        if not isinstance(profiles, list):
            raise ValueError("Accepted Anima performance profiles are missing.")
        observed: list[tuple[str, float, bool]] = []
        for profile in profiles:
            if not isinstance(profile, dict):
                raise TypeError("Anima performance profile must be an object.")
            identity = profile.get("profile_id")
            limit = profile.get("maximum_overhead_percent")
            passed = profile.get("passed")
            if (
                not isinstance(identity, str)
                or not isinstance(limit, int | float)
                or not isinstance(passed, bool)
            ):
                raise TypeError("Anima performance profile has invalid fields.")
            observed.append((identity, float(limit), passed))
        expected = tuple(
            (identity, limit, True) for identity, limit in evidence.profile_limits
        )
        if tuple(observed) != expected:
            raise ValueError("Accepted Anima performance limits or status changed.")


def _json_object(path: Path) -> dict[str, object]:
    """Read one required UTF-8 JSON object."""

    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in decoded
    ):
        raise TypeError(f"Accepted evidence must be a JSON object: {path}.")
    return decoded


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one required file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cleanup_passed(payload: dict[str, object]) -> bool:
    """Accept the two existing explicit managed-cleanup schemas."""

    if payload.get("cleanup_verified") is True:
        return True
    cleanup = payload.get("cleanup")
    return (
        isinstance(cleanup, dict)
        and bool(cleanup)
        and all(
            value is True or (key == "parent_running" and value is False)
            for key, value in cleanup.items()
        )
    )


def _walk_key_values(value: object) -> tuple[tuple[str, object], ...]:
    """Return every nested string-key value without interpreting its owner."""

    found: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str):
                found.append((key, child))
                found.extend(_walk_key_values(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_key_values(child))
    return tuple(found)
