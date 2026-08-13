# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist complete ordered Anima regression-oracle results atomically."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .artifact_validation import ArtifactObservation
from .execution import CommandObservation


class AnimaRegressionResultRecorder:
    """Own the U1 oracle journal and terminal result publication."""

    def __init__(self, root: Path) -> None:
        """Create one previously absent result directory."""

        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=False)
        self._started_at = _utc_now()

    @property
    def root(self) -> Path:
        """Return the exact owned result directory."""

        return self._root

    def publish(
        self,
        *,
        level: str,
        artifacts: tuple[ArtifactObservation, ...],
        commands: tuple[CommandObservation, ...],
        expected_command_count: int,
    ) -> Path:
        """Publish success only when every requested command passed."""

        passed = len(commands) == expected_command_count and all(
            command.return_code == 0 for command in commands
        )
        payload = {
            "schema_version": 1,
            "status": "completed" if passed else "failed",
            "level": level,
            "started_at_utc": self._started_at,
            "completed_at_utc": _utc_now(),
            "artifact_observations": [asdict(item) for item in artifacts],
            "command_observations": [asdict(item) for item in commands],
            "expected_command_count": expected_command_count,
        }
        destination = self._root / "anima-regression-oracle.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
        return destination


def _utc_now() -> str:
    """Return one stable UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
