# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist labeled workflows, histories, images, and hashes for visual review."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from tools.comfy_api import ImageReference, JsonObject

from .matrix import (
    CFG,
    HEIGHT,
    SAMPLER,
    SCHEDULER,
    SEED,
    STEPS,
    WIDTH,
    VisualOutputProfile,
)


class VisualOutputResultRecorder:
    """Own the exact profile-to-artifact association for one managed run."""

    def __init__(self, root: Path) -> None:
        """Retain an existing collision-resistant managed-run directory."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("Visual output result root must already exist.")
        self._observations: list[JsonObject] = []

    def record(
        self,
        profile: VisualOutputProfile,
        *,
        workflow: dict[str, JsonObject],
        history: JsonObject,
        prompt_id: str,
        image_reference: ImageReference,
        image_bytes: bytes,
    ) -> Path:
        """Persist one decoded image and its complete labeled evidence."""

        if any(
            item.get("profile_id") == profile.profile_id for item in self._observations
        ):
            raise ValueError(
                f"Visual profile was already recorded: {profile.profile_id}"
            )
        image_path = self._root / f"{profile.profile_id}.png"
        history_path = self._root / f"{profile.profile_id}.history.json"
        workflow_path = self._root / f"{profile.profile_id}.workflow.json"
        image_path.write_bytes(image_bytes)
        self._write_json(history_path, history)
        self._write_json(workflow_path, workflow)
        observation: JsonObject = {
            "profile_id": profile.profile_id,
            "label": profile.label,
            "execution_kind": profile.kind.value,
            "global_adapters": [asdict(adapter) for adapter in profile.global_adapters],
            "regional_adapters": [
                {
                    "region_index": adapter.region_index,
                    "branch": adapter.branch.value,
                    "lora_name": adapter.lora_name,
                    "strength": adapter.strength,
                }
                for adapter in profile.regional_adapters
            ],
            "mask_case_id": profile.mask_case_id,
            "seed": SEED,
            "width": WIDTH,
            "height": HEIGHT,
            "steps": STEPS,
            "cfg": CFG,
            "sampler": SAMPLER,
            "scheduler": SCHEDULER,
            "prompt_id": prompt_id,
            "image_file": image_path.name,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_size_bytes": len(image_bytes),
            "image_reference": {
                "filename": image_reference.filename,
                "subfolder": image_reference.subfolder,
                "type": image_reference.output_type,
            },
            "workflow_file": workflow_path.name,
            "history_file": history_path.name,
        }
        self._observations.append(observation)
        self._persist(status="in_progress")
        return image_path

    def finalize(self, profiles: tuple[VisualOutputProfile, ...]) -> Path:
        """Write a successful terminal result only for one-to-one profile coverage."""

        expected = tuple(profile.profile_id for profile in profiles)
        observed = tuple(str(item["profile_id"]) for item in self._observations)
        if observed != expected:
            raise ValueError(
                f"Visual output observations must match profile order: {observed!r}."
            )
        return self._persist(status="completed")

    def _persist(self, *, status: str) -> Path:
        """Atomically replace the authoritative labeled result record."""

        path = self._root / "visual-result.json"
        temporary = path.with_suffix(".json.tmp")
        payload: JsonObject = {
            "schema_version": 1,
            "status": status,
            "sampling": {
                "seed": SEED,
                "width": WIDTH,
                "height": HEIGHT,
                "steps": STEPS,
                "cfg": CFG,
                "sampler": SAMPLER,
                "scheduler": SCHEDULER,
            },
            "observations": self._observations,
        }
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        """Write one stable human-reviewable JSON sidecar."""

        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
