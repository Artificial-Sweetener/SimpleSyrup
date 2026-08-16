# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load and validate an external P9.4 text-encoder LoRA fixture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tools.comfy_api import JsonObject

_DIGEST_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class TextEncoderLoraFixtureIdentity:
    """Describe one immutable external LoRA evidence artifact."""

    lora_name: str
    path: Path
    sha256: str
    size_bytes: int
    tensor_count: int
    source_sha256: str
    transformation: str

    def __post_init__(self) -> None:
        """Reject incomplete or ambiguous fixture identities."""

        if not self.lora_name or Path(self.lora_name).is_absolute():
            raise ValueError("P9.4 fixture LoRA name must be non-empty and relative.")
        if not self.path.is_absolute():
            raise ValueError("P9.4 fixture path must be absolute.")
        for label, digest in (
            ("fixture", self.sha256),
            ("source", self.source_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"P9.4 {label} SHA-256 must be lowercase hex.")
        if self.size_bytes <= 0 or self.tensor_count <= 0:
            raise ValueError("P9.4 fixture size and tensor count must be positive.")
        if not self.transformation:
            raise ValueError("P9.4 fixture transformation must be recorded.")

    def as_record(self) -> JsonObject:
        """Return deterministic fixture provenance for terminal evidence."""

        return {
            "lora_name": self.lora_name,
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "tensor_count": self.tensor_count,
            "source_sha256": self.source_sha256,
            "transformation": self.transformation,
        }


def load_text_encoder_lora_fixture(path: Path) -> TextEncoderLoraFixtureIdentity:
    """Load one artifact identity from an explicit untracked JSON inventory."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("P9.4 fixture inventory must be a JSON object.")
    try:
        return TextEncoderLoraFixtureIdentity(
            lora_name=str(payload["lora_name"]),
            path=Path(str(payload["path"])),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
            tensor_count=int(payload["tensor_count"]),
            source_sha256=str(payload["source_sha256"]),
            transformation=str(payload["transformation"]),
        )
    except KeyError as error:
        raise ValueError(
            f"P9.4 fixture inventory is missing {error.args[0]!r}."
        ) from error


def validate_text_encoder_lora_fixture(
    identity: TextEncoderLoraFixtureIdentity,
) -> TextEncoderLoraFixtureIdentity:
    """Verify the inventoried bytes without deserializing model tensors."""

    if not isinstance(identity, TextEncoderLoraFixtureIdentity):
        raise TypeError("P9.4 fixture preflight requires a fixture identity.")
    if not identity.path.is_file():
        raise FileNotFoundError(
            f"P9.4 text-encoder LoRA fixture is missing: {identity.path}"
        )
    actual_size = identity.path.stat().st_size
    if actual_size != identity.size_bytes:
        raise ValueError(
            "P9.4 text-encoder LoRA fixture size mismatch: "
            f"expected {identity.size_bytes}, got {actual_size}."
        )
    digest = hashlib.sha256()
    with identity.path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_DIGEST_CHUNK_SIZE), b""):
            digest.update(chunk)
    actual_digest = digest.hexdigest()
    if actual_digest != identity.sha256:
        raise ValueError(
            "P9.4 text-encoder LoRA fixture SHA-256 mismatch: "
            f"expected {identity.sha256}, got {actual_digest}."
        )
    return identity
