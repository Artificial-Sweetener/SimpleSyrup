"""Prove exact byte-level identity checks for the P9.4 LoRA fixture."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.text_encoder_lora_integration.fixture import (
    TextEncoderLoraFixtureIdentity,
    validate_text_encoder_lora_fixture,
)


def test_fixture_preflight_accepts_exact_size_and_digest(tmp_path: Path) -> None:
    """Return the same immutable identity for exact fixture bytes."""

    path = tmp_path / "fixture.safetensors"
    path.write_bytes(b"fixture")
    identity = _identity(path, b"fixture")

    assert validate_text_encoder_lora_fixture(identity) is identity


def test_fixture_preflight_rejects_missing_or_wrong_type(tmp_path: Path) -> None:
    """Fail before server startup for absent or malformed identities."""

    with pytest.raises(TypeError, match="fixture identity"):
        validate_text_encoder_lora_fixture(object())  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError, match="fixture is missing"):
        validate_text_encoder_lora_fixture(_identity(tmp_path / "missing", b"fixture"))


def test_fixture_preflight_rejects_size_or_digest_mismatch(tmp_path: Path) -> None:
    """Detect changed external evidence bytes without tensor deserialization."""

    path = tmp_path / "fixture.safetensors"
    path.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_text_encoder_lora_fixture(_identity(path, b"fixture"))

    path.write_bytes(b"changed")
    identity = _identity(path, b"changed")
    wrong_digest = TextEncoderLoraFixtureIdentity(
        identity.lora_name,
        identity.path,
        hashlib.sha256(b"different").hexdigest(),
        identity.size_bytes,
        identity.tensor_count,
        identity.source_sha256,
        identity.transformation,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_text_encoder_lora_fixture(wrong_digest)


def _identity(path: Path, expected: bytes) -> TextEncoderLoraFixtureIdentity:
    """Build one complete temporary fixture identity."""

    return TextEncoderLoraFixtureIdentity(
        lora_name="evidence\\fixture.safetensors",
        path=path.resolve(),
        sha256=hashlib.sha256(expected).hexdigest(),
        size_bytes=len(expected),
        tensor_count=1,
        source_sha256=hashlib.sha256(b"source").hexdigest(),
        transformation="fixture",
    )
