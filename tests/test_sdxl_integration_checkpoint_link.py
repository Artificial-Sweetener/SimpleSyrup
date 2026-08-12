"""Verify exact temporary SDXL checkpoint link ownership."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from tools.sdxl_attention_coupling_integration.checkpoint_link import (
    CheckpointArtifactIdentity,
    ManagedCheckpointLink,
)


def test_checkpoint_link_uses_normal_model_directory_and_cleans_exact_target(
    tmp_path: Path,
) -> None:
    """Create one hard link without copying or retaining external state."""

    model_root, source, source_name, identity = _fixture(tmp_path)
    link = ManagedCheckpointLink(
        source=source,
        source_checkpoint_name=source_name,
        identity=identity,
    )

    with link:
        target = model_root / link.checkpoint_name
        assert target.is_file()
        assert os.path.samefile(source, target)
        assert link.checkpoint_name == "simple_syrup_p8_5_sdxl\\fixture.safetensors"

    assert link.cleaned
    assert not target.exists()
    assert not target.parent.exists()
    assert source.read_bytes() == b"checkpoint-fixture"


def test_checkpoint_link_rejects_identity_mismatch_before_creation(
    tmp_path: Path,
) -> None:
    """Fail closed before creating a model-directory entry on digest drift."""

    model_root, source, source_name, identity = _fixture(tmp_path)
    link = ManagedCheckpointLink(
        source=source,
        source_checkpoint_name=source_name,
        identity=CheckpointArtifactIdentity(
            identity.stable_name,
            identity.size,
            "0" * 64,
        ),
    )

    with pytest.raises(ValueError, match="SHA-256"):
        with link:
            raise AssertionError("invalid checkpoint must never enter")

    target = model_root / link.checkpoint_name
    assert not target.exists()


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, str, CheckpointArtifactIdentity]:
    """Return one small same-volume checkpoint and normal Comfy directory."""

    model_root = tmp_path / "configured-model-root" / "checkpoints"
    source_directory = model_root / "SDXL"
    source_directory.mkdir(parents=True)
    source = source_directory / "fixture.safetensors"
    source.write_bytes(b"checkpoint-fixture")
    return (
        model_root,
        source,
        "SDXL\\fixture.safetensors",
        CheckpointArtifactIdentity(
            "fixture.safetensors",
            source.stat().st_size,
            hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
    )
