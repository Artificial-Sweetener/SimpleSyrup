"""Pin and validate the external P9.4 Anima text-encoder LoRA fixture."""

from __future__ import annotations

import hashlib
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


PINNED_TEXT_ENCODER_LORA_FIXTURE = TextEncoderLoraFixtureIdentity(
    lora_name=(
        "SimpleSyrup Evidence\\P9.4\\text_adapter_yoshiyuki_anima_qwen3_06b_te.safetensors"
    ),
    path=(
        Path(r"<MODEL_ROOT>\Loras\SimpleSyrup Evidence\P9.4")
        / "text_adapter_yoshiyuki_anima_qwen3_06b_te.safetensors"
    ),
    sha256="a1132f426fd7d29dfa70348ea68d10a1d3e882d859b76c951d664749ed212a79",
    size_bytes=10_202_136,
    tensor_count=588,
    source_sha256=("57040de66329a481a37d64fc23e63b304e1939c48863ab1fe60db14e8e78661d"),
    transformation="retain lora_te1 tensors and rename prefix to lora_te",
)


def validate_text_encoder_lora_fixture(
    identity: TextEncoderLoraFixtureIdentity,
) -> TextEncoderLoraFixtureIdentity:
    """Verify the exact pinned bytes without deserializing model tensors."""

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
