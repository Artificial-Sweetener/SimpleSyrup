# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Inspect and validate the pinned ADAPTER_A safetensors target surface."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .matrix import PINNED_LORA_SHA256, PINNED_LORA_SIZE

_TARGET_PATTERN = re.compile(r"^diffusion_model\.blocks\.(\d+)\.(.+)$")
_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"


@dataclass(frozen=True)
class AdapterPair:
    """Describe one matched low-rank A/B target without loading its tensors."""

    target: str
    block_index: int
    family: str
    rank: int
    input_features: int
    output_features: int


@dataclass(frozen=True)
class AdapterInventory:
    """Record stable artifact identity, metadata, and complete target pairs."""

    size_bytes: int
    sha256: str
    metadata: dict[str, str]
    pairs: tuple[AdapterPair, ...]

    @property
    def target_keys(self) -> tuple[str, ...]:
        """Return every normalized model target in stable order."""

        return tuple(pair.target for pair in self.pairs)


def inspect_adapter(path: Path) -> AdapterInventory:
    """Read a safetensors header and fail closed on incomplete LoRA pairs."""

    if not path.is_file():
        raise FileNotFoundError(f"LoRA artifact does not exist: {path.name!r}.")
    size_bytes = path.stat().st_size
    sha256 = _sha256(path)
    pairs: list[AdapterPair] = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
        metadata_raw: Any = handle.metadata() or {}
        if not isinstance(metadata_raw, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata_raw.items()
        ):
            raise TypeError("LoRA safetensors metadata must contain string pairs.")
        metadata = dict(metadata_raw)
        grouped: dict[str, dict[str, str]] = defaultdict(dict)
        for key in keys:
            if key.endswith(_A_SUFFIX):
                grouped[key.removesuffix(_A_SUFFIX)]["A"] = key
            elif key.endswith(_B_SUFFIX):
                grouped[key.removesuffix(_B_SUFFIX)]["B"] = key
            else:
                raise ValueError(f"Unsupported pinned LoRA tensor key: {key!r}.")
        for target, pair_keys in sorted(grouped.items()):
            if set(pair_keys) != {"A", "B"}:
                raise ValueError(f"LoRA target has an incomplete A/B pair: {target!r}.")
            match = _TARGET_PATTERN.fullmatch(target)
            if match is None:
                raise ValueError(
                    f"LoRA target is not an Anima block target: {target!r}."
                )
            shape_a = tuple(handle.get_slice(pair_keys["A"]).get_shape())
            shape_b = tuple(handle.get_slice(pair_keys["B"]).get_shape())
            if len(shape_a) != 2 or len(shape_b) != 2 or shape_a[0] != shape_b[1]:
                raise ValueError(
                    f"LoRA target has incompatible A/B shapes: {target!r}."
                )
            pairs.append(
                AdapterPair(
                    target=target,
                    block_index=int(match.group(1)),
                    family=match.group(2),
                    rank=shape_a[0],
                    input_features=shape_a[1],
                    output_features=shape_b[0],
                )
            )
    return AdapterInventory(size_bytes, sha256, metadata, tuple(pairs))


def validate_pinned_inventory(inventory: AdapterInventory) -> None:
    """Require the exact pinned identity and complete 28-by-16 target surface."""

    if inventory.size_bytes != PINNED_LORA_SIZE:
        raise ValueError("Pinned ADAPTER_A LoRA size does not match the recorded fixture.")
    if inventory.sha256 != PINNED_LORA_SHA256:
        raise ValueError(
            "Pinned ADAPTER_A LoRA SHA-256 does not match the recorded fixture."
        )
    if len(inventory.pairs) != 448:
        raise ValueError("Pinned ADAPTER_A LoRA must contain exactly 448 adapter pairs.")
    families_by_block: dict[int, set[str]] = defaultdict(set)
    for pair in inventory.pairs:
        if pair.rank != 32:
            raise ValueError(f"Pinned ADAPTER_A target is not rank 32: {pair.target!r}.")
        families_by_block[pair.block_index].add(pair.family)
    if set(families_by_block) != set(range(28)):
        raise ValueError("Pinned ADAPTER_A LoRA must cover Anima blocks 0 through 27.")
    family_sets = {frozenset(families) for families in families_by_block.values()}
    if len(family_sets) != 1 or len(next(iter(family_sets))) != 16:
        raise ValueError("Every pinned ADAPTER_A block must contain the same 16 families.")


def _sha256(path: Path) -> str:
    """Hash one artifact with bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
