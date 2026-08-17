# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load and own exact external model visibility for complete Anima reruns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import TracebackType

from tools.anima_attention_coupling_prompts import (
    GLOBAL_GLOBAL_ADAPTER,
    PINNED_PRIMARY_ADAPTER,
    PINNED_QUATERNARY_ADAPTER,
    PINNED_SECONDARY_ADAPTER,
    PINNED_TERTIARY_ADAPTER,
)
from tools.comfy_integration.managed_model_links import (
    ManagedComfyModelLinks,
    ManagedModelLink,
)

ANIMA_ORACLE_ADAPTER_SELECTIONS = (
    PINNED_PRIMARY_ADAPTER,
    PINNED_SECONDARY_ADAPTER,
    PINNED_TERTIARY_ADAPTER,
    PINNED_QUATERNARY_ADAPTER,
    GLOBAL_GLOBAL_ADAPTER,
)


class ManagedOracleModelVisibility:
    """Validate an external inventory and own all declared Comfy aliases."""

    def __init__(self, links: ManagedComfyModelLinks) -> None:
        """Retain the sole generic link lifecycle owner."""

        self._links = links

    @property
    def cleaned(self) -> bool:
        """Report whether every owned model alias was removed."""

        return self._links.cleaned

    def __enter__(self) -> ManagedOracleModelVisibility:
        """Create all externally inventoried model aliases."""

        self._links.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Remove all model aliases even when a managed command fails."""

        self._links.__exit__(exc_type, exc, traceback)

    @classmethod
    def from_inventory(
        cls,
        inventory_path: Path,
        *,
        model_root: Path,
        required_selections: tuple[str, ...],
    ) -> ManagedOracleModelVisibility:
        """Build exact verified links from one machine-local JSON inventory."""

        links = _load_inventory(inventory_path)
        selections = tuple(link.selection_name for link in links)
        if len(set(selections)) != len(selections):
            raise ValueError("Model visibility inventory aliases must be unique.")
        if set(selections) != set(required_selections):
            raise ValueError(
                "Model visibility inventory must exactly cover declared aliases."
            )
        return cls(ManagedComfyModelLinks(model_root=model_root, links=links))


def anima_oracle_model_visibility(
    inventory_path: Path,
    *,
    model_root: Path,
) -> ManagedOracleModelVisibility:
    """Build the complete Anima oracle's declared adapter visibility scope."""

    return ManagedOracleModelVisibility.from_inventory(
        inventory_path,
        model_root=model_root,
        required_selections=ANIMA_ORACLE_ADAPTER_SELECTIONS,
    )


def _load_inventory(inventory_path: Path) -> tuple[ManagedModelLink, ...]:
    """Parse and hash-validate exact model-link declarations."""

    payload: object = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Model visibility inventory must be a non-empty JSON list.")
    return tuple(_parse_entry(entry) for entry in payload)


def _parse_entry(entry: object) -> ManagedModelLink:
    """Narrow and verify one external inventory entry."""

    if not isinstance(entry, dict):
        raise ValueError("Each model visibility inventory entry must be an object.")
    source_value = entry.get("source")
    category = entry.get("category")
    selection_name = entry.get("selection_name")
    size_bytes = entry.get("size_bytes")
    sha256 = entry.get("sha256")
    if not all(
        isinstance(value, str)
        for value in (source_value, category, selection_name, sha256)
    ) or not isinstance(size_bytes, int):
        raise ValueError("Model visibility inventory entry fields have invalid types.")
    assert isinstance(source_value, str)
    assert isinstance(category, str)
    assert isinstance(selection_name, str)
    assert isinstance(sha256, str)
    source = Path(source_value).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Inventoried model source is missing: {source}")
    if source.stat().st_size != size_bytes:
        raise ValueError(f"Inventoried model size changed: {source}")
    if _sha256(source) != sha256.lower():
        raise ValueError(f"Inventoried model hash changed: {source}")
    return ManagedModelLink(source, category, selection_name)


def _sha256(path: Path) -> str:
    """Return one file's streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
