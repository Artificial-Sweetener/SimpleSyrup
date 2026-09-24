# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare owned SD/Anima behavior to the pinned local PPM baseline."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch

from simple_syrup.runtime.negpip.anima import (
    CONDITION_MASK_KEY,
    anima_attn2_negpip,
    anima_extra_conds_negpip_wrapper,
)
from simple_syrup.runtime.negpip.standard import encode_token_weights_negpip

PPM_REVISION = "6c6c360155cace9d7091306c1b8e26d9c7438620"


class _Encoder:
    """Produce deterministic embeddings for exact upstream parity."""

    special_tokens: dict[str, int] = {}

    def gen_empty_tokens(
        self,
        special_tokens: dict[str, int],
        length: int,
    ) -> list[int]:
        """Return one equal-length empty prompt."""

        del special_tokens
        return [0] * length

    def encode(self, sections: list[list[object]]) -> tuple[torch.Tensor, None]:
        """Map each integer token to one two-channel embedding."""

        rows = [
            [[float(cast(int, token)), float(cast(int, token)) + 0.25] for token in row]
            for row in sections
        ]
        return torch.tensor(rows), None


def prove_upstream_parity(ppm_root: Path) -> dict[str, object]:
    """Return exact revision and tensor equality evidence against local PPM."""

    resolved = ppm_root.resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolved,
        check=True,
        capture_output=True,
        text=True,
        timeout=30.0,
    ).stdout.strip()
    if revision != PPM_REVISION:
        raise ValueError(
            f"PPM baseline revision is {revision}, expected {PPM_REVISION}."
        )
    upstream_standard = _load_module(
        "simple_syrup_ppm_unet_negpip",
        resolved / "src" / "negpip" / "unet_negpip.py",
    )
    upstream_anima = _load_module(
        "simple_syrup_ppm_anima_negpip",
        resolved / "src" / "negpip" / "anima_negpip.py",
    )
    encoder = cast(Any, _Encoder())
    token_pairs: list[list[tuple[object, float]]] = [[(3, -2.0), (5, 0.5), (7, 1.0)]]
    owned_standard = encode_token_weights_negpip(encoder, token_pairs)
    baseline_standard = upstream_standard.encode_token_weights_negpip(
        encoder,
        token_pairs,
    )
    standard_equal = _tuple_tensors_equal(owned_standard, baseline_standard)

    def base_extra(**kwargs: object) -> dict[str, object]:
        return {"weights": kwargs["t5xxl_weights"]}

    weights = torch.tensor([-2.0, 0.5, 1.0])
    owned_extra = anima_extra_conds_negpip_wrapper(base_extra)(
        t5xxl_weights=weights.clone()
    )
    baseline_extra = upstream_anima.anima_extra_conds_negpip_wrapper(base_extra)(
        t5xxl_weights=weights.clone()
    )
    owned_mask = cast(Any, owned_extra[CONDITION_MASK_KEY]).cond
    baseline_mask = cast(Any, baseline_extra[CONDITION_MASK_KEY]).cond
    query = torch.ones((1, 1, 512, 2))
    key = query * 2
    value = query * 3
    owned_attention = anima_attn2_negpip(
        query,
        key,
        value,
        extra_options={"ppm_negpip_mask": owned_mask},
    )
    baseline_attention = upstream_anima.cosmos_attn2_negpip(
        query,
        key,
        value,
        extra_options={"ppm_negpip_mask": baseline_mask},
    )
    anima_equal = torch.equal(owned_mask, baseline_mask) and torch.equal(
        cast(torch.Tensor, owned_attention["v"]),
        cast(torch.Tensor, baseline_attention["v"]),
    )
    if not standard_equal or not anima_equal:
        raise ValueError("Owned NegPiP behavior diverged from the pinned PPM baseline.")
    return {
        "ppm_revision": revision,
        "standard_encoding_equal": standard_equal,
        "anima_mask_and_attention_equal": anima_equal,
        "standard_output_shape": list(cast(torch.Tensor, owned_standard[0]).shape),
        "anima_mask_shape": list(owned_mask.shape),
    }


def _load_module(name: str, path: Path) -> ModuleType:
    """Load one exact baseline file without registering its Comfy nodes."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PPM source file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tuple_tensors_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    """Compare the deterministic tensor/None result used by the parity fixture."""

    if len(left) != len(right):
        return False
    return all(
        torch.equal(left_item, right_item)
        if isinstance(left_item, torch.Tensor) and isinstance(right_item, torch.Tensor)
        else left_item == right_item
        for left_item, right_item in zip(left, right, strict=True)
    )
