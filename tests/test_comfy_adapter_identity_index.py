# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact normalized Comfy adapter identity indexing."""

from __future__ import annotations

import torch
from comfy.weight_adapter.lora import LoRAAdapter

from simple_syrup.runtime.regional_lora.comfy_adapter_identity_index import (
    ComfyAdapterIdentityIndex,
)


def test_identity_index_finds_roots_nested_containers_and_adapter_weights() -> None:
    """Traverse exactly the established tuple, list, and adapter-weight edges."""

    root = object()
    nested = object()
    adapter_weight = torch.ones(1)
    adapter = LoRAAdapter(
        {"adapter"},
        (adapter_weight, adapter_weight, None, None, None, None),
    )
    index = ComfyAdapterIdentityIndex.build((root, ("diff", [nested]), adapter))

    assert index.contains(root)
    assert index.contains(nested)
    assert index.contains(adapter)
    assert index.contains(adapter_weight)


def test_identity_index_uses_identity_and_does_not_descend_into_mappings() -> None:
    """Reject equality substitution and preserve the established mapping boundary."""

    retained = torch.ones(1)
    equal_but_distinct = torch.ones(1)
    nested_in_mapping = object()
    mapping = {"nested": nested_in_mapping}
    index = ComfyAdapterIdentityIndex.build((retained, mapping))

    assert index.contains(retained)
    assert not index.contains(equal_but_distinct)
    assert index.contains(mapping)
    assert not index.contains(nested_in_mapping)


def test_identity_index_handles_cyclic_admitted_containers() -> None:
    """Index cyclic host evidence once without recursive failure."""

    cycle: list[object] = []
    cycle.append(cycle)

    index = ComfyAdapterIdentityIndex.build((cycle,))

    assert index.contains(cycle)
