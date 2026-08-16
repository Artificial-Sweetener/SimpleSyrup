# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify compact Anima self-attention work after composition is established."""

from __future__ import annotations

import torch

from simple_syrup.runtime.regional_self_attention_coherence import (
    RegionalSelfAttentionCoherencePolicy,
)
from simple_syrup.runtime.regional_self_attention_partition import (
    RegionalSelfAttentionPartitionPlan,
)


def test_hard_split_specialization_retains_global_composition_work() -> None:
    """Measure the accepted full-scene residual alongside regional attention."""

    height = width = 64
    owners = torch.cat(
        (
            torch.zeros((height, width // 2), dtype=torch.long),
            torch.ones((height, width // 2), dtype=torch.long),
        ),
        dim=1,
    ).reshape(1, -1)
    profile = RegionalSelfAttentionCoherencePolicy().resolve(
        owners,
        height=height,
        width=width,
    )
    plan = RegionalSelfAttentionPartitionPlan.build(profile)

    compact_pairs = sum(
        group.call_count * group.query_count * group.key_count
        for group in (*plan.regional_groups, *plan.global_groups)
    )
    global_pairs = (height * width) ** 2

    assert compact_pairs == 25_690_112
    assert compact_pairs > global_pairs
    assert sum(group.query_count for group in plan.global_groups) == 4096
