# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Partition cohesive attention support around independently strong peaks."""

from __future__ import annotations

from importlib import import_module

import torch

from ..masking.mask_components import connected_mask_components


class AttentionInstanceSplittingService:
    """Separate peak instances without deleting accepted concept support."""

    def partition(
        self,
        *,
        alpha: torch.Tensor,
        support: torch.Tensor,
        minimum_strength: float,
        sensitivity: float,
    ) -> tuple[torch.Tensor, ...]:
        """Assign every supported pixel to a nearby peak-derived instance."""

        active = support.detach().to(device="cpu", dtype=torch.bool)
        if sensitivity <= 0.0:
            return (active.to(device=support.device),)
        peak_threshold = minimum_strength + (
            sensitivity * (1.0 - minimum_strength) * 0.35
        )
        peak_support = active & (
            alpha.detach().to(device="cpu", dtype=torch.float32) >= peak_threshold
        )
        peaks = connected_mask_components(peak_support)
        if len(peaks) <= 1:
            return (active.to(device=support.device),)

        source = torch.ones_like(active, dtype=torch.uint8)
        for peak in peaks:
            source[peak.mask] = 0
        cv2 = import_module("cv2")
        _distance, labels = cv2.distanceTransformWithLabels(
            source.numpy(),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_CCOMP,
        )
        active_labels = torch.from_numpy(labels).to(dtype=torch.int32)
        partitions: list[torch.Tensor] = []
        seen_labels: set[int] = set()
        for peak in peaks:
            label = int(active_labels[peak.mask][0].item())
            if label in seen_labels:
                continue
            seen_labels.add(label)
            partition = active & (active_labels == label)
            if partition.any():
                partitions.append(partition.to(device=support.device))
        return tuple(partitions) if partitions else (active.to(device=support.device),)


ATTENTION_INSTANCE_SPLITTING_SERVICE = AttentionInstanceSplittingService()
