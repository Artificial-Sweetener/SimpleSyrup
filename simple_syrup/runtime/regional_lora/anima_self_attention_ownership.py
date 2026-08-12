# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve authored Anima masks into deterministic image-token ownership."""

from __future__ import annotations

import torch


class AnimaSelfAttentionOwnershipPolicy:
    """Build image-token communication rules from projected regional masks."""

    def token_owners(self, masks: torch.Tensor) -> torch.Tensor:
        """Return B/Q region indices with -1 identifying shared base tokens."""

        self._validate_masks(masks)
        maximum, owners = masks.max(dim=0)
        return torch.where(maximum > 0.0, owners, owners.new_full((), -1))

    def allowed_relation(self, masks: torch.Tensor) -> torch.Tensor:
        """Return B/Q/K truth values for same-owner or shared-token attention."""

        return self.relation(self.token_owners(masks))

    @staticmethod
    def relation(owners: torch.Tensor) -> torch.Tensor:
        """Return B/Q/K truth values from validated integer token owners."""

        if (
            not isinstance(owners, torch.Tensor)
            or owners.ndim != 2
            or owners.dtype != torch.long
        ):
            raise TypeError("Anima attention owners must use integer B/Q layout.")
        query_owners = owners.unsqueeze(2)
        key_owners = owners.unsqueeze(1)
        return query_owners.eq(key_owners) | query_owners.eq(-1) | key_owners.eq(-1)

    @staticmethod
    def _validate_masks(masks: torch.Tensor) -> None:
        """Require one finite normalized R/B/Q mask tensor."""

        if not isinstance(masks, torch.Tensor):
            raise TypeError("Anima attention ownership masks must be a tensor.")
        if masks.ndim != 3 or any(int(size) < 1 for size in masks.shape):
            raise ValueError("Anima attention ownership masks must use R/B/Q layout.")
        if not masks.is_floating_point():
            raise TypeError("Anima attention ownership masks must be floating point.")
        if not bool(torch.isfinite(masks).all()) or not bool(
            ((masks >= 0.0) & (masks <= 1.0)).all()
        ):
            raise ValueError(
                "Anima attention ownership masks require finite values in the "
                "inclusive range [0, 1]."
            )


ANIMA_SELF_ATTENTION_OWNERSHIP_POLICY = AnimaSelfAttentionOwnershipPolicy()
