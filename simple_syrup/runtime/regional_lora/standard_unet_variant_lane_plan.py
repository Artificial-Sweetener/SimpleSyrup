# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plan exact active, regional-base, and global-base UNet lanes."""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.regional_mask_bank import RegionalMaskBank
from .standard_unet_variant_masks import StandardUnetVariantMaskProjector


@dataclass(frozen=True, slots=True)
class StandardUnetVariantLanePlan:
    """Retain ordered output regions and the required global-base policy."""

    output_region_indices: tuple[int, ...]
    regional_base_indices: tuple[int, ...]
    requires_global_base: bool

    def __post_init__(self) -> None:
        """Require canonical regional lanes and one nonempty output set."""

        if not self.output_region_indices:
            raise ValueError("Standard UNet lane plan requires output regions.")
        if self.output_region_indices != tuple(sorted(set(self.output_region_indices))):
            raise ValueError("Standard UNet output lanes must be unique and sorted.")
        if self.regional_base_indices != tuple(sorted(set(self.regional_base_indices))):
            raise ValueError("Standard UNet base lanes must be unique and sorted.")
        if not set(self.regional_base_indices).issubset(self.output_region_indices):
            raise ValueError("Standard UNet base lanes must own output regions.")


class StandardUnetVariantLanePlanner:
    """Distinguish inactive authored regions from unauthored canvas gaps."""

    @staticmethod
    def plan(
        bank: RegionalMaskBank,
        active_region_indices: tuple[int, ...],
    ) -> StandardUnetVariantLanePlan:
        """Return exact regional-base lanes or one required global base."""

        if not isinstance(bank, RegionalMaskBank):
            raise TypeError("Standard UNet lane planning requires a mask bank.")
        StandardUnetVariantMaskProjector._validate_regions(
            active_region_indices,
            bank.region_count,
        )
        all_regions = tuple(range(bank.region_count))
        has_canvas_gap = StandardUnetVariantMaskProjector.requires_base(
            bank,
            all_regions,
        )
        if has_canvas_gap:
            return StandardUnetVariantLanePlan(
                output_region_indices=active_region_indices,
                regional_base_indices=(),
                requires_global_base=True,
            )
        active = frozenset(active_region_indices)
        inactive = tuple(index for index in all_regions if index not in active)
        return StandardUnetVariantLanePlan(
            output_region_indices=all_regions,
            regional_base_indices=inactive,
            requires_global_base=False,
        )


STANDARD_UNET_VARIANT_LANE_PLANNER = StandardUnetVariantLanePlanner()
