# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Process regional Attention Coupling contexts through installed Comfy APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from uuid import UUID

import torch
from comfy import sampler_helpers, samplers

from ..domain.conditioning_schedule import ConditioningScheduleRange
from ..domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from ..domain.raw_regional_attention import (
    RawRegionalAttentionBranch,
)
from ..services.attention_coupling_preparation_service import (
    AttentionCouplingPreparation,
)
from .attention_coupling.context_validation import RegionalContextValidator
from .ppm_negpip_interop import PpmNegpipInterop


class ComfyRegionalConditioningProcessor:
    """Convert every raw context through the same model path as Comfy sampling."""

    def process(
        self,
        preparation: AttentionCouplingPreparation,
        *,
        model: object,
        noise: torch.Tensor,
        device: torch.device,
        context_validator: RegionalContextValidator,
        negpip: PpmNegpipInterop | None = None,
    ) -> ProcessedRegionalAttentionPlan:
        """Return model-ready positive and negative context banks."""

        if not isinstance(preparation, AttentionCouplingPreparation):
            raise TypeError("Regional context processing requires a preparation.")
        if not isinstance(noise, torch.Tensor) or noise.ndim not in (4, 5):
            raise TypeError(
                "Regional context processing noise must be a 4D or 5D tensor."
            )
        if int(noise.shape[0]) < 1:
            raise ValueError(
                "Regional context processing noise batch must be positive."
            )
        if not isinstance(device, torch.device):
            raise TypeError("Regional context processing device must be torch.device.")
        if not isinstance(context_validator, RegionalContextValidator):
            raise TypeError("Regional context validator has an invalid type.")
        if negpip is not None and not isinstance(negpip, PpmNegpipInterop):
            raise TypeError("Regional conditioning NegPiP state has an invalid type.")
        base_model = getattr(model, "model", None)
        extra_conds = getattr(base_model, "extra_conds", None)
        if not callable(extra_conds):
            raise TypeError(
                "MODEL must expose model.extra_conds for context processing."
            )
        model_function = cast(Callable[..., dict[str, object]], extra_conds)

        self._preflight_branch("positive", preparation.plan.positive)
        self._preflight_branch("negative", preparation.plan.negative)
        positive = self._process_branch(
            preparation.plan.positive,
            prompt_type="positive",
            model=base_model,
            model_function=model_function,
            noise=noise,
            device=device,
            context_validator=context_validator,
            negpip=negpip,
        )
        negative = self._process_branch(
            preparation.plan.negative,
            prompt_type="negative",
            model=base_model,
            model_function=model_function,
            noise=noise,
            device=device,
            context_validator=context_validator,
            negpip=negpip,
        )
        return ProcessedRegionalAttentionPlan(
            positive=positive,
            negative=negative,
            mask_bank=preparation.plan.mask_bank,
            lora_plan=preparation.plan.lora_plan,
        )

    def _process_branch(
        self,
        branch: RawRegionalAttentionBranch,
        *,
        prompt_type: str,
        model: object,
        model_function: Callable[..., dict[str, object]],
        noise: torch.Tensor,
        device: torch.device,
        context_validator: RegionalContextValidator,
        negpip: PpmNegpipInterop | None,
    ) -> ProcessedRegionalAttentionBranch:
        """Process one base plus its ordered regional context bank."""

        base = self._process_context(
            branch.base_conditioning,
            conditioning_index=0,
            region_index=None,
            prompt_type=prompt_type,
            model=model,
            model_function=model_function,
            noise=noise,
            device=device,
            context_validator=context_validator,
            negpip=negpip,
        )
        regional = tuple(
            self._process_context(
                context.conditioning,
                conditioning_index=context.conditioning_index,
                region_index=context.region_index,
                prompt_type=prompt_type,
                model=model,
                model_function=model_function,
                noise=noise,
                device=device,
                context_validator=context_validator,
                negpip=negpip,
            )
            for context in branch.regional_contexts
        )
        return ProcessedRegionalAttentionBranch(base, regional)

    def _process_context(
        self,
        conditioning: object,
        *,
        conditioning_index: int,
        region_index: int | None,
        prompt_type: str,
        model: object,
        model_function: Callable[..., dict[str, object]],
        noise: torch.Tensor,
        device: torch.device,
        context_validator: RegionalContextValidator,
        negpip: PpmNegpipInterop | None,
    ) -> ProcessedRegionalAttentionContext:
        """Convert and extract one exact post-adapter Anima context tensor."""

        converted = sampler_helpers.convert_cond(conditioning)
        samplers.calculate_start_end_timesteps(model, converted)
        encoded = samplers.encode_model_conds(
            model_function,
            converted,
            noise,
            device,
            prompt_type,
        )
        if not isinstance(encoded, list) or not encoded:
            raise ValueError(
                f"{prompt_type} conditioning {conditioning_index} must convert "
                "to at least one Comfy condition."
            )
        entries = tuple(
            self._process_entry(
                encoded_item,
                entry_index=entry_index,
                conditioning_index=conditioning_index,
                prompt_type=prompt_type,
                context_validator=context_validator,
                negpip=negpip,
            )
            for entry_index, encoded_item in enumerate(encoded)
        )
        return ProcessedRegionalAttentionContext(
            conditioning_index=conditioning_index,
            region_index=region_index,
            entries=entries,
        )

    @staticmethod
    def _process_entry(
        encoded_item: object,
        *,
        entry_index: int,
        conditioning_index: int,
        prompt_type: str,
        context_validator: RegionalContextValidator,
        negpip: PpmNegpipInterop | None,
    ) -> ProcessedRegionalAttentionEntry:
        """Extract one exact post-adapter Anima context and Comfy strength."""

        if not isinstance(encoded_item, dict):
            raise TypeError("Comfy encoded conditioning item must be a dictionary.")
        model_conds = encoded_item.get("model_conds")
        if not isinstance(model_conds, dict):
            raise TypeError("Comfy encoded conditioning must contain model_conds.")
        cross_attention = model_conds.get("c_crossattn")
        context = getattr(cross_attention, "cond", None)
        if not isinstance(context, torch.Tensor):
            raise TypeError(
                f"{prompt_type} conditioning {conditioning_index} did not produce "
                "a tensor c_crossattn condition."
            )
        if context.ndim != 3:
            raise ValueError(
                f"{prompt_type} conditioning {conditioning_index} c_crossattn "
                "must use BxSxD layout."
            )
        context_validator.validate(
            context,
            prompt_type=prompt_type,
            conditioning_index=conditioning_index,
        )
        strength = encoded_item.get("strength", 1.0)
        if isinstance(strength, bool) or not isinstance(strength, int | float):
            raise TypeError(
                f"{prompt_type} conditioning {conditioning_index} entry "
                f"{entry_index} strength must be a real number."
            )
        entry_uuid = encoded_item.get("uuid")
        if not isinstance(entry_uuid, UUID):
            raise TypeError(
                f"{prompt_type} conditioning {conditioning_index} entry "
                f"{entry_index} must retain a Comfy UUID."
            )
        return ProcessedRegionalAttentionEntry(
            entry_index=entry_index,
            uuid=entry_uuid,
            schedule=ConditioningScheduleRange(
                _optional_boundary(encoded_item, "start_percent"),
                _optional_boundary(encoded_item, "end_percent"),
                _optional_boundary(encoded_item, "timestep_start"),
                _optional_boundary(encoded_item, "timestep_end"),
            ),
            cross_attention=context,
            strength=float(strength),
            cross_attention_value_multiplier=(
                None
                if negpip is None
                else negpip.extract_value_multiplier(model_conds, context)
            ),
        )

    @staticmethod
    def _preflight_branch(
        prompt_type: str,
        branch: RawRegionalAttentionBranch,
    ) -> None:
        """Require every authored context to contain at least one Comfy entry."""

        indexed = ((0, branch.base_conditioning),) + tuple(
            (context.conditioning_index, context.conditioning)
            for context in branch.regional_contexts
        )
        for conditioning_index, conditioning in indexed:
            if not isinstance(conditioning, list) or not conditioning:
                raise ValueError(
                    f"{prompt_type} conditioning {conditioning_index} must contain "
                    "at least one Comfy condition."
                )


COMFY_REGIONAL_CONDITIONING_PROCESSOR = ComfyRegionalConditioningProcessor()


def _optional_boundary(values: dict[str, object], key: str) -> float | None:
    """Narrow one optional Comfy schedule boundary for the domain owner."""

    value = values.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Comfy conditioning {key} must be a real number.")
    return float(value)
