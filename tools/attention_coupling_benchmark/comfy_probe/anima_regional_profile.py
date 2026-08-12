"""Derive fixed benchmark-only Anima regional LoRA model profiles."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import comfy.model_management
import torch
from safetensors.torch import load_file

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.raw_regional_attention import (
    build_raw_regional_attention_plan,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.masking.regional_prompt_masks import build_regional_mask_bank
from simple_syrup.runtime.attention_coupling.anima_context import (
    ANIMA_REGIONAL_CONTEXT_VALIDATOR,
)
from simple_syrup.runtime.comfy_conditioning_processing import (
    COMFY_REGIONAL_CONDITIONING_PROCESSOR,
)
from simple_syrup.runtime.patcher_lifecycle import PATCHER_LIFECYCLE
from simple_syrup.runtime.regional_attention_batching import (
    REGIONAL_ATTENTION_BATCHING_SERVICE,
)
from simple_syrup.runtime.regional_lora.anima_attention_coupling import (
    anima_attention_coupling_mutations,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaRegionalLoraAdapterExecution,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import (
    ANIMA_MODULE_SURFACE_DISCOVERY,
)
from simple_syrup.runtime.regional_lora.anima_plan_admission import (
    ANIMA_REGIONAL_LORA_PLAN_ADMISSION_SERVICE,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora_plan_adapter import (
    RegionalLoraPlanAdaptation,
)
from simple_syrup.services.attention_coupling_preparation_service import (
    ATTENTION_COUPLING_PREPARATION_SERVICE,
)

from .anima_regional_profile_spec import VisualRegionalAdapter

_folder_paths: Any = import_module("folder_paths")


@dataclass(frozen=True, slots=True)
class BuiltStaticAnimaRegionalProfile:
    """Return the derived model and ordinary base sampler conditionings."""

    model: Any
    positive: object
    negative: object


class StaticAnimaRegionalProfileBuilder:
    """Build one static visual-evidence profile through production owners."""

    def build(
        self,
        *,
        model: Any,
        positive: object,
        negative: object,
        region_masks: object,
        latent: dict[str, Any],
        adapters: tuple[VisualRegionalAdapter, ...],
    ) -> BuiltStaticAnimaRegionalProfile:
        """Return a clone patched for one all-target regional LoRA profile."""

        if not isinstance(adapters, tuple) or not adapters:
            raise ValueError("Visual profile requires ordered regional adapters.")
        if any(not isinstance(adapter, VisualRegionalAdapter) for adapter in adapters):
            raise TypeError("Visual profile contains an invalid regional adapter.")
        canonical_positive = _canonical_conditioning_batch(positive, "positive")
        canonical_negative = _canonical_conditioning_batch(negative, "negative")
        samples = latent.get("samples")
        if not isinstance(samples, torch.Tensor) or samples.ndim not in (4, 5):
            raise TypeError("Visual profile latent must contain 4D or 5D samples.")
        mask_bank = build_regional_mask_bank(
            region_masks,
            feather=0,
            canvas_height=int(samples.shape[-2]),
            canvas_width=int(samples.shape[-1]),
        )
        paths = tuple(self._lora_path(adapter.lora_name) for adapter in adapters)
        weights = tuple(load_file(str(path), device="cpu") for path in paths)
        plan = self._plan(paths, adapters, model=model)
        adaptation = RegionalLoraPlanAdaptation(
            plan,
            weights,
        )
        admitted = ANIMA_REGIONAL_LORA_PLAN_ADMISSION_SERVICE.admit(adaptation)
        raw = build_raw_regional_attention_plan(
            positive=canonical_positive,
            negative=canonical_negative,
            mask_bank=mask_bank,
            lora_plan=plan,
        )
        preparation = ATTENTION_COUPLING_PREPARATION_SERVICE.prepare(raw)

        comfy.model_management.load_models_gpu([model], force_full_load=True)
        device = torch.device(model.load_device)
        processed = COMFY_REGIONAL_CONDITIONING_PROCESSOR.process(
            preparation,
            model=model,
            noise=samples.to(device),
            device=device,
            context_validator=ANIMA_REGIONAL_CONTEXT_VALIDATOR,
        )
        base_entry = processed.positive.base_context.entries[0]
        contexts = REGIONAL_ATTENTION_BATCHING_SERVICE.align(
            processed,
            base_context=base_entry.cross_attention,
            cond_or_uncond=[0],
            conditioning_uuids=[base_entry.uuid],
            sigma=0.0,
            latent_batch_size=int(samples.shape[0]),
        )
        attention = AnimaRegionalAttentionExecution(
            contexts,
            mask_bank,
            _unit_region_strengths(mask_bank.region_count),
        )
        cache = RegionalLoraExecutionCache()
        executions = tuple(
            AnimaRegionalLoraAdapterExecution(
                adapter.adapter_plan,
                adapter.admission,
                attention,
                model,
                cache,
            )
            for adapter in admitted.adapters
        )
        composition = AnimaRegionalLoraComposition(executions)
        surface = ANIMA_MODULE_SURFACE_DISCOVERY.discover(model.model.diffusion_model)
        derived = PATCHER_LIFECYCLE.derive_model(
            model,
            anima_attention_coupling_mutations(
                surface,
                attention,
                composition=composition,
            ),
            operation=f"visual regional Anima profile with {len(adapters)} adapters",
        )
        return BuiltStaticAnimaRegionalProfile(
            derived,
            preparation.positive,
            preparation.negative,
        )

    @staticmethod
    def _lora_path(lora_name: str) -> Path:
        """Resolve one exact Comfy LoRA filename through the host registry."""

        if not isinstance(lora_name, str) or not lora_name.strip():
            raise ValueError("Visual profile lora_name must be non-empty.")
        return Path(_folder_paths.get_full_path_or_raise("loras", lora_name))

    @staticmethod
    def _plan(
        paths: tuple[Path, ...],
        adapters: tuple[VisualRegionalAdapter, ...],
        *,
        model: Any,
    ) -> RegionalLoraPlan:
        """Build static equal-strength adapters in declared composition order."""

        maximum_sigma = float(model.model.model_sampling.percent_to_sigma(0.0))
        return RegionalLoraPlan(
            tuple(
                RegionalLoraAdapterPlan(
                    adapter_identity=RegionalLoraAdapterIdentity(
                        f"{paths[index]}#visual-{index}"
                    ),
                    composition_index=index,
                    region_index=adapter.region_index,
                    branch=adapter.branch,
                    model_strength=adapter.strength,
                    schedule=(
                        RegionalLoraScheduleBoundary(
                            start_percent=0.0,
                            start_sigma=maximum_sigma,
                            strength_multiplier=1.0,
                            guarantee_steps=0,
                        ),
                    ),
                )
                for index, adapter in enumerate(adapters)
            )
        )


def _unit_region_strengths(region_count: int) -> tuple[float, ...]:
    """Return one enabled regional strength for every canonical mask."""

    if type(region_count) is not int or region_count < 1:
        raise ValueError("Visual profile region count must be positive.")
    return (1.0,) * region_count


STATIC_ANIMA_REGIONAL_PROFILE_BUILDER = StaticAnimaRegionalProfileBuilder()


def _canonical_conditioning_batch(value: object, name: str) -> ConditioningBatch:
    """Adapt the separately loaded dev-extension class at its package boundary."""

    if isinstance(value, ConditioningBatch):
        return value
    entries = getattr(value, "entries", None)
    if not isinstance(entries, tuple) or not entries:
        raise TypeError(f"Visual profile {name} must be a non-empty ConditioningBatch.")
    return ConditioningBatch(entries)
