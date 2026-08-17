# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare CPU and selected-device regional materialization without sampling."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass

import torch
from comfy.model_patcher import ModelPatcher

from simple_syrup.domain.raw_regional_attention import (
    build_raw_regional_attention_plan,
)
from simple_syrup.masking.regional_prompt_masks import build_regional_mask_bank
from simple_syrup.runtime.comfy_conditioning_model_loader import (
    COMFY_CONDITIONING_MODEL_LOADER,
)
from simple_syrup.runtime.regional_lora.standard_unet_native_admission import (
    STANDARD_UNET_NATIVE_LORA_ADMISSION_SERVICE,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_materialization import (
    STANDARD_UNET_VARIANT_MATERIALIZER,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_topology import (
    STANDARD_UNET_VARIANT_TOPOLOGY_BUILDER,
    StandardUnetRegionalVariant,
)
from simple_syrup.runtime.regional_lora_conditioning_adapter import (
    REGIONAL_LORA_CONDITIONING_ADAPTER,
)

from .conditioning_batch_bridge import normalize_conditioning_batch
from .materialized_variant_comparison import (
    MaterializedVariantComparison,
    compare_materialized_variants,
)


@dataclass(frozen=True, slots=True)
class MaterializationParityVariantObservation:
    """Retain one region's materialization timing and numerical comparison."""

    region_index: int
    selected_device_materialization_ms: float
    cpu_materialization_ms: float
    comparison_ms: float
    comparison: MaterializedVariantComparison


@dataclass(frozen=True, slots=True)
class MaterializationParityObservation:
    """Retain one complete architecture-neutral materialization comparison."""

    selected_device: str
    source_load_ms: float
    admission_ms: float
    peak_vram_bytes: int
    variants: tuple[MaterializationParityVariantObservation, ...]

    @property
    def exact(self) -> bool:
        """Report whether every compared variant bank is byte-identical."""

        return all(variant.comparison.exact for variant in self.variants)

    def as_json_object(self) -> dict[str, object]:
        """Return one stable JSON-safe diagnostic object."""

        variant_payloads = []
        for variant in self.variants:
            payload = asdict(variant)
            comparison = dict(payload["comparison"])
            comparison["exact"] = variant.comparison.exact
            payload["comparison"] = comparison
            variant_payloads.append(payload)
        element_count = sum(
            variant.comparison.element_count for variant in self.variants
        )
        differing_count = sum(
            variant.comparison.differing_element_count for variant in self.variants
        )
        absolute_error_sum = sum(
            variant.comparison.mean_absolute_error * variant.comparison.element_count
            for variant in self.variants
        )
        squared_error_sum = sum(
            variant.comparison.root_mean_squared_error**2
            * variant.comparison.element_count
            for variant in self.variants
        )
        return {
            "selected_device": self.selected_device,
            "source_load_ms": self.source_load_ms,
            "admission_ms": self.admission_ms,
            "peak_vram_bytes": self.peak_vram_bytes,
            "variant_count": len(self.variants),
            "element_count": element_count,
            "differing_element_count": differing_count,
            "max_absolute_error": max(
                (variant.comparison.max_absolute_error for variant in self.variants),
                default=0.0,
            ),
            "mean_absolute_error": (
                absolute_error_sum / element_count if element_count else 0.0
            ),
            "root_mean_squared_error": (
                math.sqrt(squared_error_sum / element_count) if element_count else 0.0
            ),
            "exact": self.exact,
            "variants": variant_payloads,
        }


class MaterializationParityProbe:
    """Resolve once and compare each complete regional bank on CPU and device."""

    def compare(
        self,
        *,
        model: object,
        positive: object,
        negative: object,
        region_masks: object,
        latent_image: object,
        region_mask_feather: int,
    ) -> MaterializationParityObservation:
        """Return bounded parity evidence without sampling or mutating the source."""

        patcher = self._require_patcher(model)
        samples = self._latent_samples(latent_image)
        selected_device = patcher.load_device
        if selected_device.type == "cpu":
            raise ValueError(
                "Materialization parity requires a non-CPU selected load device."
            )
        self._synchronize(selected_device)
        torch.cuda.reset_peak_memory_stats(selected_device)
        mask_bank = build_regional_mask_bank(
            region_masks,
            feather=region_mask_feather,
            canvas_height=int(samples.shape[-2]),
            canvas_width=int(samples.shape[-1]),
        )
        raw = build_raw_regional_attention_plan(
            positive=normalize_conditioning_batch(positive),
            negative=normalize_conditioning_batch(negative),
            mask_bank=mask_bank,
        )
        started = time.perf_counter_ns()
        COMFY_CONDITIONING_MODEL_LOADER.load(patcher)
        self._synchronize(selected_device)
        source_load_ms = self._elapsed_ms(started)
        started = time.perf_counter_ns()
        adaptation = REGIONAL_LORA_CONDITIONING_ADAPTER.adapt(
            raw,
            model=patcher.model,
        )
        admission = STANDARD_UNET_NATIVE_LORA_ADMISSION_SERVICE.admit(
            patcher,
            adaptation,
        )
        resolution = admission.resolution
        if resolution is None:
            raise ValueError("Materialization parity requires regional LoRA targets.")
        topology = STANDARD_UNET_VARIANT_TOPOLOGY_BUILDER.build(resolution)
        multipliers = self._time_invariant_multipliers(adaptation.plan.adapters)
        admission_ms = self._elapsed_ms(started)
        observations = tuple(
            self._compare_variant(
                patcher,
                variant,
                multipliers,
                selected_device=selected_device,
            )
            for variant in topology.variants
        )
        return MaterializationParityObservation(
            selected_device=str(selected_device),
            source_load_ms=source_load_ms,
            admission_ms=admission_ms,
            peak_vram_bytes=torch.cuda.max_memory_allocated(selected_device),
            variants=observations,
        )

    @staticmethod
    def _compare_variant(
        patcher: ModelPatcher,
        variant: StandardUnetRegionalVariant,
        multipliers: tuple[float, ...],
        *,
        selected_device: torch.device,
    ) -> MaterializationParityVariantObservation:
        """Materialize and release one aligned bank pair before advancing."""

        selected_patcher = patcher.clone()
        cpu_patcher = patcher.clone()
        cpu_patcher.load_device = torch.device("cpu")
        started = time.perf_counter_ns()
        selected_bank = STANDARD_UNET_VARIANT_MATERIALIZER.materialize(
            selected_patcher,
            variant,
            multipliers,
        )
        MaterializationParityProbe._synchronize(selected_device)
        selected_ms = MaterializationParityProbe._elapsed_ms(started)
        started = time.perf_counter_ns()
        cpu_bank = STANDARD_UNET_VARIANT_MATERIALIZER.materialize(
            cpu_patcher,
            variant,
            multipliers,
        )
        cpu_ms = MaterializationParityProbe._elapsed_ms(started)
        started = time.perf_counter_ns()
        comparison = compare_materialized_variants(cpu_bank, selected_bank)
        comparison_ms = MaterializationParityProbe._elapsed_ms(started)
        del selected_bank, cpu_bank, selected_patcher, cpu_patcher
        gc.collect()
        torch.cuda.empty_cache()
        return MaterializationParityVariantObservation(
            region_index=comparison.region_index,
            selected_device_materialization_ms=selected_ms,
            cpu_materialization_ms=cpu_ms,
            comparison_ms=comparison_ms,
            comparison=comparison,
        )

    @staticmethod
    def _time_invariant_multipliers(adapters: tuple[object, ...]) -> tuple[float, ...]:
        """Return one exact effective value for every canonical adapter use."""

        values: list[float] = []
        for adapter in adapters:
            schedule = getattr(adapter, "schedule", None)
            if not isinstance(schedule, tuple) or not schedule:
                raise ValueError("Materialization parity requires adapter schedules.")
            multipliers = tuple(
                getattr(boundary, "strength_multiplier", None) for boundary in schedule
            )
            first = multipliers[0]
            if not isinstance(first, float) or any(
                multiplier != first for multiplier in multipliers
            ):
                raise ValueError(
                    "Materialization parity requires time-invariant schedules."
                )
            values.append(first)
        if not values:
            raise ValueError("Materialization parity requires regional adapters.")
        return tuple(values)

    @staticmethod
    def _require_patcher(model: object) -> ModelPatcher:
        """Narrow one Comfy MODEL and its selected CUDA-capable device."""

        if not isinstance(model, ModelPatcher):
            raise TypeError("Materialization parity requires a Comfy MODEL.")
        if not isinstance(model.load_device, torch.device):
            raise TypeError("Materialization parity load device must be a device.")
        if model.load_device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Materialization parity requires available CUDA.")
        return model

    @staticmethod
    def _latent_samples(latent_image: object) -> torch.Tensor:
        """Return one valid floating latent tensor for mask geometry."""

        if not isinstance(latent_image, dict):
            raise TypeError("Materialization parity latent must be a dictionary.")
        samples = latent_image.get("samples")
        if not isinstance(samples, torch.Tensor) or not samples.is_floating_point():
            raise TypeError("Materialization parity latent must contain samples.")
        if samples.ndim not in (4, 5):
            raise ValueError("Materialization parity latent samples must be 4D or 5D.")
        return samples

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        """Complete selected-device work before recording a boundary."""

        torch.cuda.synchronize(device)

    @staticmethod
    def _elapsed_ms(started_at_ns: int) -> float:
        """Return monotonic milliseconds since one captured boundary."""

        return (time.perf_counter_ns() - started_at_ns) / 1_000_000.0


MATERIALIZATION_PARITY_PROBE = MaterializationParityProbe()
