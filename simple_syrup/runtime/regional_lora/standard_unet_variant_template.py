# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cache exact static standard-UNet variant graphs by Comfy clone lineage."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from threading import Lock

import torch
from comfy.model_patcher import ModelPatcher
from torch import nn

from ...domain.regional_lora_plan import RegionalLoraAdapterPlan
from ..model_patcher_mutations import (
    ModelExactObjectPatchMutation,
    ModelSharedObjectPatchMutation,
)
from ..patcher_lifecycle import PATCHER_LIFECYCLE
from .execution_cache import ModelCloneLineage
from .standard_unet_cold_diagnostics import (
    STANDARD_UNET_COLD_PATH_DIAGNOSTICS,
    StandardUnetColdStage,
)
from .standard_unet_native_admission import StandardUnetNativeLoraAdmission
from .standard_unet_variant_execution_session import (
    StandardUnetVariantExecutionSession,
)
from .standard_unet_variant_forward import StandardUnetVariantForward
from .standard_unet_variant_materialization import (
    STANDARD_UNET_VARIANT_MATERIALIZER,
)
from .standard_unet_variant_root import (
    STANDARD_UNET_VARIANT_ROOT_BUILDER,
    StandardUnetVariantRoot,
)
from .standard_unet_variant_shell import STANDARD_UNET_VARIANT_SHELL_BUILDER
from .standard_unet_variant_static_residency import (
    StandardUnetStaticVariantResidency,
)
from .standard_unet_variant_topology import (
    STANDARD_UNET_VARIANT_TOPOLOGY_BUILDER,
    StandardUnetRegionalVariant,
    StandardUnetVariantTopology,
)


@dataclass(frozen=True, slots=True)
class StandardUnetVariantTemplateKey:
    """Identify one exact reusable model/adapter variant topology."""

    lineage: ModelCloneLineage
    adapters: tuple[RegionalLoraAdapterPlan, ...]
    target_signatures: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class _VariantStateKey:
    """Identify one region and exact local schedule multiplier state."""

    region_index: int
    multipliers: tuple[float, ...]


class StandardUnetVariantTemplate:
    """Own one static model, stable root, and all admitted schedule variants."""

    def __init__(
        self,
        model: ModelPatcher,
        source_diffusion: nn.Module,
        root: StandardUnetVariantRoot,
        topology: StandardUnetVariantTopology,
        adapter_count: int,
    ) -> None:
        """Retain stable graph ownership before preparing exact variants."""

        self.model = model
        self.source_diffusion = source_diffusion
        self.root = root
        self.topology = topology
        self.execution_session = StandardUnetVariantExecutionSession()
        self._adapter_count = adapter_count
        self._variants: dict[_VariantStateKey, nn.Module] = {}
        self._frozen = False

    def prepare(self) -> None:
        """Materialize every finite authored schedule state before model loading."""

        if self._frozen:
            raise RuntimeError("Standard UNet variant template is already prepared.")
        device = getattr(self.model, "load_device", None)
        measured_device = device if isinstance(device, torch.device) else None
        with STANDARD_UNET_COLD_PATH_DIAGNOSTICS.measure(
            StandardUnetColdStage.TEMPLATE_PREPARATION,
            device=measured_device,
        ) as metadata:
            for variant in self.topology.variants:
                multiplier_sets = tuple(
                    tuple(
                        dict.fromkeys(
                            boundary.strength_multiplier
                            for boundary in adapter.schedule
                        )
                    )
                    for adapter in variant.adapters
                )
                for local_values in product(*multiplier_sets):
                    if not any(
                        adapter.model_strength * multiplier != 0.0
                        for adapter, multiplier in zip(
                            variant.adapters,
                            local_values,
                            strict=True,
                        )
                    ):
                        continue
                    complete = [0.0] * self._adapter_count
                    for adapter, multiplier in zip(
                        variant.adapters,
                        local_values,
                        strict=True,
                    ):
                        complete[adapter.composition_index] = multiplier
                    self._prepare_variant(variant, tuple(complete))
            self.root.install_forward(
                StandardUnetVariantForward(self.root.module, self.execution_session)
            )
            self._frozen = True
            metadata["topology_variant_count"] = len(self.topology.variants)
            metadata["materialized_variant_count"] = len(self._variants)

    def bind_request(self, source: object) -> ModelPatcher:
        """Bind current request patcher state to the cache-stable static graph."""

        if not isinstance(source, ModelPatcher):
            raise TypeError("Standard UNet variant request requires a MODEL.")
        request = PATCHER_LIFECYCLE.derive_model_with_override(
            source,
            self.model,
            (),
            operation="standard UNet static variant request",
            disable_dynamic=True,
        )
        if not isinstance(request, ModelPatcher) or request.is_dynamic():
            raise TypeError("Comfy did not bind a static standard-UNet request.")
        if request.parent is not self.model:
            raise RuntimeError(
                "Static standard-UNet request lost its template fallback boundary."
            )
        if "diffusion_model" in request.object_patches_backup:
            ModelSharedObjectPatchMutation(
                "diffusion_model",
                self.source_diffusion,
                self.root.module,
            ).apply(request)
        else:
            ModelExactObjectPatchMutation(
                "diffusion_model",
                self.source_diffusion,
                self.root.module,
            ).apply(request)
        StandardUnetStaticVariantResidency(
            request,
            self.source_diffusion,
            self.root.module,
        ).apply(request)
        return request

    def resolve(
        self,
        variant: StandardUnetRegionalVariant,
        schedule_multipliers: tuple[float, ...],
    ) -> nn.Module:
        """Return a prebuilt resident variant or fail on undeclared schedule state."""

        if not self._frozen:
            raise RuntimeError("Standard UNet variant template is not prepared.")
        key = self._state_key(variant, schedule_multipliers)
        resolved = self._variants.get(key)
        if resolved is None:
            raise ValueError(
                "Standard UNet variant schedule state was not prepared before loading."
            )
        return resolved

    def _prepare_variant(
        self,
        variant: StandardUnetRegionalVariant,
        schedule_multipliers: tuple[float, ...],
    ) -> None:
        """Build and register one exact schedule-specific shell once."""

        key = self._state_key(variant, schedule_multipliers)
        if key in self._variants:
            return
        materialized = STANDARD_UNET_VARIANT_MATERIALIZER.materialize(
            self.model,
            variant,
            schedule_multipliers,
        )
        shell = STANDARD_UNET_VARIANT_SHELL_BUILDER.build(
            self.source_diffusion,
            materialized,
        )
        self.root.variants[f"variant_{len(self._variants):04d}"] = shell
        self._variants[key] = shell

    @staticmethod
    def _state_key(
        variant: StandardUnetRegionalVariant,
        schedule_multipliers: tuple[float, ...],
    ) -> _VariantStateKey:
        """Return the exact region-local multiplier key."""

        return _VariantStateKey(
            variant.region_index,
            tuple(
                float(schedule_multipliers[adapter.composition_index])
                for adapter in variant.adapters
            ),
        )


class StandardUnetVariantTemplateCache:
    """Bound process-local reusable static templates by semantic runtime key."""

    def __init__(self, maximum_entries: int = 2) -> None:
        """Create one bounded least-recently-used template cache."""

        if isinstance(maximum_entries, bool) or not isinstance(maximum_entries, int):
            raise TypeError("Standard UNet template cache size must be an integer.")
        if maximum_entries < 1:
            raise ValueError("Standard UNet template cache size must be positive.")
        self._maximum_entries = maximum_entries
        self._templates: OrderedDict[
            StandardUnetVariantTemplateKey,
            StandardUnetVariantTemplate,
        ] = OrderedDict()
        self._lock = Lock()

    def resolve(
        self,
        model: object,
        admission: StandardUnetNativeLoraAdmission,
    ) -> StandardUnetVariantTemplate:
        """Return one exact static template, preparing a bounded miss atomically."""

        if not isinstance(model, ModelPatcher):
            raise TypeError("Standard UNet variant template requires a MODEL.")
        if not isinstance(admission, StandardUnetNativeLoraAdmission):
            raise TypeError("Standard UNet variant template requires admission.")
        if admission.resolution is None or not admission.adaptation.plan.adapters:
            raise ValueError("Standard UNet variant template requires adapters.")
        key = self._key(model, admission)
        with self._lock:
            cached = self._templates.get(key)
            if cached is not None:
                self._templates.move_to_end(key)
                return cached
            static_model = PATCHER_LIFECYCLE.derive_model(
                model,
                (),
                operation="standard UNet static variant template",
                disable_dynamic=True,
            )
            if not isinstance(static_model, ModelPatcher) or static_model.is_dynamic():
                raise TypeError("Comfy did not return a static standard-UNet MODEL.")
            source_diffusion = static_model.get_model_object("diffusion_model")
            if not isinstance(source_diffusion, nn.Module):
                raise TypeError(
                    "Static standard-UNet diffusion model must be a module."
                )
            topology = STANDARD_UNET_VARIANT_TOPOLOGY_BUILDER.build(
                admission.resolution
            )
            root = STANDARD_UNET_VARIANT_ROOT_BUILDER.build(source_diffusion)
            template = StandardUnetVariantTemplate(
                static_model,
                source_diffusion,
                root,
                topology,
                len(admission.adaptation.plan.adapters),
            )
            template.prepare()
            self._templates[key] = template
            while len(self._templates) > self._maximum_entries:
                self._templates.popitem(last=False)
            return template

    def clear(self) -> None:
        """Release cache ownership without mutating possibly loaded graphs."""

        with self._lock:
            self._templates.clear()

    @property
    def entry_count(self) -> int:
        """Return the bounded number of retained templates."""

        with self._lock:
            return len(self._templates)

    @staticmethod
    def _key(
        model: ModelPatcher,
        admission: StandardUnetNativeLoraAdmission,
    ) -> StandardUnetVariantTemplateKey:
        """Build one architecture-neutral semantic cache key."""

        resolution = admission.resolution
        if resolution is None:
            raise ValueError("Standard UNet template key requires resolution.")
        targets = tuple(
            (
                result.adapter.composition_index,
                tuple(
                    (
                        target.path,
                        target.operation_type,
                        target.source_keys,
                        target.ordinary_additive_lora,
                    )
                    for target in result.model_targets
                ),
            )
            for result in resolution.adapters
        )
        return StandardUnetVariantTemplateKey(
            ModelCloneLineage.from_model(model),
            admission.adaptation.plan.adapters,
            targets,
        )


STANDARD_UNET_VARIANT_TEMPLATE_CACHE = StandardUnetVariantTemplateCache()
