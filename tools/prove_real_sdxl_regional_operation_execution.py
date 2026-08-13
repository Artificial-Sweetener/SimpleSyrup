# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove every admitted real SDXL regional-LoRA target executes once or more."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
for import_root in (REPO_ROOT, COMFY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import comfy.hooks  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.patcher_extension  # noqa: E402
import comfy.utils  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from simple_syrup.domain.conditioning_schedule import (  # noqa: E402
    ConditioningScheduleRange,
)
from simple_syrup.domain.processed_regional_attention import (  # noqa: E402
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank  # noqa: E402
from simple_syrup.runtime.attention_coupling.unet import (  # noqa: E402
    STANDARD_UNET_ATTENTION_BACKEND,
)
from simple_syrup.runtime.attention_coupling.unet_attention_state import (  # noqa: E402
    StandardUnetAttentionState,
)
from simple_syrup.runtime.regional_attention_diagnostics import (  # noqa: E402
    RegionalAttentionDiagnosticsBuilder,
)
from simple_syrup.runtime.regional_lora.host_operation_backing import (  # noqa: E402
    RegionalHostBackedOperation,
)
from simple_syrup.runtime.regional_lora.standard_unet_operation_preparation import (  # noqa: E402
    STANDARD_UNET_OPERATION_PREPARATION,
)
from simple_syrup.runtime.regional_lora_host_payload import (  # noqa: E402
    RegionalLoraHostPayload,
)
from simple_syrup.runtime.regional_lora_plan_adapter import (  # noqa: E402
    RegionalLoraPlanAdaptation,
)
from tools.regional_lora_real_fixtures import (  # noqa: E402
    REGIONAL_LORA_REAL_FIXTURES,
)

LATENT_SIZE = 16
CONTEXT_TOKENS = 77
CONTEXT_FEATURES = 2048
SDXL_VECTOR_FEATURES = 2816


def main(argv: Sequence[str] | None = None) -> int:
    """Run one tiny real SDXL denoiser call and persist complete coverage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    evidence = _execute_proof()
    serialized = json.dumps(evidence, indent=2, sort_keys=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(f"{serialized}\n", encoding="utf-8")
    print(serialized)
    return 0


def _execute_proof() -> dict[str, object]:
    """Derive, load, execute, unload, and verify one complete target surface."""

    fixture = REGIONAL_LORA_REAL_FIXTURES["sdxl"]
    source = fixture.load_model()
    raw_weights = comfy.utils.load_torch_file(str(fixture.lora_path), safe_load=True)
    hook = comfy.hooks.create_hook_lora(raw_weights, 1.0, 0.0).get_type(
        comfy.hooks.EnumHookType.Weight
    )[0]
    adaptation = RegionalLoraPlanAdaptation(
        fixture.plan(),
        (RegionalLoraHostPayload.from_hook(hook),),
    )
    admission = STANDARD_UNET_OPERATION_PREPARATION.admit(source, adaptation)
    if admission.binding is None or admission.cache is None:
        raise RuntimeError("Real SDXL execution requires complete admission evidence.")
    state = _attention_state(adaptation)
    derived_build = STANDARD_UNET_ATTENTION_BACKEND.derive(
        model=source,
        state=state,
        admission=admission,
    )
    derived_value = derived_build.model
    if not isinstance(derived_value, comfy.model_patcher.ModelPatcher):
        raise TypeError("Real SDXL derivation must return a ModelPatcher.")
    derived = derived_value
    if derived.parent is not source or source.object_patches:
        raise RuntimeError("Real SDXL derivation changed its direct source lineage.")
    original_modules = {
        path: source.get_model_object(path) for path in admission.module_roles
    }
    counts = dict.fromkeys(admission.module_roles, 0)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    output: torch.Tensor | None = None
    cleanup_verified = False
    device = torch.device(derived.load_device)
    try:
        comfy.model_management.load_models_gpu([derived], force_full_load=True)
        for path in admission.module_roles:
            module = derived.get_model_object(path)
            if not isinstance(module, RegionalHostBackedOperation):
                raise RuntimeError(f"Regional operation was not installed: {path!r}.")
            handles.append(module.register_forward_hook(_counter_hook(path, counts)))
        dtype = derived.model.get_dtype_inference()
        latent = torch.zeros(
            (1, 4, LATENT_SIZE, LATENT_SIZE),
            device=device,
            dtype=dtype,
        )
        context = torch.zeros(
            (1, CONTEXT_TOKENS, CONTEXT_FEATURES),
            device=device,
            dtype=dtype,
        )
        sigma = torch.ones((1,), device=device, dtype=torch.float32)
        options = comfy.patcher_extension.copy_nested_dicts(derived.model_options)
        transformer_options = options["transformer_options"]
        comfy.patcher_extension.merge_nested_dicts(
            transformer_options.setdefault("wrappers", {}),
            derived.wrappers,
            copy_dict1=False,
        )
        transformer_options.update(
            {
                "cond_or_uncond": [0],
                "sample_sigmas": torch.tensor(
                    [1.0, 0.0],
                    device=device,
                ),
                "sigmas": sigma,
            }
        )
        with torch.inference_mode():
            value = derived.model.apply_model(
                latent,
                sigma,
                c_crossattn=context,
                y=torch.zeros(
                    (1, SDXL_VECTOR_FEATURES),
                    device=device,
                    dtype=dtype,
                ),
                transformer_options=transformer_options,
            )
        if not isinstance(value, torch.Tensor) or value.shape != latent.shape:
            raise RuntimeError("Real SDXL denoiser returned an invalid output.")
        output = value.detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
        comfy.model_management.unload_all_models()
        derived.detach(unpatch_all=True)
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        cleanup_verified = all(
            source.get_model_object(path) is module
            for path, module in original_modules.items()
        )
    omitted = tuple(path for path, count in counts.items() if count < 1)
    if omitted:
        raise RuntimeError(f"Real SDXL execution omitted targets: {omitted!r}.")
    if not cleanup_verified or admission.cache.size != 0:
        raise RuntimeError("Real SDXL execution did not restore all runtime state.")
    if output is None:
        raise RuntimeError("Real SDXL execution produced no output evidence.")
    return {
        "family": fixture.family,
        "model_path": str(fixture.model_path),
        "lora_path": str(fixture.lora_path),
        "latent_shape": [1, 4, LATENT_SIZE, LATENT_SIZE],
        "context_shape": [1, CONTEXT_TOKENS, CONTEXT_FEATURES],
        "vector_shape": [1, SDXL_VECTOR_FEATURES],
        "admitted_target_count": len(admission.binding.entries),
        "executed_target_count": len(counts),
        "denoiser_call_count": 1,
        "minimum_target_call_count": min(counts.values()),
        "maximum_target_call_count": max(counts.values()),
        "omitted_targets": [],
        "target_call_counts": dict(sorted(counts.items())),
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all()),
        "source_lineage_preserved": derived.parent is source,
        "cleanup_verified": cleanup_verified,
        "execution_cache_empty_after_cleanup": admission.cache.size == 0,
    }


def _attention_state(
    adaptation: RegionalLoraPlanAdaptation,
) -> StandardUnetAttentionState:
    """Return one fractional-mask SDXL attention plan sharing the LoRA plan."""

    base = _context(0, None, 0.0)
    regional = _context(1, 0, 0.25)
    masks = torch.full((1, LATENT_SIZE, LATENT_SIZE), 0.5)
    bank = RegionalMaskBank(masks, masks.clone(), LATENT_SIZE, LATENT_SIZE)
    plan = ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(base, (regional,)),
        ProcessedRegionalAttentionBranch(base, (regional,)),
        bank,
        adaptation.plan,
    )
    return StandardUnetAttentionState(
        plan,
        (1.0,),
        RegionalAttentionDiagnosticsBuilder(bank, backend="real.sdxl.UNetModel"),
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
) -> ProcessedRegionalAttentionContext:
    """Return one always-active CPU context for runtime device adaptation."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                ConditioningScheduleRange(None, None, None, None),
                torch.full((1, CONTEXT_TOKENS, CONTEXT_FEATURES), value),
                1.0,
            ),
        ),
    )


def _counter_hook(
    path: str,
    counts: dict[str, int],
) -> Any:
    """Return one PyTorch hook that increments an exact module-path counter."""

    def count(
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        """Record one complete replacement-module invocation."""

        del module, inputs, output
        counts[path] += 1

    return count


if __name__ == "__main__":
    raise SystemExit(main())
