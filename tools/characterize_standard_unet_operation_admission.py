# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize complete real standard-UNet regional-LoRA admission."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
for import_root in (REPO_ROOT, COMFY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import comfy.hooks  # noqa: E402
import comfy.utils  # noqa: E402

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


def main(argv: Sequence[str] | None = None) -> int:
    """Load the real SDXL fixture and publish complete admission evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    evidence = _characterize_sdxl()
    serialized = json.dumps(evidence, indent=2, sort_keys=True)
    print(serialized)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{serialized}\n", encoding="utf-8")
    return 0


def _characterize_sdxl() -> dict[str, object]:
    """Admit every real SDXL LoRA target without cloning or mutating the MODEL."""

    fixture = REGIONAL_LORA_REAL_FIXTURES["sdxl"]
    patcher = fixture.load_model()
    raw_weights = comfy.utils.load_torch_file(str(fixture.lora_path), safe_load=True)
    hook = comfy.hooks.create_hook_lora(raw_weights, 1.0, 0.0).get_type(
        comfy.hooks.EnumHookType.Weight
    )[0]
    adaptation = RegionalLoraPlanAdaptation(
        fixture.plan(),
        (RegionalLoraHostPayload.from_hook(hook),),
    )
    before = _mutation_fingerprint(patcher)
    admission = STANDARD_UNET_OPERATION_PREPARATION.admit(patcher, adaptation)
    if admission.binding is None or admission.cache is None:
        raise RuntimeError("Real SDXL adapter produced incomplete admission evidence.")
    after = _mutation_fingerprint(patcher)
    roles = Counter(role.value for role in admission.module_roles.values())
    entry_roles = Counter(
        entry.spatial_capability.value for entry in admission.binding.entries
    )
    if admission.binding.issues:
        raise RuntimeError(
            f"Real SDXL admission reported issues: {admission.binding.issues!r}."
        )
    if before != after:
        raise RuntimeError("Real SDXL admission mutated the source MODEL.")
    return {
        "family": fixture.family,
        "model_class": (
            f"{type(patcher.model).__module__}.{type(patcher.model).__qualname__}"
        ),
        "diffusion_model_class": (
            f"{type(patcher.model.diffusion_model).__module__}."
            f"{type(patcher.model.diffusion_model).__qualname__}"
        ),
        "model_path": str(fixture.model_path),
        "lora_path": str(fixture.lora_path),
        "bound_targets": len(admission.binding.entries),
        "module_role_count": len(admission.module_roles),
        "module_roles": dict(sorted(roles.items())),
        "bound_entry_roles": dict(sorted(entry_roles.items())),
        "issues": [],
        "source_model_unchanged": True,
        "execution_cache_empty": admission.cache.size == 0,
    }


def _mutation_fingerprint(patcher: object) -> tuple[object, ...]:
    """Fingerprint every patcher mutation surface used by regional execution."""

    return (
        id(patcher),
        id(patcher.model),  # type: ignore[attr-defined]
        tuple(patcher.object_patches.items()),  # type: ignore[attr-defined]
        tuple(patcher.object_patches_backup.items()),  # type: ignore[attr-defined]
        tuple(patcher.patches.items()),  # type: ignore[attr-defined]
        tuple(patcher.callbacks.items()),  # type: ignore[attr-defined]
        tuple(patcher.wrappers.items()),  # type: ignore[attr-defined]
    )


if __name__ == "__main__":
    raise SystemExit(main())
