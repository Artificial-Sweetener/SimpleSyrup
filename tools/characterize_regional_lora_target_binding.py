# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize real installed regional LoRA target binding without mutation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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

from simple_syrup.runtime.regional_lora.comfy_adapter_resolver import (  # noqa: E402
    COMFY_REGIONAL_ADAPTER_RESOLVER,
)
from simple_syrup.runtime.regional_lora.resolved_operation_translator import (  # noqa: E402
    COMFY_RESOLVED_OPERATION_TRANSLATOR,
)
from simple_syrup.runtime.regional_lora.target_binder import (  # noqa: E402
    REGIONAL_LORA_TARGET_BINDER,
)
from simple_syrup.runtime.regional_lora_host_payload import (  # noqa: E402
    RegionalLoraHostPayload,
)
from simple_syrup.runtime.regional_lora_plan_adapter import (  # noqa: E402
    RegionalLoraPlanAdaptation,
)
from tools.regional_lora_real_fixtures import (  # noqa: E402
    REGIONAL_LORA_REAL_FIXTURES,
    RegionalLoraRealFixture,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Bind selected fixtures and atomically publish complete JSON evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "families",
        nargs="*",
        choices=tuple(REGIONAL_LORA_REAL_FIXTURES),
        default=(),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    families = tuple(arguments.families) or tuple(REGIONAL_LORA_REAL_FIXTURES)
    serialized = json.dumps(
        [_characterize(REGIONAL_LORA_REAL_FIXTURES[family]) for family in families],
        indent=2,
        sort_keys=True,
    )
    print(serialized)
    if arguments.output is not None:
        _write_atomically(arguments.output, serialized)
    return 0


def _characterize(fixture: RegionalLoraRealFixture) -> dict[str, object]:
    """Resolve, translate, and bind one fixture while proving immutability."""

    patcher = fixture.load_model()
    raw_weights = comfy.utils.load_torch_file(str(fixture.lora_path), safe_load=True)
    hook = comfy.hooks.create_hook_lora(raw_weights, 1.0, 0.0).get_type(
        comfy.hooks.EnumHookType.Weight
    )[0]
    payload = RegionalLoraHostPayload.from_hook(hook)
    before_model = _model_fingerprint(patcher.model)
    before_patcher = _patcher_fingerprint(patcher)
    before_hook = _hook_fingerprint(hook)
    resolution = COMFY_REGIONAL_ADAPTER_RESOLVER.resolve(
        RegionalLoraPlanAdaptation(fixture.plan(), (payload,)),
        model=patcher,
    )
    operations = COMFY_RESOLVED_OPERATION_TRANSLATOR.translate(resolution)
    binding = REGIONAL_LORA_TARGET_BINDER.bind(
        source=patcher,
        candidate=patcher,
        resolution=resolution,
        operations=operations,
    )
    module_classes = Counter(entry.module_class.value for entry in binding.entries)
    capabilities = Counter(entry.spatial_capability.value for entry in binding.entries)
    observation = {
        "family": fixture.family,
        "model_path": str(fixture.model_path),
        "lora_path": str(fixture.lora_path),
        "normalized_targets": sum(
            len(adapter.model_targets) for adapter in resolution.adapters
        ),
        "translated_operations": len(operations.entries),
        "bound_targets": sum(entry.bound for entry in binding.entries),
        "module_classes": dict(sorted(module_classes.items())),
        "spatial_capabilities": dict(sorted(capabilities.items())),
        "issues": [
            {
                "code": issue.code.value,
                "message": issue.message,
                "parameter_path": issue.parameter_path,
                "target_index": issue.target_index,
            }
            for issue in binding.issues
        ],
        "model_unchanged": before_model == _model_fingerprint(patcher.model),
        "patcher_unchanged": before_patcher == _patcher_fingerprint(patcher),
        "hook_unchanged": before_hook == _hook_fingerprint(hook),
        "identity_preserved": all(
            entry.normalized_target
            is resolution.adapters[
                entry.descriptor.adapter.composition_index
            ].model_targets[entry.descriptor.target_index]
            for entry in binding.entries
        ),
        "sample_parameter_paths": [
            entry.descriptor.target.parameter_path for entry in binding.entries[:3]
        ],
    }
    required_truths = (
        "model_unchanged",
        "patcher_unchanged",
        "hook_unchanged",
        "identity_preserved",
    )
    if binding.issues:
        raise RuntimeError(
            f"{fixture.family} binding reported issues: {binding.issues}"
        )
    if len(binding.entries) != len(operations.entries):
        raise RuntimeError(f"{fixture.family} binding omitted translated operations.")
    if not all(bool(observation[key]) for key in required_truths):
        raise RuntimeError(f"{fixture.family} binding changed or copied source state.")
    return observation


def _model_fingerprint(model: object) -> tuple[tuple[object, ...], ...]:
    """Fingerprint tensor identities, versions, shapes, dtypes, and devices."""

    state_dict = model.state_dict()  # type: ignore[attr-defined]
    return tuple(
        (
            key,
            value.data_ptr(),
            value._version,
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
        )
        for key, value in state_dict.items()
    )


def _patcher_fingerprint(patcher: object) -> tuple[object, ...]:
    """Fingerprint mutation surfaces binding must leave untouched."""

    return (
        id(patcher),
        id(patcher.model),  # type: ignore[attr-defined]
        tuple(patcher.object_patches.items()),  # type: ignore[attr-defined]
        tuple(patcher.object_patches_backup.items()),  # type: ignore[attr-defined]
        tuple(patcher.patches.items()),  # type: ignore[attr-defined]
    )


def _hook_fingerprint(hook: comfy.hooks.WeightHook) -> tuple[object, ...]:
    """Fingerprint the host hook fields binding must leave unchanged."""

    return (
        hook.need_weight_init,
        id(hook.weights),
        id(hook.weights_clip),
        hook._strength_model,
        hook._strength_clip,
        hook.hook_ref,
    )


def _write_atomically(destination: Path, content: str) -> None:
    """Publish complete evidence without leaving partial output."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
