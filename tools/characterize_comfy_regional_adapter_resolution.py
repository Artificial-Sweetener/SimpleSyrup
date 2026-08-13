# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize real installed Comfy regional adapter resolution fixtures."""

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
    """Resolve selected real fixtures and print their immutable observations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "families",
        nargs="*",
        choices=tuple(REGIONAL_LORA_REAL_FIXTURES),
        default=(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Atomically write the complete JSON evidence to this path.",
    )
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
    """Resolve one real fixture and verify source model and hook immutability."""

    fixture.require_files()
    patcher = fixture.load_model()
    raw_weights = comfy.utils.load_torch_file(str(fixture.lora_path), safe_load=True)
    hook = comfy.hooks.create_hook_lora(raw_weights, 1.0, 0.0).get_type(
        comfy.hooks.EnumHookType.Weight
    )[0]
    payload = RegionalLoraHostPayload.from_hook(hook)
    before_model = _model_fingerprint(patcher.model)
    before_hook = _hook_fingerprint(hook)
    adaptation = RegionalLoraPlanAdaptation(fixture.plan(), (payload,))

    resolution = COMFY_REGIONAL_ADAPTER_RESOLVER.resolve(
        adaptation,
        model=patcher,
    )
    result = resolution.adapters[0]
    scopes = Counter(entry.scope.value for entry in result.source_entries)
    observation = {
        "family": fixture.family,
        "model_path": str(fixture.model_path),
        "lora_path": str(fixture.lora_path),
        "model_state_entries": len(before_model),
        "raw_source_entries": len(raw_weights),
        "resolved_model_targets": len(result.model_targets),
        "ordinary_additive_targets": sum(
            target.ordinary_additive_lora for target in result.model_targets
        ),
        "source_scopes": dict(sorted(scopes.items())),
        "issues": [
            {"code": issue.code.value, "message": issue.message}
            for issue in resolution.issues
        ],
        "model_unchanged": before_model == _model_fingerprint(patcher.model),
        "hook_unchanged": before_hook == _hook_fingerprint(hook),
        "payload_identity_preserved": result.payload.raw_weights is raw_weights,
        "sample_target_paths": [
            target.path.parameter_key for target in result.model_targets[:3]
        ],
    }
    if resolution.issues:
        raise RuntimeError(
            f"{fixture.family} resolution reported issues: {resolution.issues}"
        )
    if not all(
        bool(observation[key])
        for key in ("model_unchanged", "hook_unchanged", "payload_identity_preserved")
    ):
        raise RuntimeError(f"{fixture.family} resolution mutated source state.")
    return observation


def _model_fingerprint(model: object) -> tuple[tuple[object, ...], ...]:
    """Fingerprint model tensor identities and versions without moving tensors."""

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


def _hook_fingerprint(hook: comfy.hooks.WeightHook) -> tuple[object, ...]:
    """Fingerprint the host hook fields resolution must leave unchanged."""

    return (
        hook.need_weight_init,
        id(hook.weights),
        id(hook.weights_clip),
        hook._strength_model,
        hook._strength_clip,
        hook.hook_ref,
    )


def _write_atomically(destination: Path, content: str) -> None:
    """Publish complete characterization evidence without partial results."""

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
