# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove U11 nodes and managed model choices through CPU-only live Comfy startup."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.run_sdxl_regional_lora_visual_matrix import _model_links
from tools.sdxl_attention_coupling_integration.visual_cases import (
    character_b_NAME,
    CHECKPOINT_NAME,
    ELDEN_STYLE_NAME,
    character_c_NAME,
    visual_cases,
)
from tools.sdxl_attention_coupling_integration.visual_launch import (
    sdxl_visual_registration_launch_arguments,
)
from tools.sdxl_attention_coupling_integration.visual_workflow import (
    build_sdxl_visual_workflow,
)

DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\universal-regional-adapter\u11-registration"
)
_METADATA_NODE_IDS = (
    "CheckpointLoaderSimple",
    "LoraLoaderModelOnly",
    "CreateHookLoraModelOnly",
    "SimpleSyrup.KSamplerAttentionCoupling",
    "SimpleSyrup.KSamplerAttentionCouplingTiled",
    "SimpleSyrup.KSamplerAttentionCouplingContextual",
)


def execute_registration_probe(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    readiness_timeout: float,
) -> Path:
    """Start CPU-only Comfy, validate live choices, and clean exact owners."""

    model_links = _model_links(comfy_root)
    server_cleanup = False
    metadata: dict[str, JsonObject] = {}
    system_stats: JsonObject = {}
    required = _required_node_ids(artifacts.run_id)
    with model_links:
        with ManagedComfyServer(
            comfy_root=comfy_root,
            artifacts=artifacts,
            required_node_ids=required,
            readiness_timeout=readiness_timeout,
            launch_arguments=sdxl_visual_registration_launch_arguments(),
        ) as running:
            system_stats = running.system_stats
            metadata = {
                node_id: running.client.node_metadata(node_id)
                for node_id in _METADATA_NODE_IDS
            }
            _require_choice(metadata["CheckpointLoaderSimple"], CHECKPOINT_NAME)
            for name in (character_b_NAME, character_c_NAME, ELDEN_STYLE_NAME):
                _require_choice(metadata["LoraLoaderModelOnly"], name)
                _require_choice(metadata["CreateHookLoraModelOnly"], name)
            port = running.port
            process = running.process
        server_cleanup = not process.is_running and is_loopback_port_available(port)
        artifacts.record_cleanup(
            process_running=process.is_running,
            port_available=is_loopback_port_available(port),
        )
    if not server_cleanup or not model_links.cleaned:
        raise RuntimeError("U11 registration probe did not clean every exact owner.")
    result: JsonObject = {
        "status": "completed",
        "launch_arguments": list(sdxl_visual_registration_launch_arguments()),
        "required_node_count": len(required),
        "required_node_ids": sorted(required),
        "checkpoint_choice": CHECKPOINT_NAME,
        "lora_choices": [character_b_NAME, character_c_NAME, ELDEN_STYLE_NAME],
        "metadata": metadata,
        "system_stats": system_stats,
        "cleanup": {"server": True, "model_links": True},
    }
    path = artifacts.root / "registration-proof.json"
    _write_json(path, result)
    return path


def _required_node_ids(run_id: str) -> frozenset[str]:
    """Return the union of exact live nodes required by all U11 workflows."""

    return frozenset(
        node_id
        for case in visual_cases()
        for node_id in build_sdxl_visual_workflow(
            run_id=run_id,
            checkpoint_name=CHECKPOINT_NAME,
            mask_names=("left.png", "right.png"),
            case=case,
        ).required_node_ids
    )


def _require_choice(metadata: JsonObject, expected: str) -> None:
    """Require one exact model choice anywhere in live Comfy node metadata."""

    if expected not in _nested_text_values(metadata):
        raise ValueError(f"Live Comfy metadata omits model choice {expected!r}.")


def _nested_text_values(value: object) -> frozenset[str]:
    """Collect all nested text values without assuming Comfy metadata layout."""

    if isinstance(value, str):
        return frozenset((value,))
    if isinstance(value, dict):
        return frozenset(
            text for nested in value.values() for text in _nested_text_values(nested)
        )
    if isinstance(value, list | tuple):
        return frozenset(
            text for nested in value for text in _nested_text_values(nested)
        )
    return frozenset()


def _write_json(path: Path, payload: JsonObject) -> None:
    """Atomically persist one stable registration proof."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command arguments and return one process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=300.0)
    arguments = parser.parse_args(argv)
    artifacts = IntegrationArtifacts(arguments.output_root)
    try:
        result = execute_registration_probe(
            artifacts,
            comfy_root=arguments.comfy_root,
            readiness_timeout=arguments.readiness_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        raise
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
