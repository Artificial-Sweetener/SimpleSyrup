# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run exact user-prompt global and regional LoRA proof workflows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.default_paths import (
    default_benchmark_artifact_root,
    default_comfy_root,
)
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer

DEFAULT_OUTPUT_ROOT = default_benchmark_artifact_root("global-prompt-lora-proof")
_SEPARATOR = re.compile(r"(\[SEP(?:\|[^\]]+)?\])")
_SDXL_GLOBAL = r"<lora:Illustrious\Style\IriaStyleIllustriousV1-000005.safetensors:0.5>"
_ANIMA_ARCANE_GLOBAL = r"<lora:Anima\style\ArcaneViolet_mpt_64.safetensors:0.6>"
_ANIMA_ARCANE_REGIONAL = r"<lora:Anima\style\ArcaneViolet_mpt_64.safetensors:0.8>"
_ANIMA_TURBO_GLOBAL = r"<lora:Anima\anima-turbo-lora-v0.2.safetensors:0.7>"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProofCase:
    """Describe one exact template mutation and expected output pair."""

    case_id: str
    template: str
    global_tag: str
    regional_tag: str
    region_mask_feather: int = 64
    turbo: bool = False
    contextual: bool = False


def cases() -> tuple[ProofCase, ...]:
    """Return SDXL, Anima duplicate, Turbo, and contextual proof coverage."""

    return (
        ProofCase(
            "sdxl-global-and-regional-same-lora",
            "sdxl",
            _SDXL_GLOBAL,
            _SDXL_GLOBAL,
            region_mask_feather=10,
        ),
        ProofCase(
            "anima-global-and-regional-same-lora",
            "anima",
            _ANIMA_ARCANE_GLOBAL,
            _ANIMA_ARCANE_REGIONAL,
        ),
        ProofCase(
            "anima-turbo-global-arcane-regional-tiled",
            "anima",
            _ANIMA_TURBO_GLOBAL,
            _ANIMA_ARCANE_REGIONAL,
            turbo=True,
        ),
        ProofCase(
            "anima-turbo-global-arcane-regional-contextual",
            "anima",
            _ANIMA_TURBO_GLOBAL,
            _ANIMA_ARCANE_REGIONAL,
            turbo=True,
            contextual=True,
        ),
    )


def render_global_first_prompt(
    prompt: str,
    *,
    global_tag: str,
    regional_tag: str,
) -> str:
    """Add one global tag and the same or distinct tag to the first region."""

    pieces = _SEPARATOR.split(prompt)
    segment_indices = tuple(range(0, len(pieces), 2))
    if len(segment_indices) != 3:
        raise ValueError("Global LoRA proof requires exactly three prompt segments.")
    pieces[segment_indices[0]] = f"{global_tag}\n{pieces[segment_indices[0]].strip()}"
    pieces[segment_indices[1]] = f"{regional_tag}\n{pieces[segment_indices[1]].strip()}"
    return "".join(pieces)


def build_case_graph(
    template: dict[str, JsonObject],
    case: ProofCase,
    *,
    run_id: str,
) -> tuple[dict[str, JsonObject], tuple[str, str]]:
    """Return one isolated API graph plus its source and refinement save IDs."""

    graph = copy.deepcopy(template)
    _wrap_replayed_list_widgets(graph)
    for node_id in tuple(graph):
        if node_id.startswith("__sugarcubes_cube_output__"):
            del graph[node_id]
    prefix = "prompt-region" if case.template == "sdxl" else "anima-prompt-region"
    refine_prefix = (
        "tiled-upscale" if case.template == "sdxl" else "anima-diffusion-upscale"
    )
    for node_id in (
        f"{prefix}:positive_prompt",
        f"{refine_prefix}:positive_prompt",
    ):
        inputs = _inputs(graph, node_id)
        prompt = inputs.get("value")
        if not isinstance(prompt, str):
            raise TypeError(f"Proof prompt node {node_id!r} lacks string value.")
        inputs["value"] = render_global_first_prompt(
            prompt,
            global_tag=case.global_tag,
            regional_tag=case.regional_tag,
        )
    for sampler_id in (f"{prefix}:ksampler", f"{refine_prefix}:ksampler"):
        _inputs(graph, sampler_id)["region_mask_feather"] = case.region_mask_feather
    if case.turbo:
        for sampler_id in (f"{prefix}:ksampler", f"{refine_prefix}:ksampler"):
            sampler = _inputs(graph, sampler_id)
            sampler.update(
                {
                    "steps": 10,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                }
            )
    if case.contextual:
        sampler = graph[f"{refine_prefix}:ksampler"]
        sampler["class_type"] = "SimpleSyrup.KSamplerAttentionCouplingContextual"
        inputs = _inputs(graph, f"{refine_prefix}:ksampler")
        for key in (
            "latent_tile_width",
            "latent_tile_height",
            "latent_tile_overlap",
            "latent_tile_batch_size",
        ):
            inputs.pop(key, None)
        inputs.update(
            {
                "diffusion_mode": "mixture_of_diffusers",
                "latent_context_size": 128,
                "latent_context_overlap": 32,
                "latent_context_batch_size": 4,
                "global_weight": 1.0,
                "global_steps": 1,
                "global_decay": 0.5,
            }
        )
    source_save = "proof:source"
    refinement_save = "proof:refinement"
    output_prefix = f"simple_syrup_global_prompt_lora/{run_id}/{case.case_id}"
    graph[source_save] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": [f"{prefix}:vae_decode", 0],
            "filename_prefix": f"{output_prefix}-source",
        },
    }
    graph[refinement_save] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": [f"{refine_prefix}:vae_decode", 0],
            "filename_prefix": f"{output_prefix}-refinement",
        },
    }
    return graph, (source_save, refinement_save)


def execute(
    *,
    sdxl_template_path: Path,
    anima_template_path: Path,
    output_root: Path,
    comfy_root: Path,
    case_ids: frozenset[str] | None = None,
) -> Path:
    """Run all proof cases in one managed Comfy process and persist evidence."""

    artifacts = IntegrationArtifacts(output_root)
    templates = {
        "sdxl": _load_execution_prompt(sdxl_template_path),
        "anima": _load_execution_prompt(anima_template_path),
    }
    available = cases()
    definitions = tuple(
        case for case in available if case_ids is None or case.case_id in case_ids
    )
    if not definitions:
        raise ValueError("Global LoRA proof selection contains no known cases.")
    if case_ids is not None:
        unknown = case_ids - frozenset(case.case_id for case in available)
        if unknown:
            raise ValueError(f"Unknown global LoRA proof cases: {sorted(unknown)!r}.")
    built = tuple(
        (
            case,
            *build_case_graph(
                templates[case.template],
                case,
                run_id=artifacts.run_id,
            ),
        )
        for case in definitions
    )
    required = frozenset(
        cast(str, node["class_type"])
        for _case, graph, _save_ids in built
        for node in graph.values()
    )
    observations: list[JsonObject] = []
    port = 0
    process = None
    with ManagedComfyServer(
        comfy_root=comfy_root,
        artifacts=artifacts,
        required_node_ids=required,
        readiness_timeout=300.0,
    ) as running:
        for case, graph, save_ids in built:
            prompt_id = running.client.submit(graph)
            history = running.client.wait_for_history(prompt_id, timeout=1800.0)
            status = history.get("status")
            if not isinstance(status, dict) or status.get("status_str") != "success":
                raise ValueError(f"Proof case {case.case_id!r} did not succeed.")
            files: list[JsonObject] = []
            for label, save_id in zip(
                ("source", "refinement"),
                save_ids,
                strict=True,
            ):
                reference = extract_saved_image(history, save_id)
                image_bytes = running.client.download_image(reference)
                path = artifacts.root / f"{case.case_id}--{label}.png"
                path.write_bytes(image_bytes)
                with Image.open(path) as image:
                    width, height = image.size
                files.append(
                    {
                        "label": label,
                        "file": path.name,
                        "width": width,
                        "height": height,
                        "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    }
                )
            workflow_path = artifacts.root / f"{case.case_id}.workflow.json"
            history_path = artifacts.root / f"{case.case_id}.history.json"
            _write_json(workflow_path, graph)
            _write_json(history_path, history)
            observations.append(
                {
                    "case_id": case.case_id,
                    "execution_status": "success",
                    "visual_review": "pending",
                    "prompt_id": prompt_id,
                    "global_tag": case.global_tag,
                    "regional_tag": case.regional_tag,
                    "region_mask_feather": case.region_mask_feather,
                    "turbo_settings": (
                        {
                            "steps": 10,
                            "cfg": 1.0,
                            "sampler": "euler",
                            "scheduler": "simple",
                        }
                        if case.turbo
                        else None
                    ),
                    "refinement_mode": "contextual" if case.contextual else "tiled",
                    "files": files,
                    "workflow_file": workflow_path.name,
                    "history_file": history_path.name,
                }
            )
        system_stats = running.system_stats
        port = running.port
        process = running.process
    if process is None:
        raise RuntimeError("Managed proof process did not reach ready state.")
    port_available = is_loopback_port_available(port)
    artifacts.record_cleanup(
        process_running=process.is_running,
        port_available=port_available,
    )
    if process.is_running or not port_available:
        raise RuntimeError("Managed proof process or port cleanup failed.")
    result = artifacts.root / "global-prompt-lora-proof.json"
    _write_json(
        result,
        {
            "schema_version": 2,
            "execution_status": "completed",
            "visual_review": "pending",
            "observations": observations,
            "system_stats": system_stats,
            "cleanup": {"process_stopped": True, "port_available": True},
        },
    )
    return result


def _load_execution_prompt(path: Path) -> dict[str, JsonObject]:
    """Load one SugarCubes queue response's expanded execution prompt."""

    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError("Proof template response must be an object.")
    prompt = decoded.get("execution_prompt")
    if not isinstance(prompt, dict):
        raise TypeError("Proof template response lacks execution_prompt.")
    return cast(dict[str, JsonObject], prompt)


def _inputs(graph: dict[str, JsonObject], node_id: str) -> JsonObject:
    """Return one mutable node input mapping from a proof graph."""

    node = graph.get(node_id)
    if not isinstance(node, dict):
        raise KeyError(f"Proof graph lacks node {node_id!r}.")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError(f"Proof node {node_id!r} inputs must be an object.")
    return inputs


def _wrap_replayed_list_widgets(graph: dict[str, JsonObject]) -> None:
    """Restore Comfy's literal wrapper around executed multiselect list values."""

    for node in graph.values():
        if node.get("class_type") != "SimpleSyrup.LoadMaskBatch":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise TypeError("Replayed mask loader inputs must be an object.")
        image = inputs.get("image")
        if isinstance(image, list):
            inputs["image"] = {"__value__": image}


def _write_json(path: Path, value: object) -> None:
    """Write deterministic UTF-8 proof evidence."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse exact templates, run the proof, and report its manifest path."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdxl-template", type=Path, required=True)
    parser.add_argument("--anima-template", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--comfy-root", type=Path, default=default_comfy_root())
    parser.add_argument("--case-id", action="append", default=None)
    args = parser.parse_args(argv)
    try:
        result = execute(
            sdxl_template_path=args.sdxl_template,
            anima_template_path=args.anima_template,
            output_root=args.output_root,
            comfy_root=args.comfy_root,
            case_ids=(frozenset(args.case_id) if args.case_id else None),
        )
    except BaseException:
        LOGGER.exception("Global prompt LoRA proof failed.")
        return 1
    LOGGER.info("Global prompt LoRA proof completed: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
