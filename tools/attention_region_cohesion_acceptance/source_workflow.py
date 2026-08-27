# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build a first-pass Anima concept-isolation comparison from source JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .workflow import Graph, configure_mask


def build_source_workflow(
    source_path: Path,
    *,
    output_prefix: str,
    concept: str,
) -> Graph:
    """Return one validated first-generation workflow with comparison masks."""

    source = _source(source_path)
    graph: Graph = {
        "1": {
            "class_type": "SimpleSyrup.SimpleLoadAnima",
            "inputs": {
                "diffusion_model": _string(source, "model"),
                "quantization": "Original",
                "diffusion_weight_dtype": "default",
                "text_encoder": "auto",
                "text_encoder_device": "default",
                "vae": "auto",
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": _string(source, "positive"), "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": _string(source, "negative"), "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": _integer(source, "width"),
                "height": _integer(source, "height"),
                "batch_size": 1,
            },
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": int(_string(source, "seed")),
                "steps": _integer(source, "steps"),
                "cfg": _number(source, "cfg"),
                "sampler_name": _string(source, "sampler"),
                "scheduler": _string(source, "scheduler"),
                "denoise": 1,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "900": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": f"{output_prefix}/image",
            },
        },
    }
    for request_id, conversion_id, save_id in (
        ("10", "100", "101"),
        ("11", "1100", "1101"),
        ("14", "1130", "1131"),
        ("15", "1140", "1141"),
    ):
        graph[request_id] = {
            "class_type": "SimpleSyrup.ConceptAttentionSEGS",
            "inputs": {"image": ["6", 0]},
        }
        graph[conversion_id] = {
            "class_type": "MaskToImage",
            "inputs": {"mask": [request_id, 2]},
        }
        graph[save_id] = {
            "class_type": "SaveImage",
            "inputs": {"images": [conversion_id, 0]},
        }
    configure_mask(
        graph,
        request_id="10",
        save_id="101",
        concept=concept,
        strength=0.0,
        consensus=0.0,
        split=0.0,
        minimum_size=1,
        keep_only=0,
        solidity=0.0,
        evidence_mode="raw attention",
        prefix=f"{output_prefix}/raw_alpha",
    )
    configure_mask(
        graph,
        request_id="11",
        save_id="1101",
        concept=concept,
        strength=0.15,
        consensus=0.25,
        split=0.35,
        minimum_size=512,
        keep_only=1,
        solidity=0.75,
        evidence_mode="raw attention",
        prefix=f"{output_prefix}/previous_aggregate",
    )
    configure_mask(
        graph,
        request_id="14",
        save_id="1131",
        concept=concept,
        strength=0.15,
        consensus=0.25,
        split=0.35,
        minimum_size=512,
        keep_only=1,
        solidity=0.0,
        evidence_mode="concept isolation",
        prefix=f"{output_prefix}/concept_isolation",
    )
    configure_mask(
        graph,
        request_id="15",
        save_id="1141",
        concept=concept,
        strength=0.15,
        consensus=0.25,
        split=0.35,
        minimum_size=512,
        keep_only=1,
        solidity=1.0,
        evidence_mode="concept isolation",
        prefix=f"{output_prefix}/concept_isolation_solid",
    )
    return graph


def _source(path: Path) -> dict[str, Any]:
    """Read one source object from an explicit local JSON fixture."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Attention acceptance source must contain an object.")
    return value


def _string(source: dict[str, Any], key: str) -> str:
    """Return one required non-empty source string."""

    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Attention acceptance source '{key}' must be a string.")
    return value


def _integer(source: dict[str, Any], key: str) -> int:
    """Return one required strict source integer."""

    value = source.get(key)
    if type(value) is not int:
        raise ValueError(f"Attention acceptance source '{key}' must be an integer.")
    return value


def _number(source: dict[str, Any], key: str) -> float:
    """Return one required finite source number."""

    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Attention acceptance source '{key}' must be numeric.")
    return float(value)
