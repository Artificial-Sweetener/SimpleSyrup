# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run SDXL and Anima attention-cohesion visual acceptance cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import compose_sheet
from .client import execute
from .source_workflow import build_source_workflow
from .workflow import build_workflow

OUTPUT_ROOT = Path("E:/ComfyUI/output")
PROOF_ROOT = OUTPUT_ROOT / "simple_syrup_attention_cohesion_proof"
SERVER = "http://127.0.0.1:8207"


@dataclass(frozen=True, slots=True)
class AcceptanceCase:
    """Describe one real-model cohesion acceptance case."""

    source: Path
    concept: str
    strength: float
    consensus: float
    split: float
    title: str


def main() -> None:
    """Execute both models and record labeled artifact paths."""

    cases = _cases()
    case_manifest: dict[str, object] = {}
    for key, case in cases.items():
        prefix = f"simple_syrup_attention_cohesion_proof/{key}"
        graph = build_workflow(
            case.source,
            output_prefix=prefix,
            concept=case.concept,
            strength=case.strength,
            consensus=case.consensus,
            split=case.split,
        )
        outputs = execute(SERVER, graph, OUTPUT_ROOT)
        sheet = PROOF_ROOT / f"{key}_cohesive_silhouette_proof.png"
        compose_sheet(
            title=case.title,
            concept=case.concept,
            image_path=outputs["900"],
            raw_path=outputs["101"],
            previous_path=outputs["1101"],
            isolated_path=outputs["1131"],
            solid_path=outputs["1141"],
            destination=sheet,
        )
        case_manifest[key] = {
            "concept": case.concept,
            "outputs": {node_id: str(path) for node_id, path in outputs.items()},
            "proof_sheet": str(sheet),
        }
    atlas_key = "anima_outfit"
    atlas_concept = "pink and blue witch outfit"
    atlas_prefix = f"simple_syrup_attention_cohesion_proof/{atlas_key}"
    atlas_graph = build_source_workflow(
        Path("E:/devprojects/attention-atlas/fixtures/source/pink-witch.json"),
        output_prefix=atlas_prefix,
        concept=atlas_concept,
    )
    atlas_outputs = execute(SERVER, atlas_graph, OUTPUT_ROOT)
    atlas_sheet = PROOF_ROOT / f"{atlas_key}_cohesive_silhouette_proof.png"
    compose_sheet(
        title="Hassaku Anima - Compound Outfit Concept Isolation",
        concept=atlas_concept,
        image_path=atlas_outputs["900"],
        raw_path=atlas_outputs["101"],
        previous_path=atlas_outputs["1101"],
        isolated_path=atlas_outputs["1131"],
        solid_path=atlas_outputs["1141"],
        destination=atlas_sheet,
    )
    case_manifest[atlas_key] = {
        "concept": atlas_concept,
        "outputs": {node_id: str(path) for node_id, path in atlas_outputs.items()},
        "proof_sheet": str(atlas_sheet),
    }
    manifest = {"server": SERVER, "cases": case_manifest}
    destination = PROOF_ROOT / "cohesion_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination)


def _cases() -> dict[str, AcceptanceCase]:
    """Return the requested fixed-model, fixed-prompt acceptance matrix."""

    prior_root = OUTPUT_ROOT / "simple_syrup_attention_acceptance_v2"
    return {
        "sdxl": AcceptanceCase(
            prior_root / "sdxl/final_many_image_00002_.png",
            "pink hair",
            0.12,
            0.25,
            0.30,
            "SDXL / Illustrious Amanatsu - Cohesive Attention Silhouette",
        ),
        "anima": AcceptanceCase(
            prior_root / "anima/final_many_image_00002_.png",
            "holding cat",
            0.14,
            0.30,
            0.35,
            "Hassaku Anima - Cohesive Attention Silhouette",
        ),
    }


if __name__ == "__main__":
    main()
