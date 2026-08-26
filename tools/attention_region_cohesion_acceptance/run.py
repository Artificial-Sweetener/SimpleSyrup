"""Run SDXL and Anima attention-cohesion visual acceptance cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import compose_sheet
from .client import execute
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
            default_path=outputs["1101"],
            solid_path=outputs["1131"],
            destination=sheet,
        )
        case_manifest[key] = {
            "concept": case.concept,
            "outputs": {node_id: str(path) for node_id, path in outputs.items()},
            "proof_sheet": str(sheet),
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
