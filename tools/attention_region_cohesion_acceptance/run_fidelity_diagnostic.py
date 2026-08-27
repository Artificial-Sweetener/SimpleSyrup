# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the fixed SDXL hair fine-structure diagnostic."""

from __future__ import annotations

import json

from .client import execute
from .fidelity_diagnostic import (
    COMPONENT_STAGES,
    STRENGTH_STAGES,
    build_component_diagnostic_workflow,
    build_fidelity_diagnostic_workflow,
    build_strength_diagnostic_workflow,
    compose_fidelity_diagnostic_sheet,
    compose_mask_diagnostic_sheet,
)
from .run import OUTPUT_ROOT, PROOF_ROOT, SERVER


def main() -> None:
    """Execute and persist the progressive hair-processing comparison."""

    source = (
        OUTPUT_ROOT
        / "simple_syrup_attention_acceptance_v2/sdxl/final_many_image_00002_.png"
    )
    prefix = "simple_syrup_attention_cohesion_proof/sdxl_hair_fidelity"
    outputs = execute(
        SERVER,
        build_fidelity_diagnostic_workflow(
            source,
            output_prefix=prefix,
            concept="hair",
        ),
        OUTPUT_ROOT,
    )
    stages = (
        ("RAW CONCEPT EVIDENCE", outputs["101"]),
        ("STRENGTH 0.15", outputs["1101"]),
        ("+ CONSENSUS 0.25", outputs["1131"]),
        ("+ FEATHER 8 PX", outputs["1141"]),
    )
    sheet = PROOF_ROOT / "sdxl_hair_fine_structure_diagnostic.png"
    compose_fidelity_diagnostic_sheet(
        image_path=outputs["900"],
        stage_paths=stages,
        destination=sheet,
    )
    manifest = {
        "sheet": str(sheet),
        "outputs": {node_id: str(path) for node_id, path in outputs.items()},
    }
    destination = PROOF_ROOT / "sdxl_hair_fine_structure_diagnostic.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination)

    strength_prefix = "simple_syrup_attention_cohesion_proof/sdxl_hair_strength"
    strength_outputs = execute(
        SERVER,
        build_strength_diagnostic_workflow(
            source,
            output_prefix=strength_prefix,
            concept="hair",
        ),
        OUTPUT_ROOT,
    )
    strength_sheet = PROOF_ROOT / "sdxl_hair_strength_diagnostic.png"
    compose_mask_diagnostic_sheet(
        title="SDXL Hair Concept Strength Sweep",
        subtitle=(
            "Same generation and shared fast capture; consensus 0.25; no morphology"
        ),
        image_path=strength_outputs["900"],
        stage_paths=tuple(
            (label, strength_outputs[save_id])
            for _request_id, save_id, label, _strength in STRENGTH_STAGES
        ),
        destination=strength_sheet,
    )
    strength_manifest = {
        "sheet": str(strength_sheet),
        "outputs": {node_id: str(path) for node_id, path in strength_outputs.items()},
    }
    strength_destination = PROOF_ROOT / "sdxl_hair_strength_diagnostic.json"
    strength_destination.write_text(
        json.dumps(strength_manifest, indent=2),
        encoding="utf-8",
    )
    print(strength_destination)

    component_prefix = "simple_syrup_attention_cohesion_proof/sdxl_hair_components"
    component_outputs = execute(
        SERVER,
        build_component_diagnostic_workflow(
            source,
            output_prefix=component_prefix,
            concept="hair",
        ),
        OUTPUT_ROOT,
    )
    component_sheet = PROOF_ROOT / "sdxl_hair_component_diagnostic.png"
    compose_mask_diagnostic_sheet(
        title="SDXL Hair Component Cleanup",
        subtitle=(
            "Strength 0.12; consensus 0.25; no splitting, solidity, or feathering"
        ),
        image_path=component_outputs["900"],
        stage_paths=tuple(
            (label, component_outputs[save_id])
            for _request_id, save_id, label, _minimum_size, _keep_only in (
                COMPONENT_STAGES
            )
        ),
        destination=component_sheet,
    )
    component_manifest = {
        "sheet": str(component_sheet),
        "outputs": {node_id: str(path) for node_id, path in component_outputs.items()},
    }
    component_destination = PROOF_ROOT / "sdxl_hair_component_diagnostic.json"
    component_destination.write_text(
        json.dumps(component_manifest, indent=2),
        encoding="utf-8",
    )
    print(component_destination)


if __name__ == "__main__":
    main()
