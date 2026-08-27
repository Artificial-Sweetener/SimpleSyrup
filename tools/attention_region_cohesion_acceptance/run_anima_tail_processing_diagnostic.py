# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the exact-seed Anima tail processing-boundary diagnostic."""

from __future__ import annotations

import json

from .anima_tail_diagnostic import (
    TAIL_SEED,
    build_tail_processing_workflow,
    tail_processing_labels,
)
from .client import execute
from .fidelity_diagnostic import compose_mask_diagnostic_sheet
from .run import OUTPUT_ROOT, PROOF_ROOT, SERVER


def main() -> None:
    """Generate a labeled sheet showing where valid tail evidence is lost."""

    source = (
        OUTPUT_ROOT
        / "simple_syrup_attention_acceptance_v2/anima/final_many_image_00002_.png"
    )
    prefix = "simple_syrup_attention_cohesion_proof/anima_tail_processing"
    outputs = execute(
        SERVER,
        build_tail_processing_workflow(source, output_prefix=prefix),
        OUTPUT_ROOT,
    )
    sheet = PROOF_ROOT / "anima_tail_processing_boundaries.png"
    compose_mask_diagnostic_sheet(
        title="Hassaku Anima - Tail Processing Boundaries",
        subtitle=(
            f"Seed {TAIL_SEED}; shared fast capture; identical concept evidence "
            "with one shaping boundary changed per panel"
        ),
        image_path=outputs["900"],
        stage_paths=tuple(
            (label, outputs[str(3000 + index)])
            for index, label in enumerate(tail_processing_labels(), start=1)
        ),
        destination=sheet,
    )
    destination = PROOF_ROOT / "anima_tail_processing_boundaries.json"
    destination.write_text(
        json.dumps(
            {
                "sheet": str(sheet),
                "seed": TAIL_SEED,
                "outputs": {node_id: str(path) for node_id, path in outputs.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
