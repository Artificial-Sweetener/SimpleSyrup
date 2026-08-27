# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the exact-seed Anima cat-tail evidence diagnostic."""

from __future__ import annotations

import argparse
import json

from .anima_tail_diagnostic import (
    TAIL_SEED,
    build_tail_evidence_workflow,
    tail_evidence_labels,
)
from .client import execute
from .fidelity_diagnostic import compose_mask_diagnostic_sheet
from .run import OUTPUT_ROOT, PROOF_ROOT, SERVER


def main() -> None:
    """Generate one labeled exact-seed sheet for the requested capture profile."""

    arguments = _arguments()
    profile = arguments.profile
    diagnostic_name = "raw_strength" if arguments.raw_only else "support"
    source = (
        OUTPUT_ROOT
        / "simple_syrup_attention_acceptance_v2/anima/final_many_image_00002_.png"
    )
    prefix = (
        f"simple_syrup_attention_cohesion_proof/anima_tail_{profile}_{diagnostic_name}"
    )
    outputs = execute(
        SERVER,
        build_tail_evidence_workflow(
            source,
            output_prefix=prefix,
            profile=profile,
            raw_only=arguments.raw_only,
        ),
        OUTPUT_ROOT,
    )
    sheet = PROOF_ROOT / f"anima_tail_{profile}_{diagnostic_name}_sweep.png"
    compose_mask_diagnostic_sheet(
        title=(
            "Hassaku Anima - Exact-Seed Cat Tail "
            f"({profile.title()} {diagnostic_name.replace('_', ' ').title()})"
        ),
        subtitle=(
            f"Seed {TAIL_SEED}; raw attention and concept isolation; "
            "no component filtering, solidity, splitting, or feathering"
        ),
        image_path=outputs["900"],
        stage_paths=tuple(
            (label, outputs[str(3000 + index)])
            for index, label in enumerate(
                tail_evidence_labels(raw_only=arguments.raw_only),
                start=1,
            )
        ),
        destination=sheet,
    )
    destination = PROOF_ROOT / f"anima_tail_{profile}_{diagnostic_name}_sweep.json"
    destination.write_text(
        json.dumps(
            {
                "sheet": str(sheet),
                "profile": profile,
                "seed": TAIL_SEED,
                "outputs": {node_id: str(path) for node_id, path in outputs.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(destination)


def _arguments() -> argparse.Namespace:
    """Parse the capture profile requested for this diagnostic run."""

    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("fast", "balanced"))
    parser.add_argument("--raw-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
