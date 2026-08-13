# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Record, seal, and unblind P10.3 visual scores."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.manifest_types import JsonObject
from tools.regional_visual_benchmark.scoring import BlindScoreLedger, VisualScore
from tools.regional_visual_benchmark.unblind import unblind_scores


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one explicit blind-scoring lifecycle operation."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "next":
        return _next(args.packet)
    if args.command == "record":
        return _record(args)
    if args.command == "seal":
        print(BlindScoreLedger(args.packet).seal())
        return 0
    if args.command == "unblind":
        print(
            unblind_scores(
                load_manifest(),
                corpus_path=args.corpus,
                packet_path=args.packet,
                mapping_path=args.mapping,
                sealed_scores_path=args.scores,
            )
        )
        return 0
    raise AssertionError(f"Unhandled P10.3 score command: {args.command}")


def _next(packet_path: Path) -> int:
    """Print only the next strategy-blind packet entry."""

    ledger = BlindScoreLedger(packet_path)
    opaque_id = ledger.next_opaque_id
    if opaque_id is None:
        print(json.dumps({"remaining": 0}, indent=2))
        return 0
    packet_value: object = json.loads(packet_path.read_text(encoding="utf-8"))
    if not isinstance(packet_value, dict):
        raise TypeError("P10.3 blind packet must be an object.")
    packet = cast(JsonObject, packet_value)
    entries = packet.get("entries")
    if not isinstance(entries, list):
        raise TypeError("P10.3 blind packet entries must be an array.")
    entry = next(
        value
        for value in entries
        if isinstance(value, dict) and value.get("opaque_id") == opaque_id
    )
    print(
        json.dumps(
            {"remaining": ledger.remaining_count, "entry": entry},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _record(args: argparse.Namespace) -> int:
    """Validate and persist one complete score from CLI fields."""

    score = VisualScore(
        subject_count=args.subject_count,
        identity_separation=args.identity_separation,
        attribute_isolation=args.attribute_isolation,
        anatomical_integrity=args.anatomical_integrity,
        pose_continuity=args.pose_continuity,
        boundary_integrity=args.boundary_integrity,
        global_composition=args.global_composition,
        style_consistency=args.style_consistency,
        failure_severity=args.failure_severity,
        failure_classes=tuple(args.failure_class),
        notes=args.notes,
    )
    ledger = BlindScoreLedger(args.packet)
    ledger.record(args.opaque_id, score)
    print(json.dumps({"remaining": ledger.remaining_count}, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the explicit score-lifecycle argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    next_command = commands.add_parser("next")
    next_command.add_argument("--packet", type=Path, required=True)
    record = commands.add_parser("record")
    record.add_argument("--packet", type=Path, required=True)
    record.add_argument("--opaque-id", required=True)
    for name in (
        "subject-count",
        "identity-separation",
        "attribute-isolation",
        "anatomical-integrity",
        "pose-continuity",
        "boundary-integrity",
        "global-composition",
        "style-consistency",
        "failure-severity",
    ):
        record.add_argument(f"--{name}", type=_score_value, required=True)
    record.add_argument("--failure-class", action="append", default=[])
    record.add_argument("--notes", default="")
    seal = commands.add_parser("seal")
    seal.add_argument("--packet", type=Path, required=True)
    unblind = commands.add_parser("unblind")
    unblind.add_argument("--corpus", type=Path, required=True)
    unblind.add_argument("--packet", type=Path, required=True)
    unblind.add_argument("--mapping", type=Path, required=True)
    unblind.add_argument("--scores", type=Path, required=True)
    return parser


def _score_value(value: str) -> int:
    """Parse one inclusive 0-to-4 rubric value."""

    result = int(value)
    if not 0 <= result <= 4:
        raise argparse.ArgumentTypeError("score must be between 0 and 4")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
