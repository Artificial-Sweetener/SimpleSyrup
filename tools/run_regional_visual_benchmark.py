# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run or resume the complete corrected P10.3 visual corpus in managed ComfyUI."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.comfy_integration.default_paths import (
    default_benchmark_artifact_root,
    default_comfy_root,
)
from tools.regional_visual_benchmark.execution import (
    DEFAULT_SEGMENT_SIZE,
    VisualCorpusExecutor,
)
from tools.regional_visual_benchmark.run_artifacts import VisualBenchmarkRunArtifacts

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = default_comfy_root()
DEFAULT_OUTPUT_ROOT = default_benchmark_artifact_root(
    "anima-regional-prompting-v1/p10.3"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse managed-run controls and return a process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--readiness-timeout", type=float, default=300.0)
    parser.add_argument("--prompt-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--segment-size", type=_positive_integer, default=DEFAULT_SEGMENT_SIZE
    )
    parser.add_argument("--max-sources", type=_positive_integer)
    parser.add_argument("--max-visuals", type=_positive_integer)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    manifest = load_manifest()
    run = (
        VisualBenchmarkRunArtifacts.resume(args.resume_root, manifest.benchmark_id)
        if args.resume_root is not None
        else VisualBenchmarkRunArtifacts.create(args.output_root, manifest.benchmark_id)
    )
    try:
        result = VisualCorpusExecutor(
            comfy_root=args.comfy_root,
            run_artifacts=run,
            manifest=manifest,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
            segment_size=args.segment_size,
        ).execute(
            max_sources=args.max_sources,
            max_visuals=args.max_visuals,
        )
    except BaseException as error:
        run.record_failure(error)
        LOGGER.exception("P10.3 managed visual corpus failed at %s", run.root)
        return 1
    LOGGER.info("P10.3 managed visual artifact completed at %s", result)
    return 0


def _positive_integer(value: str) -> int:
    """Parse one positive bounded-run or segment count."""

    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
