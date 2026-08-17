# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the blocking accepted Anima regression oracle."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.anima_regression_oracle.artifact_validation import (  # noqa: E402
    AcceptedArtifactValidator,
    ArtifactObservation,
)
from tools.anima_regression_oracle.execution import (  # noqa: E402
    CommandObservation,
    OracleCommandExecutor,
)
from tools.anima_regression_oracle.manifest import (  # noqa: E402
    DEFAULT_MODEL_VISIBILITY_INVENTORY,
    OracleCommand,
    default_manifest,
)
from tools.anima_regression_oracle.model_visibility import (  # noqa: E402
    anima_oracle_model_visibility,
)
from tools.anima_regression_oracle.results import (  # noqa: E402
    AnimaRegressionResultRecorder,
)
from tools.sdxl_attention_coupling_integration.comfy_model_root import (  # noqa: E402
    resolve_active_comfy_model_root,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\universal-regional-adapter\anima-oracle"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate accepted artifacts and execute the requested blocking gates."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        choices=("accepted", "focused", "full", "complete"),
        default="focused",
        help=(
            "Select accepted artifacts only, focused Anima gates, full repository "
            "gates, or the complete oracle including managed Anima reruns."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model-visibility-inventory",
        type=Path,
        default=DEFAULT_MODEL_VISIBILITY_INVENTORY,
        help="Load machine-local adapter sources for complete managed reruns.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    recorder = AnimaRegressionResultRecorder(args.output_root / _run_id())
    manifest = default_manifest(REPO_ROOT)
    commands = _commands(
        args.level,
        manifest.focused_command,
        manifest.repository_commands,
        manifest.managed_rerun_commands,
    )
    artifacts: tuple[ArtifactObservation, ...] = ()
    observations: tuple[CommandObservation, ...] = ()
    try:
        artifacts = AcceptedArtifactValidator().validate(manifest)
        if args.level == "complete":
            visibility = anima_oracle_model_visibility(
                args.model_visibility_inventory,
                model_root=_active_model_root(REPO_ROOT.parents[1]),
            )
            with visibility:
                observations = OracleCommandExecutor().execute(
                    commands,
                    repo_root=REPO_ROOT,
                    output_root=recorder.root,
                )
            if not visibility.cleaned:
                raise RuntimeError("Anima oracle model visibility cleanup failed.")
        else:
            observations = OracleCommandExecutor().execute(
                commands,
                repo_root=REPO_ROOT,
                output_root=recorder.root,
            )
    except BaseException:
        LOGGER.exception("Anima regression oracle failed before command completion.")
    result = recorder.publish(
        level=args.level,
        artifacts=artifacts,
        commands=observations,
        expected_command_count=len(commands),
    )
    passed = len(observations) == len(commands) and all(
        item.return_code == 0 for item in observations
    )
    if args.level == "accepted":
        passed = passed and bool(artifacts)
    LOGGER.info("Anima regression oracle result: %s", result)
    return 0 if passed else 1


def _commands(
    level: str,
    focused: OracleCommand,
    repository: tuple[OracleCommand, ...],
    managed: tuple[OracleCommand, ...],
) -> tuple[OracleCommand, ...]:
    """Return the exact ordered commands for one declared verification level."""

    if level == "accepted":
        return ()
    if level == "focused":
        return (focused,)
    if level == "full":
        return (focused, *repository)
    if level == "complete":
        return (focused, *repository, *managed)
    raise ValueError(f"Unsupported Anima regression-oracle level: {level!r}")


def _run_id() -> str:
    """Return one collision-resistant UTC result identity."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _active_model_root(comfy_root: Path) -> Path:
    """Resolve model visibility through the same authority as managed Comfy."""

    return resolve_active_comfy_model_root(comfy_root)


if __name__ == "__main__":
    raise SystemExit(main())
