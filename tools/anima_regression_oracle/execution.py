# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute ordered Anima oracle commands without owning their policy."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .manifest import OracleCommand


@dataclass(frozen=True, slots=True)
class CommandObservation:
    """Capture one completed oracle command."""

    identity: str
    arguments: tuple[str, ...]
    return_code: int
    duration_seconds: float
    stdout_file: str
    stderr_file: str


class OracleCommandExecutor:
    """Run exact argument-list commands from the repository root."""

    def execute(
        self,
        commands: tuple[OracleCommand, ...],
        *,
        repo_root: Path,
        output_root: Path,
    ) -> tuple[CommandObservation, ...]:
        """Run commands in order and stop after preserving the first failure."""

        observations: list[CommandObservation] = []
        for index, command in enumerate(commands):
            stdout_path = output_root / f"{index:02d}-{command.identity}.stdout.log"
            stderr_path = output_root / f"{index:02d}-{command.identity}.stderr.log"
            started = time.perf_counter()
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                completed = subprocess.run(
                    command.arguments,
                    cwd=repo_root,
                    check=False,
                    shell=False,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                )
            observation = CommandObservation(
                command.identity,
                command.arguments,
                completed.returncode,
                time.perf_counter() - started,
                stdout_path.name,
                stderr_path.name,
            )
            observations.append(observation)
            if completed.returncode != 0:
                break
        return tuple(observations)
