# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define immutable accepted evidence and executable Anima regression gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AcceptedImage:
    """Identify one immutable accepted visual artifact."""

    identity: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class CompletedJsonEvidence:
    """Describe one completed managed result and its structural invariants."""

    identity: str
    path: Path
    detail_paths: tuple[Path, ...] = ()
    observation_count: int | None = None
    required_integer_occurrences: tuple[tuple[str, int, int], ...] = ()
    required_image_hashes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PerformanceEvidence:
    """Describe the accepted fixed-workload Anima performance result."""

    path: Path
    adapter_target_count: int
    profile_limits: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class OracleCommand:
    """Name one ordered executable oracle gate."""

    identity: str
    arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnimaRegressionManifest:
    """Hold the complete immutable U1 oracle definition."""

    images: tuple[AcceptedImage, ...]
    managed_results: tuple[CompletedJsonEvidence, ...]
    performance: PerformanceEvidence
    focused_command: OracleCommand
    repository_commands: tuple[OracleCommand, ...]
    managed_rerun_commands: tuple[OracleCommand, ...]


def default_manifest(repo_root: Path) -> AnimaRegressionManifest:
    """Build the accepted Anima oracle for the authoritative Comfy workspace."""

    python = str(
        (repo_root / ".." / ".." / "venv" / "Scripts" / "python.exe").resolve()
    )
    ruff = str((repo_root / ".." / ".." / "venv" / "Scripts" / "ruff.exe").resolve())
    mypy = str((repo_root / ".." / ".." / "venv" / "Scripts" / "mypy.exe").resolve())
    evidence_root = Path(r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1")
    character_a_root = (
        evidence_root
        / "user-prompt-character_a-ownership-proof"
        / "20260812T192527Z-28deb131"
    )
    adapter_a_root = (
        evidence_root / "user-prompt-adapter_a-ownership-proof" / "20260812T192754Z-89ebd7cb"
    )
    style_character_root = (
        evidence_root / "global-style-character-proof" / "20260812T224908Z-892f5d8f"
    )
    focused_tests = (
        "tests/test_anima_module_surface.py",
        "tests/test_anima_lora_weight_categories.py",
        "tests/test_anima_lora_linear.py",
        "tests/test_anima_multi_lora_fidelity.py",
        "tests/test_anima_multi_lora_composition.py",
        "tests/test_anima_full_tile_lora_equivalence.py",
        "tests/test_anima_global_lora_overlap.py",
        "tests/test_anima_lora_block.py",
        "tests/test_anima_lora_block_inactive_schedule.py",
        "tests/test_anima_projection_batch.py",
        "tests/test_anima_lora_combined_multiplier_cache.py",
        "tests/test_anima_regional_lora_device_cache_lifecycle.py",
        "tests/test_anima_single_adapter_mutations.py",
        "tests/test_anima_regional_lora_target_ownership.py",
        "tests/test_anima_composition_phase.py",
        "tests/test_anima_cross_attention.py",
        "tests/test_anima_self_attention_ownership.py",
        "tests/test_anima_self_attention_coherence.py",
        "tests/test_anima_regional_self_attention.py",
        "tests/test_anima_branch_batch.py",
        "tests/test_anima_query_masks.py",
        "tests/test_anima_query_activity.py",
        "tests/test_anima_regional_diagnostics.py",
        "tests/test_anima_regional_permutation_diagnostics.py",
        "tests/test_anima_attention_device_cache_lifecycle.py",
        "tests/test_anima_attention_coupling_integration.py",
        "tests/test_anima_tiled_attention_coupling_integration.py",
        "tests/test_anima_contextual_attention_coupling_integration.py",
        "tests/test_anima_regional_lora_admission_results.py",
        "tests/test_anima_regional_lora_performance_manifest.py",
        "tests/test_anima_regional_lora_performance_results.py",
        "tests/test_anima_regional_lora_scaling_results.py",
    )
    return AnimaRegressionManifest(
        images=(
            AcceptedImage(
                "character_a-phased-composition",
                character_a_root / "baseline.png",
                "66b4851bc48f9fcafa9965209f34c1bb8add3ff675e73c1ff32e85ed89aea7a0",
            ),
            AcceptedImage(
                "adapter_a-one-side-phased-composition",
                adapter_a_root / "baseline.png",
                "3783e8ec83c3088ddb00e4d4d60af44499a3fb73c0cbb3f39e10c38d68a03499",
            ),
            AcceptedImage(
                "global-style-character-control",
                style_character_root / "character_a-right-only.png",
                "a5521ded7595322c1ccc39732fb4f30e9b6e29d5a454811ffb1e26d7916c9c17",
            ),
            AcceptedImage(
                "global-adapter_a-050-regional-character_a",
                style_character_root / "global-adapter_a-050-character_a-right.png",
                "631d206299f3754891f968f01052678f25e22a1b7a3d6188f3081d237a2999a5",
            ),
            AcceptedImage(
                "global-adapter_a-100-regional-character_a",
                style_character_root / "global-adapter_a-100-character_a-right.png",
                "beb9deb0c37547bcd0691d783be9ec37fa41e1360c0abf7203a814e550915d8f",
            ),
        ),
        managed_results=(
            CompletedJsonEvidence(
                "character_a-phased-composition",
                character_a_root / "run.json",
                detail_paths=(character_a_root / "history.json",),
                required_integer_occurrences=(
                    ("model_call_count", 30, 1),
                    ("target_count", 280, 1),
                ),
                required_image_hashes=(
                    "66b4851bc48f9fcafa9965209f34c1bb8add3ff675e73c1ff32e85ed89aea7a0",
                ),
            ),
            CompletedJsonEvidence(
                "adapter_a-one-side-phased-composition",
                adapter_a_root / "run.json",
                detail_paths=(adapter_a_root / "history.json",),
                required_integer_occurrences=(
                    ("model_call_count", 30, 1),
                    ("target_count", 448, 1),
                ),
                required_image_hashes=(
                    "3783e8ec83c3088ddb00e4d4d60af44499a3fb73c0cbb3f39e10c38d68a03499",
                ),
            ),
            CompletedJsonEvidence(
                "global-style-regional-character",
                style_character_root / "global-style-character-proof.json",
                detail_paths=(
                    style_character_root / "character_a-right-only.history.json",
                    style_character_root / "global-adapter_a-050-character_a-right.history.json",
                    style_character_root / "global-adapter_a-100-character_a-right.history.json",
                ),
                observation_count=3,
                required_integer_occurrences=(
                    ("model_call_count", 30, 3),
                    ("target_count", 280, 3),
                ),
                required_image_hashes=(
                    "a5521ded7595322c1ccc39732fb4f30e9b6e29d5a454811ffb1e26d7916c9c17",
                    "631d206299f3754891f968f01052678f25e22a1b7a3d6188f3081d237a2999a5",
                    "beb9deb0c37547bcd0691d783be9ec37fa41e1360c0abf7203a814e550915d8f",
                ),
            ),
            CompletedJsonEvidence(
                "full-managed-matrix",
                evidence_root
                / "p5.9"
                / "20260810T205325Z-ab058756"
                / "p5.9-result.json",
                observation_count=8,
            ),
            CompletedJsonEvidence(
                "tiled-managed-matrix",
                evidence_root
                / "p6.9"
                / "20260810T224653Z-71544322"
                / "p6.9-result.json",
                observation_count=8,
            ),
            CompletedJsonEvidence(
                "contextual-managed-matrix",
                evidence_root
                / "p7.8"
                / "20260811T011406Z-cefad93b"
                / "p7.8-result.json",
                observation_count=6,
            ),
            CompletedJsonEvidence(
                "contextual-sub-token-diagnostics",
                evidence_root
                / "p7.8"
                / "20260811T014916Z-b49b6be3"
                / "p7.8-result.json",
                observation_count=1,
            ),
        ),
        performance=PerformanceEvidence(
            evidence_root / "p5.7" / "20260810T200643Z" / "result.json",
            448,
            (
                ("attention-only", 0.0),
                ("regional-lora-1", 15.0),
                ("regional-lora-4", 35.0),
            ),
        ),
        focused_command=OracleCommand(
            "focused-anima-characterization",
            (python, "-m", "pytest", "-n", "auto", "-q", *focused_tests),
        ),
        repository_commands=(
            OracleCommand("ruff-format-check", (ruff, "format", "--check", ".")),
            OracleCommand("ruff-check", (ruff, "check", ".")),
            OracleCommand("strict-mypy", (mypy, "--strict", "simple_syrup", "tests")),
            OracleCommand(
                "full-python-suite", (python, "-m", "pytest", "-n", "auto", "-q")
            ),
        ),
        managed_rerun_commands=(
            OracleCommand(
                "character_a-managed-proof",
                (python, "-m", "tools.run_character_a_ownership_proof"),
            ),
            OracleCommand(
                "adapter_a-managed-proof",
                (python, "-m", "tools.run_adapter_a_ownership_proof"),
            ),
            OracleCommand(
                "global-style-character-proof",
                (python, "-m", "tools.run_global_style_character_proof"),
            ),
            OracleCommand(
                "full-managed-matrix",
                (python, "-m", "tools.run_anima_attention_coupling_integration"),
            ),
            OracleCommand(
                "tiled-managed-matrix",
                (
                    python,
                    "-m",
                    "tools.run_anima_tiled_attention_coupling_integration",
                ),
            ),
            OracleCommand(
                "contextual-managed-matrix",
                (
                    python,
                    "-m",
                    "tools.run_anima_contextual_attention_coupling_integration",
                ),
            ),
            OracleCommand(
                "performance-gate",
                (python, "-m", "tools.benchmark_anima_regional_lora"),
            ),
        ),
    )
