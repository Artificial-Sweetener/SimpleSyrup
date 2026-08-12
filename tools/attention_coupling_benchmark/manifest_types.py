"""Define immutable Attention Coupling benchmark manifest records."""

from __future__ import annotations

from dataclasses import dataclass

JsonObject = dict[str, object]


@dataclass(frozen=True)
class ModelArtifact:
    """Identify one model-stack artifact without exposing its local path."""

    artifact_id: str
    role: str
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SamplingSettings:
    """Hold the fixed generation settings shared by every benchmark run."""

    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    denoise: float
    seeds: tuple[int, ...]
    negative_prompt: str


@dataclass(frozen=True)
class MaskRectangle:
    """Describe one normalized rectangular authored region mask."""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class BenchmarkCase:
    """Describe one fixed prompt and mask scenario."""

    case_id: str
    scenario_tags: tuple[str, ...]
    global_prompt: str
    regional_prompts: tuple[str, ...]
    masks: tuple[MaskRectangle, ...]
    regional_prompt_weight: float
    region_mask_feather: int


@dataclass(frozen=True)
class ExecutionProfile:
    """Name one spatial execution path and its fixed controls."""

    execution_id: str
    strategy: str
    spatial_mode: str
    controls: JsonObject


@dataclass(frozen=True)
class BenchmarkRun:
    """Identify one non-optional case, seed, and execution combination."""

    artifact_id: str
    case_id: str
    execution_id: str
    seed: int


@dataclass(frozen=True)
class BenchmarkManifest:
    """Provide the validated source of truth for benchmark expansion."""

    schema_version: int
    benchmark_id: str
    simple_syrup_commit: str
    comfyui_commit: str
    models: tuple[ModelArtifact, ...]
    sampling: SamplingSettings
    executions: tuple[ExecutionProfile, ...]
    cases: tuple[BenchmarkCase, ...]
    artifact_name_template: str

    def runs(self) -> tuple[BenchmarkRun, ...]:
        """Expand every configured case, seed, and execution exactly once."""

        return tuple(
            BenchmarkRun(
                artifact_id=self.artifact_name_template.format(
                    benchmark_id=self.benchmark_id,
                    case_id=case.case_id,
                    execution_id=execution.execution_id,
                    seed=seed,
                ),
                case_id=case.case_id,
                execution_id=execution.execution_id,
                seed=seed,
            )
            for case in self.cases
            for seed in self.sampling.seeds
            for execution in self.executions
        )
