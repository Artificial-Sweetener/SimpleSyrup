"""Observe one upstream sampler through Comfy's per-model sampler wrapper."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import torch

from ..domain.noise_reuse import PROVENANCE_KEY, SamplingProvenance
from ..shared.logging import get_logger
from .model_patcher_mutations import ModelKeyedWrapperMutation
from .native_noise_scaling import recover_residual
from .patcher_lifecycle import PATCHER_LIFECYCLE

LOGGER = get_logger(__name__)
SUPPORTED_CAPTURE_SAMPLERS = frozenset({"euler", "euler_ancestral"})


def insert_capture_coordinates(
    sigmas: torch.Tensor, coordinates: tuple[float, ...]
) -> torch.Tensor:
    """Insert exact float32 boundaries without replacing native scheduler points."""
    if sigmas.ndim != 1 or len(sigmas) < 2 or not bool(torch.isfinite(sigmas).all()):
        raise ValueError(
            "Source capture requires a finite descending sampling schedule."
        )
    if not bool((sigmas[:-1] > sigmas[1:]).all()):
        raise ValueError(
            "Source capture requires strictly descending sigma boundaries."
        )
    if float(sigmas[-1]) != 0:
        raise ValueError(
            "Source capture requires a schedule ending at zero noise; "
            "disable leftover noise on the originating sampler."
        )
    result = sigmas
    for value in sorted(set(coordinates), reverse=True):
        coordinate = sigmas.new_tensor(value)
        if not 0 < float(coordinate) < float(sigmas[0]):
            raise ValueError(
                f"Requested source sigma {value:.9g} lies outside the "
                f"source sampling schedule."
            )
        if not bool((result == coordinate).any()):
            index = int((result > coordinate).sum())
            result = torch.cat((result[:index], coordinate.reshape(1), result[index:]))
    return result


@dataclass(slots=True)
class TrajectoryCapture:
    """Keep capture storage local to an injected, cache-keyed MODEL derivation."""

    model: Any
    sampler_name: str
    coordinates: tuple[float, ...]
    states: dict[float, torch.Tensor] = field(default_factory=dict)
    clean: torch.Tensor | None = None
    schedule: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Store the exact float32 coordinates used by native sampler schedules."""
        self.coordinates = tuple(
            sorted(
                {float(torch.tensor(value)) for value in self.coordinates}, reverse=True
            )
        )

    def prepare_model(self) -> Any:
        """Clone only this source MODEL and leave unrelated samplers untouched."""
        if self.coordinates and self.sampler_name not in SUPPORTED_CAPTURE_SAMPLERS:
            raise ValueError(
                "Noise reuse source capture supports Euler and Euler "
                "ancestral; select a supported source sampler."
            )
        if not self.coordinates:
            return self.model
        wrappers = import_module("comfy.patcher_extension").WrappersMP
        return PATCHER_LIFECYCLE.derive_model(
            self.model,
            (
                ModelKeyedWrapperMutation(
                    wrappers.SAMPLER_SAMPLE,
                    "simple_syrup.trajectory_capture",
                    self.observe,
                ),
            ),
            operation="SimpleSyrup trajectory capture",
        )

    def observe(
        self, executor: Any, *arguments: Any, **keyword_arguments: Any
    ) -> torch.Tensor:
        """Wrap the individual native solver instance and record its pre-update x."""
        local = copy(executor.class_obj)
        solver = local.sampler_function
        self.states.clear()
        self.clean = None

        def recorded(
            model: Any, state: torch.Tensor, sigmas: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            """Use native solver randomness and preserve its original callback."""
            schedule = insert_capture_coordinates(sigmas, self.coordinates)
            self.schedule = schedule.detach().float().cpu().clone()
            original_callback = kwargs.get("callback")

            def observe_state(record: dict[str, Any]) -> None:
                """Capture actual noisy state, rejecting churned or ambiguous
                coordinates.
                """
                coordinate = float(schedule[int(record["i"])])
                if coordinate in self.coordinates:
                    if float(record.get("sigma_hat", record["sigma"])) != coordinate:
                        raise ValueError(
                            "Noise reuse cannot capture churned source states."
                        )
                    self.states[coordinate] = record["x"].detach().float().cpu().clone()
                if original_callback is not None:
                    original_callback(record)

            kwargs["callback"] = observe_state
            result: torch.Tensor = solver(model, state, schedule, **kwargs)
            self.clean = result.detach().float().cpu().clone()
            return result

        local.sampler_function = recorded
        isolated = copy(executor)
        isolated.original = local.sample
        result: torch.Tensor = isolated(*arguments, **keyword_arguments)
        return result

    def attach(self, latent: dict[str, Any]) -> dict[str, Any]:
        """Associate captured states with their matching external clean sampler
        output.
        """
        samples = latent.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise TypeError(
                "Source sampler output must contain a latent samples tensor."
            )
        actual = self.model.model.process_latent_in(samples).detach().float().cpu()
        if self.coordinates:
            if (
                self.clean is None
                or self.schedule is None
                or set(self.states) != set(self.coordinates)
            ):
                raise ValueError(
                    "Source trajectory capture is incomplete; run the full "
                    "connected workflow again."
                )
            if not torch.allclose(actual, self.clean, atol=1e-5, rtol=1e-5):
                raise ValueError(
                    "Captured trajectory endpoint does not match the source "
                    "sampler output."
                )
            actual = self.clean
        result = latent.copy()
        result[PROVENANCE_KEY] = SamplingProvenance(
            type(self.model.model.latent_format).__name__,
            actual,
            tuple(
                (coordinate, self.states[coordinate]) for coordinate in self.coordinates
            ),
            self.schedule,
            tuple(
                (
                    coordinate,
                    recover_residual(
                        self.model.get_model_object("model_sampling"),
                        actual,
                        self.states[coordinate],
                        coordinate,
                    ),
                )
                for coordinate in self.coordinates
            ),
        )
        LOGGER.info(
            "Sampling provenance attached",
            extra={
                "operation": "trajectory_capture",
                "sampler": self.sampler_name,
                "coordinates": self.coordinates,
                "source_shape": tuple(actual.shape),
            },
        )
        return result
