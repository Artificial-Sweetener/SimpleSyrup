# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observe native source sampling through solver-independent Comfy boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import torch

from ..domain.noise_reuse import (
    PROVENANCE_KEY,
    CapturedSamplingState,
    SamplingProvenance,
)
from ..shared.logging import get_logger
from .model_patcher_mutations import ModelKeyedWrapperMutation
from .native_noise_scaling import recover_residual
from .patcher_lifecycle import PATCHER_LIFECYCLE

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ObservedSamplingState:
    """Retain one real model-space observation without manufacturing a timestep."""

    coordinate: float
    state: torch.Tensor


@dataclass(slots=True)
class TrajectoryCapture:
    """Own bounded source observations without changing its solver or schedule."""

    model: Any
    sampler_name: str
    coordinates: tuple[float, ...]
    graph_context: str = ""
    states: dict[float, ObservedSamplingState] = field(default_factory=dict)
    clean: torch.Tensor | None = None
    schedule: torch.Tensor | None = None
    _active: bool = False

    def __post_init__(self) -> None:
        """Normalize requests to the receiving native float32 coordinate precision."""
        if any(not math.isfinite(value) or value <= 0 for value in self.coordinates):
            raise ValueError("Requested noise levels must be finite and above zero.")
        self.coordinates = tuple(
            sorted(
                {float(torch.tensor(value)) for value in self.coordinates}, reverse=True
            )
        )

    def prepare_model(self) -> Any:
        """Install one observer for every solver using Comfy's sampling contract."""
        try:
            return self._prepare_model()
        except (TypeError, ValueError) as error:
            raise type(error)(self._failure_message(error)) from error

    def _prepare_model(self) -> Any:
        """Derive the capture model through the authoritative patcher lifecycle."""
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
                ModelKeyedWrapperMutation(
                    wrappers.PREDICT_NOISE,
                    "simple_syrup.trajectory_state",
                    self.observe_prediction,
                ),
            ),
            operation="SimpleSyrup trajectory capture",
        )

    def observe(
        self, executor: Any, *arguments: Any, **keyword_arguments: Any
    ) -> torch.Tensor:
        """Scope observations to one unchanged native sampler execution and endpoint."""
        try:
            return self._observe(executor, *arguments, **keyword_arguments)
        except (TypeError, ValueError) as error:
            raise type(error)(self._failure_message(error)) from error

    def _observe(
        self, executor: Any, *arguments: Any, **keyword_arguments: Any
    ) -> torch.Tensor:
        """Retain actual states and release observation scope on every exit."""
        schedule = (
            arguments[1] if len(arguments) > 1 else keyword_arguments.get("sigmas")
        )
        if not isinstance(schedule, torch.Tensor) or schedule.ndim != 1:
            raise ValueError("The original sampler did not provide its noise schedule.")
        if not bool(torch.isfinite(schedule).all()) or bool((schedule < 0).any()):
            raise ValueError("The original sampler provided an invalid noise schedule.")
        self.states.clear()
        self.clean = None
        self.schedule = schedule.detach().float().cpu().clone()
        self._active = True
        try:
            result = executor(*arguments, **keyword_arguments)
            if not isinstance(result, torch.Tensor):
                raise TypeError("The original sampler did not return a latent tensor.")
            self.clean = result.detach().float().cpu().clone()
            return result
        finally:
            self._active = False

    def observe_prediction(
        self, executor: Any, *arguments: Any, **keyword_arguments: Any
    ) -> torch.Tensor:
        """Save real input states at actual solver coordinates, including substeps."""
        if self._active:
            state = arguments[0] if arguments else keyword_arguments.get("x")
            sigma = (
                arguments[1]
                if len(arguments) > 1
                else keyword_arguments.get("timestep")
            )
            if not isinstance(state, torch.Tensor) or not isinstance(
                sigma, torch.Tensor
            ):
                raise TypeError("The original sampler did not expose its noisy latent.")
            values = sigma.detach().float().flatten()
            if values.numel() == 0 or not bool(torch.isfinite(values).all()):
                raise ValueError(
                    "The original sampler reported an invalid noise level."
                )
            if not bool((values == values[0]).all()):
                raise ValueError(
                    "Noise reuse requires one source noise level per batch."
                )
            coordinate = float(values[0])
            if coordinate > 0:
                self._record(state, coordinate)
        result = executor(*arguments, **keyword_arguments)
        if not isinstance(result, torch.Tensor):
            raise TypeError(
                "The original sampler's model returned an invalid prediction."
            )
        return result

    def _record(self, state: torch.Tensor, coordinate: float) -> None:
        """Prefer exact matches, then the closest real level with deterministic ties."""
        selected: torch.Tensor | None = None
        for requested in self.coordinates:
            previous = self.states.get(requested)
            rank = (abs(coordinate - requested), -coordinate)
            if previous is not None and rank >= (
                abs(previous.coordinate - requested),
                -previous.coordinate,
            ):
                continue
            if selected is None:
                selected = state.detach().float().cpu().clone()
                if not bool(torch.isfinite(selected).all()):
                    raise ValueError(
                        "The original sampler produced an invalid noisy latent."
                    )
            self.states[requested] = ObservedSamplingState(coordinate, selected)

    def attach(self, latent: dict[str, Any]) -> dict[str, Any]:
        """Surface capture failures with the original graph identities and algorithm."""
        try:
            return self._attach(latent)
        except (TypeError, ValueError) as error:
            raise type(error)(self._failure_message(error)) from error

    def _failure_message(self, error: Exception) -> str:
        """Identify the graph reach-through and selected algorithm in plain English."""
        LOGGER.error(
            "Source noise handoff failed",
            extra={
                "sampler": self.sampler_name,
                "graph_context": self.graph_context,
                "operation": "trajectory_capture",
                "reason": str(error),
            },
        )
        return (
            f"{self.graph_context} Chosen source sampler: '{self.sampler_name}'. "
            f"{error}"
        ).strip()

    def _attach(self, latent: dict[str, Any]) -> dict[str, Any]:
        """Associate observations with their source output and parameterization."""
        samples = latent.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise TypeError("The original sampler output must contain a latent tensor.")
        actual = self.model.model.process_latent_in(samples).detach().float().cpu()
        if self.coordinates:
            if (
                self.clean is None
                or self.schedule is None
                or set(self.states) != set(self.coordinates)
            ):
                raise ValueError(
                    f"The original sampler '{self.sampler_name}' "
                    "is not supported by noise reuse on this execution path: "
                    "it finished without exposing the original noisy latent. "
                    "Run the complete connected workflow again; if this repeats, "
                    "that source node must expose ComfyUI's sampling boundaries."
                )
            if not torch.allclose(actual, self.clean, atol=1e-5, rtol=1e-5):
                raise ValueError(
                    "Saved noise belongs to a different original sampler output."
                )
            actual = self.clean
        scaling = (
            self.model.get_model_object("model_sampling") if self.coordinates else None
        )
        captures = tuple(
            CapturedSamplingState(
                requested,
                observed.coordinate,
                observed.state,
                recover_residual(scaling, actual, observed.state, observed.coordinate),
            )
            for requested, observed in self.states.items()
        )
        result = latent.copy()
        result[PROVENANCE_KEY] = SamplingProvenance(
            type(self.model.model.latent_format).__name__,
            actual,
            captures,
            self.schedule,
            self.graph_context,
        )
        LOGGER.info(
            "Sampling provenance attached",
            extra={
                "operation": "trajectory_capture",
                "sampler": self.sampler_name,
                "requested_coordinates": self.coordinates,
                "actual_coordinates": tuple(c.actual_coordinate for c in captures),
                "source_shape": tuple(actual.shape),
            },
        )
        return result
