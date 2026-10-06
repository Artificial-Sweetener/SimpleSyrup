# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose internal cache-keyed capture planning and handoff nodes."""

from __future__ import annotations

from typing import Any

from ..domain.sampler_options import SamplerOptions
from ..runtime.trajectory_capture import TrajectoryCapture
from ..services.noise_capture_request_service import NoiseCaptureRequestService
from .sampler_options_schema import COMFY_IO, OptionsNodeBase, options_input

REQUEST_TYPE = "SIMPLE_SYRUP_CAPTURE_REQUEST"
CAPTURE_TYPE = "SIMPLE_SYRUP_TRAJECTORY_CAPTURE"


class NoiseCaptureRequestV3(OptionsNodeBase):
    """Resolve the receiver's live schedule before source generation."""

    @classmethod
    def define_schema(cls) -> Any:
        """Include every schedule-affecting dependency in capture caching."""
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.NoiseCaptureRequest",
            display_name="Noise Capture Request (Internal)",
            category="SimpleSyrup/Internal",
            is_dev_only=True,
            description=(
                "Coordinates upstream state capture using receiving sampler settings."
            ),
            inputs=[
                COMFY_IO.Model.Input(
                    "model",
                    tooltip="Receiving sampler model and schedule parameterization.",
                ),
                options_input(COMFY_IO),
                COMFY_IO.Latent.Input(
                    "source_latent",
                    tooltip="Original source input used only for schedule geometry.",
                ),
                COMFY_IO.Int.Input("steps", tooltip="Receiving sampling step count."),
                COMFY_IO.String.Input(
                    "sampler_name", tooltip="Receiving sampler identifier."
                ),
                COMFY_IO.String.Input(
                    "scheduler", tooltip="Receiving scheduler identifier."
                ),
                COMFY_IO.Float.Input(
                    "denoise", tooltip="Receiving sampler denoise strength."
                ),
                COMFY_IO.String.Input(
                    "geometry_json",
                    tooltip="Validated resizing operations for schedule planning.",
                ),
                COMFY_IO.Custom(REQUEST_TYPE).Input(
                    "previous",
                    optional=True,
                    tooltip="Other receiving requests sharing this source sampler.",
                ),
            ],
            outputs=[
                COMFY_IO.Custom(REQUEST_TYPE).Output(
                    "requests", tooltip="Exact source capture coordinates."
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        source_latent: dict[str, Any],
        steps: int,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        geometry_json: str,
        options: SamplerOptions | None = None,
        previous: tuple[float, ...] = (),
    ) -> tuple[tuple[float, ...]]:
        """Delegate to the shared receiver schedule authority."""
        return (
            NoiseCaptureRequestService().coordinates(
                model=model,
                options=options or SamplerOptions(),
                steps=steps,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                source_latent=source_latent,
                geometry_json=geometry_json,
                previous=previous,
            ),
        )


class TrajectoryCaptureModelV3(OptionsNodeBase):
    """Install capture on a source MODEL without modifying shared model instances."""

    @classmethod
    def define_schema(cls) -> Any:
        """Expose only internal derivation dependencies."""
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.TrajectoryCaptureModel",
            display_name="Trajectory Capture Model (Internal)",
            category="SimpleSyrup/Internal",
            is_dev_only=True,
            description=(
                "Captures real sampler states for downstream latent noise reuse."
            ),
            inputs=[
                COMFY_IO.Model.Input(
                    "model", tooltip="Source MODEL cloned for capture."
                ),
                COMFY_IO.String.Input(
                    "sampler_name", tooltip="Source solver identifier."
                ),
                COMFY_IO.Sampler.Input(
                    "sampler",
                    optional=True,
                    tooltip="Connected source algorithm used by a custom sampler.",
                ),
                COMFY_IO.String.Input(
                    "graph_context",
                    optional=True,
                    default="",
                    tooltip="Source and receiving node names for error messages.",
                ),
                COMFY_IO.Custom(REQUEST_TYPE).Input(
                    "requests",
                    optional=True,
                    tooltip=(
                        "Requested real-state coordinates; omitted records format "
                        "provenance only."
                    ),
                ),
            ],
            outputs=[
                COMFY_IO.Model.Output(
                    "model", tooltip="Source MODEL carrying its capture observer."
                ),
                COMFY_IO.Custom(CAPTURE_TYPE).Output(
                    "capture", tooltip="Source-local capture lifecycle."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        sampler_name: str,
        requests: tuple[float, ...] = (),
        sampler: Any = None,
        graph_context: str = "",
    ) -> tuple[Any, TrajectoryCapture]:
        """Keep observer state scoped to this source and cache identity."""
        if not sampler_name and sampler is not None:
            function = getattr(sampler, "sampler_function", None)
            sampler_name = getattr(function, "__name__", type(sampler).__name__)
        capture = TrajectoryCapture(
            model, sampler_name, requests, graph_context=graph_context
        )
        return capture.prepare_model(), capture


class AttachSamplingProvenanceV3(OptionsNodeBase):
    """Attach a verified source capture before latents enter resizing or decode."""

    @classmethod
    def define_schema(cls) -> Any:
        """Require both completed source output and its precise capture owner."""
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.AttachSamplingProvenance",
            display_name="Attach Sampling Provenance (Internal)",
            category="SimpleSyrup/Internal",
            is_dev_only=True,
            description=(
                "Associates original sampler output with matching captured states."
            ),
            inputs=[
                COMFY_IO.Latent.Input(
                    "latent", tooltip="Completed source sampler output."
                ),
                COMFY_IO.Custom(CAPTURE_TYPE).Input(
                    "capture",
                    tooltip="Capture belonging to this exact source sampler execution.",
                ),
            ],
            outputs=[
                COMFY_IO.Latent.Output(
                    "latent",
                    tooltip="Original latent with verified sampling provenance.",
                )
            ],
        )

    @classmethod
    def execute(
        cls, latent: dict[str, Any], capture: TrajectoryCapture
    ) -> tuple[dict[str, Any]]:
        """Validate source identity before publishing capture metadata."""
        return (capture.attach(latent),)


class NoiseReuseErrorV3(OptionsNodeBase):
    """Surface graph planning failures through normal Comfy execution errors."""

    @classmethod
    def define_schema(cls) -> Any:
        """Avoid prompt-handler exception swallowing by making failure explicit."""
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.NoiseReuseError",
            display_name="Noise Reuse Validation (Internal)",
            category="SimpleSyrup/Internal",
            is_dev_only=True,
            description=(
                "Reports invalid sampling provenance before downstream sampling."
            ),
            inputs=[
                COMFY_IO.String.Input(
                    "message", tooltip="Actionable source graph validation error."
                )
            ],
            outputs=[
                COMFY_IO.Latent.Output(
                    "latent", tooltip="Validation must succeed before sampling."
                )
            ],
        )

    @classmethod
    def execute(cls, message: str) -> tuple[dict[str, Any]]:
        """Fail closed instead of substituting fresh noise or encoded pixels."""
        raise ValueError(message)
