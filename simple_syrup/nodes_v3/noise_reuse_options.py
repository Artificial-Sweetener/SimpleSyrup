"""Expose captured noise reuse and optional full-resolution inversion finishing."""

from __future__ import annotations

from typing import Any, cast

from ..domain.noise_inversion import InversionMethod
from ..domain.noise_reuse import NoiseReuseOptions
from ..domain.sampler_options import SamplerOptions, append_sampler_capability
from .sampler_options_schema import (
    COMFY_IO,
    OptionsNodeBase,
    options_input,
    options_output,
)


class NoiseReuseOptionsV3(OptionsNodeBase):
    """Contribute one enabled request without changing preceding capabilities."""

    @classmethod
    def define_schema(cls) -> Any:
        """Place the explicit enable switch above independent finishing controls."""
        return COMFY_IO.Schema(
            node_id="SimpleSyrup.NoiseReuseOptions",
            display_name="Noise Reuse Options",
            category="SimpleSyrup/Sampling/Options",
            description=(
                "Reuses noise from the source sampling run, with optional "
                "full-resolution inversion finishing before sampling."
            ),
            inputs=[
                options_input(COMFY_IO),
                COMFY_IO.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip=(
                        "Requests captured source noise when enabled; disabled "
                        "passes preceding options through unchanged."
                    ),
                ),
                COMFY_IO.Int.Input(
                    "finishing_steps",
                    default=0,
                    min=0,
                    max=100,
                    tooltip=(
                        "Full-resolution inversion steps before sampling; 0 "
                        "reuses captured noise without finishing."
                    ),
                ),
                COMFY_IO.Combo.Input(
                    "finishing_method",
                    options=["euler", "heun"],
                    default="euler",
                    tooltip=(
                        "Euler uses one evaluation per finishing step; Heun uses "
                        "two for greater accuracy."
                    ),
                ),
                COMFY_IO.Float.Input(
                    "finishing_start_fraction",
                    default=0.75,
                    min=0.01,
                    max=0.99,
                    step=0.01,
                    tooltip=(
                        "With finishing enabled, captures at this fraction of the "
                        "receiving sampler's starting noise coordinate."
                    ),
                ),
            ],
            outputs=[options_output(COMFY_IO)],
        )

    @classmethod
    def execute(
        cls,
        enabled: bool = True,
        finishing_steps: int = 0,
        finishing_method: str = "euler",
        finishing_start_fraction: float = 0.75,
        options: SamplerOptions | None = None,
    ) -> tuple[SamplerOptions]:
        """Validate enabled controls and preserve the exact disabled input object."""
        if type(enabled) is not bool:
            raise TypeError("Noise reuse enabled must be a boolean.")
        if finishing_method not in ("euler", "heun"):
            raise ValueError("Finishing method must be Euler or Heun.")
        reuse = (
            NoiseReuseOptions(
                finishing_steps,
                cast(InversionMethod, finishing_method),
                finishing_start_fraction,
            )
            if enabled
            else None
        )
        return (append_sampler_capability(options, reuse),)
