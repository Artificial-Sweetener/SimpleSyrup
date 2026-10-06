# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify reused starting states at native scaling before any sampler algorithm runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ..shared.logging import get_logger
from .model_patcher_mutations import ModelCallableObjectPatchMutation
from .patcher_lifecycle import PATCHER_LIFECYCLE

LOGGER = get_logger(__name__)


class NoiseReuseStartVerification:
    """Check the first native scaling result without changing callbacks or tensors."""

    def __init__(
        self,
        original: Callable[..., torch.Tensor],
        expected: torch.Tensor,
        *,
        graph_context: str,
        sampler_name: str,
    ) -> None:
        """Retain one receiver's expected state and actionable graph identity."""
        self.original = original
        self.expected = expected
        self.graph_context = graph_context
        self.sampler_name = sampler_name
        self.verified = False

    def __call__(
        self,
        sigma: torch.Tensor,
        noise: torch.Tensor,
        latent_image: torch.Tensor,
        max_denoise: bool = False,
    ) -> torch.Tensor:
        """Return native scaling unchanged after validating its first result."""
        result = self.original(sigma, noise, latent_image, max_denoise)
        if not self.verified:
            if result.shape != self.expected.shape:
                raise ValueError(self._message("produced a different latent shape"))
            error = float((result.detach().float().cpu() - self.expected).abs().max())
            if not bool(torch.isfinite(result).all()) or error > 2e-5:
                raise ValueError(
                    self._message(
                        f"changed the reused starting latent (error {error:.6g})"
                    )
                )
            self.verified = True
            LOGGER.info(
                "Noise reuse starting state verified",
                extra={
                    "operation": "noise_reuse_start",
                    "sampler_name": self.sampler_name,
                    "graph_context": self.graph_context,
                    "max_error": error,
                },
            )
        return result

    def require_verified(self) -> None:
        """Reject receiver execution paths that bypass native starting-noise scaling."""
        if not self.verified:
            raise ValueError(
                self._message(
                    "is not supported by noise reuse on this execution path: "
                    "it did not expose native starting-noise scaling"
                )
            )

    def _message(self, reason: str) -> str:
        """Identify the graph traversal and selected receiving algorithm plainly."""
        return (
            f"{self.graph_context} The receiving sampler '{self.sampler_name}' "
            f"{reason}. Choose a sampler node that uses "
            "ComfyUI's native sampling boundary."
        )

    def derive_model(self, model: Any) -> Any:
        """Install verification through the authoritative MODEL lifecycle."""
        return PATCHER_LIFECYCLE.derive_model(
            model,
            (ModelCallableObjectPatchMutation("model_sampling.noise_scaling", self),),
            operation="noise_reuse_start",
        )
