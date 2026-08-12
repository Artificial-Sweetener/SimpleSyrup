"""Select decayed reduced-global authority from denoising timesteps."""

from __future__ import annotations

import torch


class GlobalContextSchedule:
    """Limit whole-image authority to an initial denoising-step fraction."""

    def __init__(
        self, *, sigmas: torch.Tensor, active_steps: int, decay: float
    ) -> None:
        """Capture model-evaluation sigmas and the active initial step count."""

        step_sigmas = sigmas.detach().to(device="cpu", dtype=torch.float64).flatten()
        if step_sigmas.numel() < 2:
            raise ValueError(
                "Contextual Diffusion requires at least one denoising step."
            )
        self._step_sigmas = step_sigmas[:-1]
        self._active_steps = min(len(self._step_sigmas), max(0, active_steps))
        self._decay = decay

    def scale_for(self, timestep: object) -> float:
        """Return the decayed global scale for the nearest scheduled step."""

        if self._active_steps == 0:
            return 0.0
        if not isinstance(timestep, torch.Tensor) or timestep.numel() == 0:
            raise ValueError(
                "Contextual Diffusion timestep must be a non-empty tensor."
            )
        sigma = timestep.detach().flatten()[0].to(device="cpu", dtype=torch.float64)
        step_index = int(torch.argmin(torch.abs(self._step_sigmas - sigma)).item())
        if step_index >= self._active_steps:
            return 0.0
        return self._decay**step_index
