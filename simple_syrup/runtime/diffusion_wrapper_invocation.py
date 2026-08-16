# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate live ownership of keyed Comfy diffusion-wrapper invocations."""

from __future__ import annotations

from collections.abc import Callable

from comfy.patcher_extension import WrappersMP

from .diffusion_wrapper_executor import DiffusionWrapperExecutor


class DiffusionWrapperInvocationValidator:
    """Require the current executor to invoke one exactly registered wrapper."""

    @staticmethod
    def require_owned(
        executor: DiffusionWrapperExecutor,
        transformer_options: dict[object, object],
        *,
        key: str,
        wrapper: Callable[..., object],
    ) -> None:
        """Fail closed unless the live keyed wrapper tree owns this invocation."""

        if executor.class_obj is None:
            raise ValueError("Diffusion wrapper invocation requires a bound model.")
        if not isinstance(key, str) or not key.startswith("simple_syrup."):
            raise ValueError("Diffusion wrapper key must be namespaced.")
        wrappers = transformer_options.get("wrappers")
        if not isinstance(wrappers, dict):
            raise ValueError("Diffusion wrapper invocation has no wrapper registry.")
        by_type = wrappers.get(WrappersMP.DIFFUSION_MODEL)
        if not isinstance(by_type, dict):
            raise ValueError(
                "Diffusion wrapper invocation has no diffusion-model registry."
            )
        registered = by_type.get(key)
        if not isinstance(registered, list) or any(
            not callable(candidate) for candidate in registered
        ):
            raise ValueError("Diffusion wrapper registration is malformed.")
        if len(registered) != 1 or registered[0] is not wrapper:
            raise ValueError(
                "Diffusion wrapper invocation does not own its keyed registration."
            )


DIFFUSION_WRAPPER_INVOCATION_VALIDATOR = DiffusionWrapperInvocationValidator()
