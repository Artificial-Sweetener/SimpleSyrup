# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve the effective Comfy MODEL before global conditioning hooks execute."""

from __future__ import annotations

from typing import cast

from ..domain.conditioning_batch import select_conditioning
from .regional_lora_conditioning_sources import conditioning_hook_groups


class GlobalHookModelResolver:
    """Mirror Comfy's dynamic-to-static handoff before regional derivation."""

    def resolve(
        self,
        model: object,
        *,
        positive: object,
        negative: object,
    ) -> object:
        """Return the model that Comfy will use for global hooked conditioning."""

        if not self._has_global_hooks(positive) and not self._has_global_hooks(
            negative
        ):
            return model
        is_dynamic = getattr(model, "is_dynamic", None)
        if not callable(is_dynamic):
            raise TypeError(
                "Global conditioning hooks require MODEL dynamic-mode state."
            )
        dynamic = is_dynamic()
        if not isinstance(dynamic, bool):
            raise TypeError("MODEL is_dynamic() must return a bool.")
        if not dynamic:
            return model
        delegate_factory = getattr(model, "get_non_dynamic_delegate", None)
        if not callable(delegate_factory):
            raise TypeError(
                "Dynamic MODEL global conditioning hooks require Comfy's "
                "get_non_dynamic_delegate()."
            )
        resolved = delegate_factory()
        if resolved is model:
            raise RuntimeError("Dynamic MODEL returned itself as its static delegate.")
        resolved_is_dynamic = getattr(resolved, "is_dynamic", None)
        if not callable(resolved_is_dynamic) or resolved_is_dynamic() is not False:
            raise RuntimeError("Global conditioning hook delegate must be static.")
        return cast(object, resolved)

    @staticmethod
    def _has_global_hooks(conditioning: object) -> bool:
        """Report hooks only on consumer-defined global entry zero."""

        global_conditioning = select_conditioning(conditioning, 0)
        return bool(conditioning_hook_groups(global_conditioning))


GLOBAL_HOOK_MODEL_RESOLVER = GlobalHookModelResolver()
