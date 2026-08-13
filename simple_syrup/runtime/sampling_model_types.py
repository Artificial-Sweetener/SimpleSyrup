# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define shared callable types for model-function wrapper composition."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import torch

ApplyModel: TypeAlias = Callable[..., torch.Tensor]
ModelFunctionWrapper: TypeAlias = Callable[[ApplyModel, dict[str, Any]], torch.Tensor]
