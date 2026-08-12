"""Define shared callable types for model-function wrapper composition."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import torch

ApplyModel: TypeAlias = Callable[..., torch.Tensor]
ModelFunctionWrapper: TypeAlias = Callable[[ApplyModel, dict[str, Any]], torch.Tensor]
