"""Build one public upstream MODEL modifier without conditioning policy."""

from __future__ import annotations

from tools.anima_attention_coupling_workflow import ModifiedAnimaModel
from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference

from .matrix import PatchInteropModifier


class RegionalPatchModifierWorkflow:
    """Apply the exact public modifier declared by one matrix case."""

    def __init__(self, modifier: PatchInteropModifier) -> None:
        """Retain one immutable modifier identity."""

        if not isinstance(modifier, PatchInteropModifier):
            raise TypeError("P9.7 modifier workflow requires a modifier value.")
        self._modifier = modifier

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
        clip: NodeReference,
    ) -> ModifiedAnimaModel:
        """Return explicit MODEL/CLIP links from one public modifier node."""

        if self._modifier is PatchInteropModifier.NONE:
            return ModifiedAnimaModel(model, clip)
        if self._modifier is PatchInteropModifier.EASYCACHE:
            node = graph.add(
                "EasyCache",
                model=model,
                reuse_threshold=0.2,
                start_percent=0.15,
                end_percent=0.95,
                verbose=True,
            )
            return ModifiedAnimaModel([node, 0], clip)
        if self._modifier is PatchInteropModifier.LAZYCACHE:
            node = graph.add(
                "LazyCache",
                model=model,
                reuse_threshold=0.2,
                start_percent=0.15,
                end_percent=0.95,
                verbose=True,
            )
            return ModifiedAnimaModel([node, 0], clip)
        if self._modifier is PatchInteropModifier.OPTIMIZED_ATTENTION:
            node = graph.add(
                "ModelAttentionSelector",
                model=model,
                attention="optimized",
            )
            return ModifiedAnimaModel([node, 0], clip)
        if self._modifier is PatchInteropModifier.NEGPIP:
            node = graph.add("CLIPNegPip", model=model, clip=clip)
            return ModifiedAnimaModel([node, 0], [node, 1])
        raise AssertionError(f"Unhandled P9.7 modifier: {self._modifier.value}.")
