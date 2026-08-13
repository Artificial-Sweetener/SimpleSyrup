# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute one resumable P10.3 corpus through bounded managed-Comfy segments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tools.attention_coupling_benchmark.manifest_types import BenchmarkManifest
from tools.attention_coupling_benchmark.results import parse_completed_outputs
from tools.comfy_api import LoopbackComfyClient
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer

from .blind import BlindCorpusStager
from .corpus import VisualCorpusRecorder
from .input_artifacts import VisualPositionInputs
from .matrix import SourcePosition, VisualPosition, source_positions, visual_positions
from .run_artifacts import VisualBenchmarkRunArtifacts
from .workflow import BuiltVisualWorkflow, VisualBenchmarkWorkflowBuilder

LOGGER = logging.getLogger(__name__)
DEFAULT_SEGMENT_SIZE = 10


@dataclass(frozen=True)
class PendingCorpusWork:
    """Describe exact unrecorded sources and scored outputs in frozen order."""

    sources: tuple[SourcePosition, ...]
    visuals: tuple[VisualPosition, ...]

    @property
    def count(self) -> int:
        """Return the number of positions still requiring execution."""

        return len(self.sources) + len(self.visuals)


class VisualCorpusExecutor:
    """Coordinate durable corpus work across bounded clean Comfy lifetimes."""

    def __init__(
        self,
        *,
        comfy_root: Path,
        run_artifacts: VisualBenchmarkRunArtifacts,
        manifest: BenchmarkManifest,
        readiness_timeout: float,
        prompt_timeout: float,
        segment_size: int = DEFAULT_SEGMENT_SIZE,
    ) -> None:
        """Retain validated execution policy and focused collaborators."""

        if segment_size < 1:
            raise ValueError("P10.3 segment size must be at least one.")
        self._comfy_root = comfy_root.resolve()
        self._run = run_artifacts
        self._manifest = manifest
        self._readiness_timeout = readiness_timeout
        self._prompt_timeout = prompt_timeout
        self._segment_size = segment_size
        self._builder = VisualBenchmarkWorkflowBuilder()
        self._recorder = VisualCorpusRecorder(run_artifacts.root, manifest)
        self._sources = source_positions(manifest)
        self._visuals = visual_positions(manifest)
        self._cases = {case.case_id: case for case in manifest.cases}
        self._required = required_node_ids(
            manifest,
            self._builder,
            self._sources,
            self._visuals,
            run_artifacts.run_id,
        )

    def execute(
        self,
        *,
        max_sources: int | None = None,
        max_visuals: int | None = None,
    ) -> Path:
        """Execute bounded pending segments and publish only a complete corpus."""

        pending = self.pending_work(
            max_sources=max_sources,
            max_visuals=max_visuals,
        )
        source_index = 0
        visual_index = 0
        while source_index < len(pending.sources) or visual_index < len(
            pending.visuals
        ):
            segment_sources = pending.sources[
                source_index : source_index + self._segment_size
            ]
            remaining_capacity = self._segment_size - len(segment_sources)
            segment_visuals = pending.visuals[
                visual_index : visual_index + remaining_capacity
            ]
            self._execute_segment(segment_sources, segment_visuals)
            source_index += len(segment_sources)
            visual_index += len(segment_visuals)

        if max_sources is not None or max_visuals is not None:
            return self._recorder.journal_path
        corpus_path = self._recorder.finalize(
            system_stats=self._run.system_stats,
            cleanup_verified=True,
        )
        packet_path = BlindCorpusStager(self._manifest).stage(corpus_path).packet_path
        self._run.record_completed(packet_path)
        return packet_path

    def pending_work(
        self,
        *,
        max_sources: int | None = None,
        max_visuals: int | None = None,
    ) -> PendingCorpusWork:
        """Select frozen ordered work while preserving exact resume semantics."""

        bounded_sources = (
            self._sources[:max_sources] if max_sources is not None else self._sources
        )
        bounded_visuals = (
            self._visuals[:max_visuals] if max_visuals is not None else self._visuals
        )
        pending_sources = tuple(
            source
            for source in bounded_sources
            if source.source_id not in self._recorder.completed_source_ids
        )
        available_sources = self._recorder.completed_source_ids | {
            source.source_id for source in bounded_sources
        }
        pending_visuals: list[VisualPosition] = []
        for visual in bounded_visuals:
            if visual.artifact_id in self._recorder.completed_artifact_ids:
                continue
            if (
                visual.source_id is not None
                and visual.source_id not in available_sources
            ):
                if max_sources is not None:
                    break
                raise RuntimeError(
                    f"P10.3 required neutral source is unavailable: {visual.source_id}"
                )
            pending_visuals.append(visual)
        return PendingCorpusWork(pending_sources, tuple(pending_visuals))

    def _execute_segment(
        self,
        sources: tuple[SourcePosition, ...],
        visuals: tuple[VisualPosition, ...],
    ) -> None:
        """Execute at most one bounded batch in a fresh owned Comfy process."""

        segment = self._run.create_segment()
        completed = 0
        all_input_cleanup = True
        try:
            with ManagedComfyServer(
                comfy_root=self._comfy_root,
                artifacts=segment,
                required_node_ids=self._required,
                readiness_timeout=self._readiness_timeout,
            ) as running:
                for source in sources:
                    LOGGER.info("P10.3 segment source: %s", source.source_id)
                    workflow = self._builder.build_source(
                        self._manifest,
                        source,
                        run_id=self._run.run_id,
                    )
                    _execute_source(
                        running.client,
                        self._recorder,
                        source,
                        workflow,
                        prompt_timeout=self._prompt_timeout,
                    )
                    completed += 1
                for visual in visuals:
                    LOGGER.info("P10.3 segment visual: %s", visual.artifact_id)
                    cleanup = self._execute_visual(running.client, visual)
                    all_input_cleanup = all_input_cleanup and cleanup
                    completed += 1
                system_stats = running.system_stats
                port = running.port
                process = running.process
            process_running = process.is_running
            port_available = is_loopback_port_available(port)
            segment.record_cleanup(
                process_running=process_running,
                port_available=port_available,
            )
            if not all_input_cleanup:
                raise RuntimeError("P10.3 segment input cleanup failed.")
            self._run.record_segment_completed(
                segment,
                completed_positions=completed,
                system_stats=system_stats,
            )
        except BaseException as error:
            segment.record_failure(error)
            self._run.record_segment_failed(segment, error)
            raise

    def _execute_visual(
        self,
        client: LoopbackComfyClient,
        position: VisualPosition,
    ) -> bool:
        """Execute one visual with exact temporary input cleanup."""

        source_path = (
            self._recorder.source_path(position.source_id)
            if position.source_id is not None
            else None
        )
        inputs = VisualPositionInputs(
            self._comfy_root / "input",
            self._run.run_id,
            position,
            self._cases[position.case_id],
            source_path=source_path,
        )
        with inputs:
            workflow = self._builder.build_visual(
                self._manifest,
                position,
                run_id=self._run.run_id,
                mask_names=inputs.mask_names,
                source_name=inputs.source_name,
            )
            _execute_visual(
                client,
                self._recorder,
                position,
                workflow,
                mask_sha256s=inputs.mask_sha256s,
                prompt_timeout=self._prompt_timeout,
            )
        return inputs.cleanup_verified


def _execute_source(
    client: LoopbackComfyClient,
    recorder: VisualCorpusRecorder,
    source: SourcePosition,
    workflow: BuiltVisualWorkflow,
    *,
    prompt_timeout: float,
) -> None:
    """Execute and record one neutral source through the shared history parser."""

    prompt_id = client.submit(workflow.prompt)
    history = client.wait_for_history(prompt_id, timeout=prompt_timeout)
    outputs = parse_completed_outputs(
        history,
        metrics_node_id=workflow.metrics_node_id,
        save_node_id=workflow.save_node_id,
    )
    recorder.record_source(
        source,
        workflow,
        outputs,
        history=history,
        prompt_id=prompt_id,
        image_bytes=client.download_image(outputs.image),
    )


def _execute_visual(
    client: LoopbackComfyClient,
    recorder: VisualCorpusRecorder,
    position: VisualPosition,
    workflow: BuiltVisualWorkflow,
    *,
    mask_sha256s: tuple[str, ...],
    prompt_timeout: float,
) -> None:
    """Execute and record one exact scored-output position."""

    prompt_id = client.submit(workflow.prompt)
    history = client.wait_for_history(prompt_id, timeout=prompt_timeout)
    outputs = parse_completed_outputs(
        history,
        metrics_node_id=workflow.metrics_node_id,
        save_node_id=workflow.save_node_id,
    )
    recorder.record_visual(
        position,
        workflow,
        outputs,
        history=history,
        prompt_id=prompt_id,
        image_bytes=client.download_image(outputs.image),
        mask_sha256s=mask_sha256s,
    )


def required_node_ids(
    manifest: BenchmarkManifest,
    builder: VisualBenchmarkWorkflowBuilder,
    sources: tuple[SourcePosition, ...],
    positions: tuple[VisualPosition, ...],
    run_id: str,
) -> frozenset[str]:
    """Build representative graphs and return every exact required node ID."""

    required = set(
        builder.build_source(manifest, sources[0], run_id=run_id).required_node_ids
    )
    case = next(
        value for value in manifest.cases if value.case_id == positions[0].case_id
    )
    mask_names = tuple(f"mask-{index}.png" for index in range(len(case.masks)))
    for position in positions[:10]:
        required.update(
            builder.build_visual(
                manifest,
                position,
                run_id=run_id,
                mask_names=mask_names,
                source_name="source.png" if position.is_refinement else None,
            ).required_node_ids
        )
    return frozenset(required)
