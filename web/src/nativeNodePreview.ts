// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import type {
  ComfyApi,
  ComfyApp,
  ComfyGraph,
  ComfyGraphNode,
  ComfyNodeExecutionOutput
} from "./types";
import type { OrderedMediaPreviewTarget } from "./orderedMediaPreview";

interface PreviewNode {
  readonly id?: string | number;
  readonly graph?: ComfyGraph;
}

/** Feed input-backed media through the same output path as Preview Image. */
export class NativeNodePreview implements OrderedMediaPreviewTarget {
  private readonly publishListeners = new Set<() => void>();

  constructor(
    private readonly app: ComfyApp,
    private readonly api: ComfyApi,
    private readonly node: PreviewNode
  ) {}

  /** Publish media as an execution-shaped output for both node renderers. */
  publish(output: ComfyNodeExecutionOutput): void {
    const localNodeId = this.localNodeId();
    if (!localNodeId) return;
    const outputLocator = this.outputLocator(localNodeId);
    this.app.nodeOutputs ??= {};
    this.app.nodeOutputs[localNodeId] = output;
    this.app.nodeOutputs[outputLocator] = output;
    for (const executionId of this.executionIds(localNodeId)) {
      this.api.dispatchEvent(
        new CustomEvent("executed", {
          detail: { node: executionId, display_node: executionId, output }
        })
      );
    }
    for (const listener of this.publishListeners) listener();
  }

  /** Notify renderer adapters after native output changes. */
  subscribe(listener: () => void): () => void {
    this.publishListeners.add(listener);
    return () => this.publishListeners.delete(listener);
  }

  /** Remove loader-owned media from Comfy's native preview surface. */
  clear(): void {
    this.publish({ images: [], animated: [] });
  }

  private localNodeId(): string | undefined {
    if (this.node.id === undefined) return undefined;
    return String(this.node.id);
  }

  /** Match the output key selected by Comfy's Nodes 2.0 subgraph renderer. */
  private outputLocator(localNodeId: string): string {
    const graphId = this.node.graph?.id;
    const isSubgraph = this.node.graph && this.node.graph !== this.app.rootGraph;
    return isSubgraph && graphId !== undefined
      ? `${String(graphId)}:${localNodeId}`
      : localNodeId;
  }

  /** Resolve instance paths because Comfy events consume execution IDs, not graph locators. */
  private executionIds(localNodeId: string): string[] {
    const graph = this.node.graph;
    const rootGraph = this.app.rootGraph;
    if (!graph || !rootGraph || graph === rootGraph || graph.isRootGraph) {
      return [localNodeId];
    }
    const parentPaths = findGraphInstancePaths(rootGraph, graph);
    return parentPaths.length > 0
      ? parentPaths.map((path) => `${path}:${localNodeId}`)
      : [];
  }
}

/** Find every root-to-instance path that embeds one shared subgraph definition. */
function findGraphInstancePaths(
  root: ComfyGraph,
  target: ComfyGraph,
  visited: ReadonlySet<ComfyGraph> = new Set()
): string[] {
  if (visited.has(root)) return [];
  const nextVisited = new Set(visited);
  nextVisited.add(root);
  const paths: string[] = [];
  for (const node of root.nodes ?? []) {
    const nodeId = graphNodeId(node);
    if (!nodeId || !node.subgraph) continue;
    if (node.subgraph === target) paths.push(nodeId);
    for (const nestedPath of findGraphInstancePaths(
      node.subgraph,
      target,
      nextVisited
    )) {
      paths.push(`${nodeId}:${nestedPath}`);
    }
  }
  return paths;
}

/** Normalize one graph-node identity without accepting empty execution segments. */
function graphNodeId(node: ComfyGraphNode): string | undefined {
  if (node.id === undefined) return undefined;
  const identity = String(node.id);
  return identity.length > 0 ? identity : undefined;
}
