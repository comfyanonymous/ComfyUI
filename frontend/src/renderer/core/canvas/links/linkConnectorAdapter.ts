import type { SlotLayout } from '@/renderer/core/layout/types'
import type { Point } from '@/lib/litegraph/src/interfaces'
import type { LGraph } from '@/lib/litegraph/src/LGraph'
import type { NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { RerouteId } from '@/lib/litegraph/src/Reroute'
import type { LinkConnector } from '@/lib/litegraph/src/canvas/LinkConnector'
import type { RenderLink } from '@/lib/litegraph/src/canvas/RenderLink'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import { app } from '@/scripts/app'
import { isSubgraph } from '@/utils/typeGuardUtil'

// Keep one adapter per graph so rendering and interaction share state.
const adapterByGraph = new WeakMap<LGraph, LinkConnectorAdapter>()

/**
 * Renderer‑agnostic adapter around LiteGraph's LinkConnector.
 *
 * - Uses layoutStore for hit‑testing (nodes/reroutes).
 * - Exposes minimal, imperative APIs to begin link drags and query drop validity.
 * - Preserves existing Vue composable behavior.
 */
export class LinkConnectorAdapter {
  constructor(
    /** Network the links belong to (typically `app.canvas.graph`). */
    readonly network: LGraph,
    readonly linkConnector: LinkConnector
  ) {}

  /**
   * The currently rendered/dragged links, typed for consumer use.
   * Prefer this over accessing `linkConnector.renderLinks` directly.
   */
  get renderLinks(): ReadonlyArray<RenderLink> {
    return this.linkConnector.renderLinks
  }

  // Drag helpers

  /**
   * Begin dragging from an output slot.
   * @param nodeId Output node id
   * @param outputIndex Output slot index
   * @param opts Optional: moveExisting (shift), fromRerouteId
   */
  beginFromOutput(
    nodeId: NodeId,
    outputIndex: number,
    opts?: { moveExisting?: boolean; fromRerouteId?: RerouteId }
  ): void {
    const node = this.network.getNodeById(nodeId)
    const output = node?.outputs?.[outputIndex]
    if (!node || !output) return

    const fromReroute = this.network.getReroute(opts?.fromRerouteId)

    if (opts?.moveExisting) {
      this.linkConnector.moveOutputLink(this.network, output)
    } else {
      this.linkConnector.dragNewFromOutput(
        this.network,
        node,
        output,
        fromReroute
      )
    }
  }

  /**
   * Begin dragging from an input slot.
   * @param nodeId Input node id
   * @param inputIndex Input slot index
   * @param opts Optional: moveExisting (when a link/floating exists), fromRerouteId
   */
  beginFromInput(
    nodeId: NodeId,
    inputIndex: number,
    opts?: {
      fromRerouteId?: RerouteId
      layout?: SlotLayout
      moveExisting?: boolean
    }
  ): void {
    const node = this.network.getNodeById(nodeId)
    const input = node?.inputs?.[inputIndex]
    if (!node || !input) return

    const fromReroute = this.network.getReroute(opts?.fromRerouteId)

    if (opts?.moveExisting) {
      const startPoint: Point | undefined = opts.layout
        ? [opts.layout.position.x, opts.layout.position.y]
        : undefined
      this.linkConnector.moveInputLink(this.network, input, { startPoint })
    } else {
      this.linkConnector.dragNewFromInput(
        this.network,
        node,
        input,
        fromReroute
      )
    }
  }

  // Validation helpers

  isNodeValidDrop(nodeId: NodeId): boolean {
    const node = this.network.getNodeById(nodeId)
    if (!node) return false
    return this.linkConnector.isNodeValidDrop(node)
  }

  isInputValidDrop(nodeId: NodeId, inputIndex: number): boolean {
    const node = this.network.getNodeById(nodeId)
    const input = node?.inputs?.[inputIndex]
    if (!node || !input) return false
    return this.linkConnector.isInputValidDrop(node, input)
  }

  isOutputValidDrop(nodeId: NodeId, outputIndex: number): boolean {
    const node = this.network.getNodeById(nodeId)
    const output = node?.outputs?.[outputIndex]
    if (!node || !output) return false
    return this.linkConnector.renderLinks.some((link) =>
      link.canConnectToOutput(node, output)
    )
  }

  isRerouteValidDrop(rerouteId: RerouteId): boolean {
    const reroute = this.network.getReroute(rerouteId)
    if (!reroute) return false
    return this.linkConnector.isRerouteValidDrop(reroute)
  }

  // Drop/cancel helpers for future flows

  /** Disconnects moving links (drop on canvas/no target). */
  disconnectMovingLinks(): void {
    this.linkConnector.disconnectLinks()
  }

  /** Drops moving links onto the canvas (no target). */
  dropOnCanvas(event: CanvasPointerEvent): void {
    //Add extra check for connection to subgraphInput/subgraphOutput
    if (isSubgraph(this.network)) {
      const { canvasX, canvasY } = event
      const ioNode = this.network.getIoNodeOnPos?.(canvasX, canvasY)
      if (ioNode) {
        this.linkConnector.dropOnIoNode(ioNode, event)
        return
      }
    }
    this.linkConnector.dropOnNothing(event)
  }

  /** Resets connector state and clears any temporary flags. */
  reset(): void {
    this.linkConnector.reset()
  }
}

/** Convenience creator using the current app canvas graph. */
export function createLinkConnectorAdapter(): LinkConnectorAdapter | null {
  const graph = app.canvas?.graph
  const connector = app.canvas?.linkConnector
  if (!graph || !connector) return null

  const adapter = adapterByGraph.get(graph)
  if (adapter && adapter.linkConnector === connector) {
    return adapter
  }

  const newAdapter = new LinkConnectorAdapter(graph, connector)
  adapterByGraph.set(graph, newAdapter)
  return newAdapter
}
