/**
 * Layout Mutations - Simplified Direct Operations
 *
 * Provides a clean API for layout operations that are CRDT-ready.
 * Operations are synchronous and applied directly to the store.
 */
import log from 'loglevel'

import type { NodeId } from '@/lib/litegraph/src/LGraphNode'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import type {
  LayoutSource,
  LinkId,
  NodeLayout,
  Point,
  RerouteId,
  Size
} from '@/renderer/core/layout/types'

const logger = log.getLogger('LayoutMutations')

interface LayoutMutations {
  // Single node operations (synchronous, CRDT-ready)
  moveNode(nodeId: NodeId, position: Point): void
  resizeNode(nodeId: NodeId, size: Size): void
  setNodeZIndex(nodeId: NodeId, zIndex: number): void

  // Node lifecycle operations
  createNode(nodeId: NodeId, layout: Partial<NodeLayout>): void
  deleteNode(nodeId: NodeId): void

  // Link operations
  createLink(
    linkId: LinkId,
    sourceNodeId: NodeId,
    sourceSlot: number,
    targetNodeId: NodeId,
    targetSlot: number
  ): void
  deleteLink(linkId: LinkId): void

  // Reroute operations
  createReroute(
    rerouteId: RerouteId,
    position: Point,
    parentId?: LinkId,
    linkIds?: LinkId[]
  ): void
  deleteReroute(rerouteId: RerouteId): void
  moveReroute(
    rerouteId: RerouteId,
    position: Point,
    previousPosition: Point
  ): void

  // Stacking operations
  bringNodeToFront(nodeId: NodeId): void

  // Source tracking
  setSource(source: LayoutSource): void
  setActor(actor: string): void
}

/**
 * Composable for accessing layout mutations with clean destructuring API
 */
export function useLayoutMutations(): LayoutMutations {
  /**
   * Set the current mutation source
   */
  const setSource = (source: LayoutSource): void => {
    layoutStore.setSource(source)
  }

  /**
   * Set the current actor (for CRDT)
   */
  const setActor = (actor: string): void => {
    layoutStore.setActor(actor)
  }

  /**
   * Move a node to a new position
   */
  const moveNode = (nodeId: NodeId, position: Point): void => {
    const normalizedNodeId = String(nodeId)
    const existing = layoutStore.getNodeLayoutRef(normalizedNodeId).value
    if (!existing) return

    layoutStore.applyOperation({
      type: 'moveNode',
      entity: 'node',
      nodeId: normalizedNodeId,
      position,
      previousPosition: existing.position,
      timestamp: Date.now(),
      source: layoutStore.getCurrentSource(),
      actor: layoutStore.getCurrentActor()
    })
  }

  /**
   * Resize a node
   */
  const resizeNode = (nodeId: NodeId, size: Size): void => {
    const normalizedNodeId = String(nodeId)
    const existing = layoutStore.getNodeLayoutRef(normalizedNodeId).value
    if (!existing) return

    layoutStore.applyOperation({
      type: 'resizeNode',
      entity: 'node',
      nodeId: normalizedNodeId,
      size,
      previousSize: existing.size,
      timestamp: Date.now(),
      source: layoutStore.getCurrentSource(),
      actor: layoutStore.getCurrentActor()
    })
  }

  /**
   * Set node z-index
   */
  const setNodeZIndex = (nodeId: NodeId, zIndex: number): void => {
    const normalizedNodeId = String(nodeId)
    const existing = layoutStore.getNodeLayoutRef(normalizedNodeId).value
    if (!existing) return

    layoutStore.applyOperation({
      type: 'setNodeZIndex',
      entity: 'node',
      nodeId: normalizedNodeId,
      zIndex,
      previousZIndex: existing.zIndex,
      timestamp: Date.now(),
      source: layoutStore.getCurrentSource(),
      actor: layoutStore.getCurrentActor()
    })
  }

  /**
   * Create a new node
   */
  const createNode = (nodeId: NodeId, layout: Partial<NodeLayout>): void => {
    const normalizedNodeId = String(nodeId)
    const fullLayout: NodeLayout = {
      id: normalizedNodeId,
      position: layout.position ?? { x: 0, y: 0 },
      size: layout.size ?? { width: 200, height: 100 },
      zIndex: layout.zIndex ?? 0,
      visible: layout.visible ?? true,
      bounds: {
        x: layout.position?.x ?? 0,
        y: layout.position?.y ?? 0,
        width: layout.size?.width ?? 200,
        height: layout.size?.height ?? 100
      }
    }

    layoutStore.applyOperation({
      type: 'createNode',
      entity: 'node',
      nodeId: normalizedNodeId,
      layout: fullLayout,
      timestamp: Date.now(),
      source: layoutStore.getCurrentSource(),
      actor: layoutStore.getCurrentActor()
    })
  }

  /**
   * Delete a node
   */
  const deleteNode = (nodeId: NodeId): void => {
    const normalizedNodeId = String(nodeId)
    const existing = layoutStore.getNodeLayoutRef(normalizedNodeId).value
    if (!existing) return

    layoutStore.applyOperation({
      type: 'deleteNode',
      entity: 'node',
      nodeId: normalizedNodeId,
      previousLayout: existing,
      timestamp: Date.now(),
      source: layoutStore.getCurrentSource(),
      actor: layoutStore.getCurrentActor()
    })
  }

  /**
   * Bring a node to the front (highest z-index)
   */
  const bringNodeToFront = (nodeId: NodeId): void => {
    // Get all nodes to find the highest z-index
    const allNodes = layoutStore.getAllNodes().value
    let maxZIndex = 0

    for (const [, layout] of allNodes) {
      if (layout.zIndex > maxZIndex) {
        maxZIndex = layout.zIndex
      }
    }

    // Set this node's z-index to be one higher than the current max
    setNodeZIndex(nodeId, maxZIndex + 1)
  }

  /**
   * Create a new link
   */
  const createLink = (
    linkId: LinkId,
    sourceNodeId: NodeId,
    sourceSlot: number,
    targetNodeId: NodeId,
    targetSlot: number
  ): void => {
    // Normalize node IDs to strings for layout store consistency
    const normalizedSourceNodeId = String(sourceNodeId)
    const normalizedTargetNodeId = String(targetNodeId)

    logger.debug('Creating link:', {
      linkId,
      from: `${normalizedSourceNodeId}[${sourceSlot}]`,
      to: `${normalizedTargetNodeId}[${targetSlot}]`
    })
    layoutStore.applyOperation({
      type: 'createLink',
      entity: 'link',
      linkId,
      sourceNodeId: normalizedSourceNodeId,
      sourceSlot,
      targetNodeId: normalizedTargetNodeId,
      targetSlot,
      timestamp: Date.now(),
      source: layoutStore.getCurrentSource(),
      actor: layoutStore.getCurrentActor()
    })
  }

  /**
   * Delete a link
   */
  const deleteLink = (linkId: LinkId): void => {
    logger.debug('Deleting link:', linkId)
    layoutStore.applyOperation({
      type: 'deleteLink',
      entity: 'link',
      linkId,
      timestamp: Date.now(),
      source: layoutStore.getCurrentSource(),
      actor: layoutStore.getCurrentActor()
    })
  }

  /**
   * Create a new reroute
   */
  const createReroute = (
    rerouteId: RerouteId,
    position: Point,
    parentId?: LinkId,
    linkIds: LinkId[] = []
  ): void => {
    logger.debug('Creating reroute:', {
      rerouteId,
      position,
      parentId,
      linkCount: linkIds.length
    })
    layoutStore.applyOperation({
      type: 'createReroute',
      entity: 'reroute',
      rerouteId,
      position,
      parentId,
      linkIds,
      timestamp: Date.now(),
      source: layoutStore.getCurrentSource(),
      actor: layoutStore.getCurrentActor()
    })
  }

  /**
   * Delete a reroute
   */
  const deleteReroute = (rerouteId: RerouteId): void => {
    logger.debug('Deleting reroute:', rerouteId)
    layoutStore.applyOperation({
      type: 'deleteReroute',
      entity: 'reroute',
      rerouteId,
      timestamp: Date.now(),
      source: layoutStore.getCurrentSource(),
      actor: layoutStore.getCurrentActor()
    })
  }

  /**
   * Move a reroute
   */
  const moveReroute = (
    rerouteId: RerouteId,
    position: Point,
    previousPosition: Point
  ): void => {
    logger.debug('Moving reroute:', {
      rerouteId,
      from: previousPosition,
      to: position
    })
    layoutStore.applyOperation({
      type: 'moveReroute',
      entity: 'reroute',
      rerouteId,
      position,
      previousPosition,
      timestamp: Date.now(),
      source: layoutStore.getCurrentSource(),
      actor: layoutStore.getCurrentActor()
    })
  }

  return {
    setSource,
    setActor,
    moveNode,
    resizeNode,
    setNodeZIndex,
    createNode,
    deleteNode,
    bringNodeToFront,
    createLink,
    deleteLink,
    createReroute,
    deleteReroute,
    moveReroute
  }
}
