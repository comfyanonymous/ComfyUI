/**
 * Layout Store - Single Source of Truth
 *
 * Uses Yjs for efficient local state management and future collaboration.
 * CRDT ensures conflict-free operations for both single and multi-user scenarios.
 */
import log from 'loglevel'
import { computed, customRef, ref } from 'vue'
import type { ComputedRef, Ref } from 'vue'
import * as Y from 'yjs'

import { removeNodeTitleHeight } from '@/renderer/core/layout/utils/nodeSizeUtil'

import { ACTOR_CONFIG } from '@/renderer/core/layout/constants'
import { LayoutSource } from '@/renderer/core/layout/types'
import type {
  BatchUpdateBoundsOperation,
  Bounds,
  CreateLinkOperation,
  CreateNodeOperation,
  CreateRerouteOperation,
  DeleteLinkOperation,
  DeleteNodeOperation,
  DeleteRerouteOperation,
  LayoutChange,
  LayoutOperation,
  LayoutStore,
  LinkId,
  LinkLayout,
  LinkSegmentLayout,
  MoveNodeOperation,
  MoveRerouteOperation,
  NodeBoundsUpdate,
  NodeId,
  NodeLayout,
  Point,
  RerouteId,
  RerouteLayout,
  ResizeNodeOperation,
  SetNodeZIndexOperation,
  SlotLayout
} from '@/renderer/core/layout/types'
import {
  isBoundsEqual,
  isPointEqual
} from '@/renderer/core/layout/utils/geometry'
import {
  REROUTE_RADIUS,
  boundsIntersect,
  pointInBounds
} from '@/renderer/core/layout/utils/layoutMath'
import { makeLinkSegmentKey } from '@/renderer/core/layout/utils/layoutUtils'
import {
  layoutToYNode,
  yNodeToLayout
} from '@/renderer/core/layout/utils/mappers'
import type { NodeLayoutMap } from '@/renderer/core/layout/utils/mappers'
import { SpatialIndexManager } from '@/renderer/core/spatial/SpatialIndex'

type YEventChange = {
  action: 'add' | 'update' | 'delete'
  oldValue: unknown
}

const logger = log.getLogger('LayoutStore')

// Utility functions
function asRerouteId(id: string | number): RerouteId {
  return Number(id)
}

function asLinkId(id: string | number): LinkId {
  return Number(id)
}

interface LinkData {
  id: LinkId
  sourceNodeId: NodeId
  targetNodeId: NodeId
  sourceSlot: number
  targetSlot: number
}

interface RerouteData {
  id: RerouteId
  position: Point
  parentId: LinkId
  linkIds: LinkId[]
}

// Generic typed Y.Map interface
interface TypedYMap<T> {
  get<K extends keyof T>(key: K): T[K] | undefined
  get<K extends keyof T>(key: K, defaultValue: T[K]): T[K]
}

class LayoutStoreImpl implements LayoutStore {
  private static readonly REROUTE_DEFAULTS: RerouteData = {
    id: 0,
    position: { x: 0, y: 0 },
    parentId: 0,
    linkIds: []
  }

  // Yjs document and shared data structures
  private ydoc = new Y.Doc()
  private ynodes: Y.Map<NodeLayoutMap> // Maps nodeId -> NodeLayoutMap containing NodeLayout data
  private ylinks: Y.Map<Y.Map<unknown>> // Maps linkId -> Y.Map containing link data
  private yreroutes: Y.Map<Y.Map<unknown>> // Maps rerouteId -> Y.Map containing reroute data
  private yoperations: Y.Array<LayoutOperation> // Operation log

  // Vue reactivity layer
  private version = 0
  private currentSource: LayoutSource =
    ACTOR_CONFIG.DEFAULT_SOURCE as LayoutSource
  private currentActor = `${ACTOR_CONFIG.USER_PREFIX}${Math.random()
    .toString(36)
    .substring(2, 2 + ACTOR_CONFIG.ID_LENGTH)}`

  // Change listeners
  private changeListeners = new Set<(change: LayoutChange) => void>()

  // CustomRef cache and trigger functions
  private nodeRefs = new Map<NodeId, Ref<NodeLayout | null>>()
  private nodeTriggers = new Map<NodeId, () => void>()

  // New data structures for hit testing
  private linkLayouts = new Map<LinkId, LinkLayout>()
  private linkSegmentLayouts = new Map<string, LinkSegmentLayout>() // Internal string key: ${linkId}:${rerouteId ?? 'final'}
  private slotLayouts = new Map<string, SlotLayout>()
  private rerouteLayouts = new Map<RerouteId, RerouteLayout>()

  // Spatial index managers
  private spatialIndex: SpatialIndexManager // For nodes
  private linkSegmentSpatialIndex: SpatialIndexManager // For link segments (single index for all link geometry)
  private slotSpatialIndex: SpatialIndexManager // For slots
  private rerouteSpatialIndex: SpatialIndexManager // For reroutes

  // Vue dragging state for selection toolbox (public ref for direct mutation)
  public isDraggingVueNodes = ref(false)
  // Vue resizing state to prevent drag from activating during resize
  public isResizingVueNodes = ref(false)

  /**
   * Flag indicating slot positions are pending sync after graph reconfiguration.
   * When true, link rendering should be skipped to avoid drawing with stale positions.
   */
  private _pendingSlotSync = false

  get pendingSlotSync(): boolean {
    return this._pendingSlotSync
  }

  get hasSlotLayouts(): boolean {
    return this.slotLayouts.size > 0
  }

  setPendingSlotSync(value: boolean): void {
    this._pendingSlotSync = value
  }

  constructor() {
    // Initialize Yjs data structures
    this.ynodes = this.ydoc.getMap('nodes')
    this.ylinks = this.ydoc.getMap('links')
    this.yreroutes = this.ydoc.getMap('reroutes')
    this.yoperations = this.ydoc.getArray('operations')

    // Initialize spatial index managers
    this.spatialIndex = new SpatialIndexManager()
    this.linkSegmentSpatialIndex = new SpatialIndexManager() // Single index for all link geometry
    this.slotSpatialIndex = new SpatialIndexManager()
    this.rerouteSpatialIndex = new SpatialIndexManager()

    // Listen for Yjs changes and trigger Vue reactivity
    this.ynodes.observe((event: Y.YMapEvent<NodeLayoutMap>) => {
      this.version++

      // Trigger all affected node refs
      event.changes.keys.forEach((_change: YEventChange, key: string) => {
        const trigger = this.nodeTriggers.get(key)
        if (trigger) {
          trigger()
        }
      })
    })

    // Listen for link changes and update spatial indexes
    this.ylinks.observe((event: Y.YMapEvent<Y.Map<unknown>>) => {
      this.version++
      event.changes.keys.forEach((change, linkIdStr) => {
        this.handleLinkChange(change, linkIdStr)
      })
    })

    // Listen for reroute changes and update spatial indexes
    this.yreroutes.observe((event: Y.YMapEvent<Y.Map<unknown>>) => {
      this.version++
      event.changes.keys.forEach((change, rerouteIdStr) => {
        this.handleRerouteChange(change, rerouteIdStr)
      })
    })
  }

  private getLinkField<K extends keyof LinkData>(
    ylink: Y.Map<unknown>,
    field: K
  ): LinkData[K] | undefined {
    const typedLink = ylink as TypedYMap<LinkData>
    return typedLink.get(field)
  }

  private getRerouteField<K extends keyof RerouteData>(
    yreroute: Y.Map<unknown>,
    field: K,
    defaultValue: RerouteData[K] = LayoutStoreImpl.REROUTE_DEFAULTS[field]
  ): RerouteData[K] {
    const typedReroute = yreroute as TypedYMap<RerouteData>
    const value = typedReroute.get(field)
    return value ?? defaultValue
  }

  /**
   * Get or create a customRef for a node layout
   */
  getNodeLayoutRef(nodeId: NodeId): Ref<NodeLayout | null> {
    let nodeRef = this.nodeRefs.get(nodeId)

    if (!nodeRef) {
      nodeRef = customRef<NodeLayout | null>((track, trigger) => {
        // Store the trigger so we can call it when Yjs changes
        this.nodeTriggers.set(nodeId, trigger)

        return {
          get: () => {
            track()
            const ynode = this.ynodes.get(nodeId)
            const layout = ynode ? yNodeToLayout(ynode) : null
            return layout
          },
          set: (newLayout: NodeLayout | null) => {
            if (newLayout === null) {
              // Delete operation
              const existing = this.ynodes.get(nodeId)
              if (existing) {
                this.applyOperation({
                  type: 'deleteNode',
                  entity: 'node',
                  nodeId,
                  timestamp: Date.now(),
                  source: this.currentSource,
                  actor: this.currentActor,
                  previousLayout: yNodeToLayout(existing)
                })
              }
            } else {
              // Update operation - detect what changed
              const existing = this.ynodes.get(nodeId)
              if (!existing) {
                // Create operation
                this.applyOperation({
                  type: 'createNode',
                  entity: 'node',
                  nodeId,
                  layout: newLayout,
                  timestamp: Date.now(),
                  source: this.currentSource,
                  actor: this.currentActor
                })
              } else {
                const existingLayout = yNodeToLayout(existing)

                // Check what properties changed
                if (
                  existingLayout.position.x !== newLayout.position.x ||
                  existingLayout.position.y !== newLayout.position.y
                ) {
                  this.applyOperation({
                    type: 'moveNode',
                    entity: 'node',
                    nodeId,
                    position: newLayout.position,
                    previousPosition: existingLayout.position,
                    timestamp: Date.now(),
                    source: this.currentSource,
                    actor: this.currentActor
                  })
                }
                if (
                  existingLayout.size.width !== newLayout.size.width ||
                  existingLayout.size.height !== newLayout.size.height
                ) {
                  this.applyOperation({
                    type: 'resizeNode',
                    entity: 'node',
                    nodeId,
                    size: newLayout.size,
                    previousSize: existingLayout.size,
                    timestamp: Date.now(),
                    source: this.currentSource,
                    actor: this.currentActor
                  })
                }
                if (existingLayout.zIndex !== newLayout.zIndex) {
                  this.applyOperation({
                    type: 'setNodeZIndex',
                    entity: 'node',
                    nodeId,
                    zIndex: newLayout.zIndex,
                    previousZIndex: existingLayout.zIndex,
                    timestamp: Date.now(),
                    source: this.currentSource,
                    actor: this.currentActor
                  })
                }
              }
            }
            trigger()
          }
        }
      })

      this.nodeRefs.set(nodeId, nodeRef)
    }

    return nodeRef
  }

  /**
   * Get nodes within bounds (reactive)
   */
  getNodesInBounds(bounds: Bounds): ComputedRef<NodeId[]> {
    return computed(() => {
      // Touch version for reactivity
      void this.version

      const result: NodeId[] = []
      for (const [nodeId] of this.ynodes) {
        const ynode = this.ynodes.get(nodeId)
        if (ynode) {
          const layout = yNodeToLayout(ynode)
          if (layout && boundsIntersect(layout.bounds, bounds)) {
            result.push(nodeId)
          }
        }
      }
      return result
    })
  }

  /**
   * Get all nodes as a reactive map
   */
  getAllNodes(): ComputedRef<ReadonlyMap<NodeId, NodeLayout>> {
    return computed(() => {
      // Touch version for reactivity
      void this.version

      const result = new Map<NodeId, NodeLayout>()
      for (const [nodeId] of this.ynodes) {
        const ynode = this.ynodes.get(nodeId)
        if (ynode) {
          const layout = yNodeToLayout(ynode)
          if (layout) {
            result.set(nodeId, layout)
          }
        }
      }
      return result
    })
  }

  /**
   * Get current version for change detection
   */
  getVersion(): ComputedRef<number> {
    return computed(() => this.version)
  }

  /**
   * Query node at point (non-reactive for performance)
   */
  queryNodeAtPoint(point: Point): NodeId | null {
    const nodes: Array<[NodeId, NodeLayout]> = []

    for (const [nodeId] of this.ynodes) {
      const ynode = this.ynodes.get(nodeId)
      if (ynode) {
        const layout = yNodeToLayout(ynode)
        if (layout) {
          nodes.push([nodeId, layout])
        }
      }
    }

    // Sort by zIndex (top to bottom)
    nodes.sort(([, a], [, b]) => b.zIndex - a.zIndex)

    for (const [nodeId, layout] of nodes) {
      if (pointInBounds(point, layout.bounds)) {
        return nodeId
      }
    }

    return null
  }

  /**
   * Query nodes in bounds (non-reactive for performance)
   */
  queryNodesInBounds(bounds: Bounds): NodeId[] {
    return this.spatialIndex.query(bounds)
  }

  /**
   * Update link layout data (for geometry/debug, no separate spatial index)
   */
  updateLinkLayout(linkId: LinkId, layout: LinkLayout): void {
    const existing = this.linkLayouts.get(linkId)

    // Short-circuit if bounds and centerPos unchanged
    if (
      existing &&
      isBoundsEqual(existing.bounds, layout.bounds) &&
      isPointEqual(existing.centerPos, layout.centerPos)
    ) {
      // Only update path if provided (for hit detection)
      if (layout.path) {
        existing.path = layout.path
      }
      return
    }

    this.linkLayouts.set(linkId, layout)
  }

  /**
   * Delete link layout data
   */
  deleteLinkLayout(linkId: LinkId): void {
    const deleted = this.linkLayouts.delete(linkId)
    if (deleted) {
      // Clean up any segment layouts for this link
      const keysToDelete: string[] = []
      for (const [key] of this.linkSegmentLayouts) {
        if (key.startsWith(`${linkId}:`)) {
          keysToDelete.push(key)
        }
      }
      for (const key of keysToDelete) {
        this.linkSegmentLayouts.delete(key)
        this.linkSegmentSpatialIndex.remove(key)
      }
    }
  }

  /**
   * Update slot layout data
   */
  updateSlotLayout(key: string, layout: SlotLayout): void {
    const existing = this.slotLayouts.get(key)

    if (existing) {
      // Short-circuit if geometry is unchanged
      if (
        isPointEqual(existing.position, layout.position) &&
        isBoundsEqual(existing.bounds, layout.bounds)
      ) {
        return
      }
      // Update spatial index
      this.slotSpatialIndex.update(key, layout.bounds)
    } else {
      // Insert into spatial index
      this.slotSpatialIndex.insert(key, layout.bounds)
    }

    this.slotLayouts.set(key, layout)
  }

  /**
   * Batch update slot layouts and spatial index in one pass
   */
  batchUpdateSlotLayouts(
    updates: Array<{ key: string; layout: SlotLayout }>
  ): void {
    if (!updates.length) return

    // Update spatial index and map entries (skip unchanged)
    for (const { key, layout } of updates) {
      const existing = this.slotLayouts.get(key)

      if (existing) {
        // Short-circuit if geometry is unchanged
        if (
          isPointEqual(existing.position, layout.position) &&
          isBoundsEqual(existing.bounds, layout.bounds)
        ) {
          continue
        }
        this.slotSpatialIndex.update(key, layout.bounds)
      } else {
        this.slotSpatialIndex.insert(key, layout.bounds)
      }
      this.slotLayouts.set(key, layout)
    }
  }

  /**
   * Delete slot layout data
   */
  deleteSlotLayout(key: string): void {
    const deleted = this.slotLayouts.delete(key)
    if (deleted) {
      // Remove from spatial index
      this.slotSpatialIndex.remove(key)
    }
  }

  /**
   * Clear all slot layouts and their spatial index (O(1) operations)
   * Used when switching rendering modes (Vue ↔ LiteGraph)
   */
  clearAllSlotLayouts(): void {
    this.slotLayouts.clear()
    this.slotSpatialIndex.clear()
  }

  /**
   * Update reroute layout data
   */
  updateRerouteLayout(rerouteId: RerouteId, layout: RerouteLayout): void {
    const existing = this.rerouteLayouts.get(rerouteId)

    if (!existing) {
      logger.debug('Adding reroute layout:', {
        rerouteId,
        position: layout.position,
        bounds: layout.bounds
      })
    }

    if (existing) {
      // Update spatial index
      this.rerouteSpatialIndex.update(String(rerouteId), layout.bounds) // Spatial index uses strings
    } else {
      // Insert into spatial index
      this.rerouteSpatialIndex.insert(String(rerouteId), layout.bounds) // Spatial index uses strings
    }

    this.rerouteLayouts.set(rerouteId, layout)
  }

  /**
   * Delete reroute layout data
   */
  deleteRerouteLayout(rerouteId: RerouteId): void {
    const deleted = this.rerouteLayouts.delete(rerouteId)
    if (deleted) {
      // Remove from spatial index
      this.rerouteSpatialIndex.remove(String(rerouteId)) // Spatial index uses strings
    }
  }

  /**
   * Get link layout data
   */
  getLinkLayout(linkId: LinkId): LinkLayout | null {
    return this.linkLayouts.get(linkId) || null
  }

  /**
   * Get slot layout data
   */
  getSlotLayout(key: string): SlotLayout | null {
    return this.slotLayouts.get(key) || null
  }

  /**
   * Get reroute layout data
   */
  getRerouteLayout(rerouteId: RerouteId): RerouteLayout | null {
    return this.rerouteLayouts.get(rerouteId) || null
  }

  /**
   * Returns all slot layout keys currently tracked by the store.
   * Useful for global passes without relying on spatial queries.
   */
  getAllSlotKeys(): string[] {
    return Array.from(this.slotLayouts.keys())
  }

  /**
   * Update link segment layout data
   */
  updateLinkSegmentLayout(
    linkId: LinkId,
    rerouteId: RerouteId | null,
    layout: Omit<LinkSegmentLayout, 'linkId' | 'rerouteId'>
  ): void {
    const key = makeLinkSegmentKey(linkId, rerouteId)
    const existing = this.linkSegmentLayouts.get(key)

    // Short-circuit if bounds and centerPos unchanged (prevents spatial index churn)
    if (
      existing &&
      isBoundsEqual(existing.bounds, layout.bounds) &&
      isPointEqual(existing.centerPos, layout.centerPos)
    ) {
      // Only update path if provided (for hit detection)
      if (layout.path) {
        existing.path = layout.path
      }
      return
    }

    const fullLayout: LinkSegmentLayout = {
      ...layout,
      linkId,
      rerouteId
    }

    if (!existing) {
      logger.debug('Adding link segment:', {
        linkId,
        rerouteId,
        bounds: layout.bounds,
        hasPath: !!layout.path
      })
    }

    if (existing) {
      // Update spatial index
      this.linkSegmentSpatialIndex.update(key, layout.bounds)
    } else {
      // Insert into spatial index
      this.linkSegmentSpatialIndex.insert(key, layout.bounds)
    }

    this.linkSegmentLayouts.set(key, fullLayout)
  }

  /**
   * Delete link segment layout data
   */
  deleteLinkSegmentLayout(linkId: LinkId, rerouteId: RerouteId | null): void {
    const key = makeLinkSegmentKey(linkId, rerouteId)
    const deleted = this.linkSegmentLayouts.delete(key)
    if (deleted) {
      // Remove from spatial index
      this.linkSegmentSpatialIndex.remove(key)
    }
  }

  /**
   * Query link segment at point (returns structured data)
   */
  queryLinkSegmentAtPoint(
    point: Point,
    ctx?: CanvasRenderingContext2D
  ): { linkId: LinkId; rerouteId: RerouteId | null } | null {
    // Determine tolerance from current canvas state (if available)
    // - Use the caller-provided ctx.lineWidth (LGraphCanvas sets this to connections_width + padding)
    // - Fall back to a sensible default when ctx is not provided
    const hitWidth = ctx?.lineWidth ?? 10
    const halfSize = Math.max(10, hitWidth) // keep a minimum window for spatial index

    // Use spatial index to get candidate segments
    const searchArea = {
      x: point.x - halfSize,
      y: point.y - halfSize,
      width: halfSize * 2,
      height: halfSize * 2
    }
    const candidateKeys = this.linkSegmentSpatialIndex.query(searchArea)

    if (candidateKeys.length > 0) {
      logger.debug('Checking link segments at point:', {
        point,
        candidateCount: candidateKeys.length,
        tolerance: hitWidth
      })
    }

    // Precise hit test only on candidates
    for (const key of candidateKeys) {
      const segmentLayout = this.linkSegmentLayouts.get(key)
      if (!segmentLayout) continue

      if (ctx && segmentLayout.path) {
        // Match LiteGraph behavior: hit test uses device pixel ratio for coordinates
        const dpi =
          (typeof window !== 'undefined' && window?.devicePixelRatio) || 1
        const hit = ctx.isPointInStroke(
          segmentLayout.path,
          point.x * dpi,
          point.y * dpi
        )

        if (hit) {
          logger.debug('Link segment hit:', {
            linkId: segmentLayout.linkId,
            rerouteId: segmentLayout.rerouteId,
            point
          })
          return {
            linkId: segmentLayout.linkId,
            rerouteId: segmentLayout.rerouteId
          }
        }
      } else if (pointInBounds(point, segmentLayout.bounds)) {
        // Fallback to bounding box test
        return {
          linkId: segmentLayout.linkId,
          rerouteId: segmentLayout.rerouteId
        }
      }
    }

    return null
  }

  /**
   * Query link at point (derived from segment query)
   */
  queryLinkAtPoint(
    point: Point,
    ctx?: CanvasRenderingContext2D
  ): LinkId | null {
    // Invoke segment query and return just the linkId
    const segment = this.queryLinkSegmentAtPoint(point, ctx)
    return segment ? segment.linkId : null
  }

  /**
   * Query slot at point
   */
  querySlotAtPoint(point: Point): SlotLayout | null {
    // Use spatial index to get candidate slots
    const searchArea = {
      x: point.x - 10, // Tolerance for slot size
      y: point.y - 10,
      width: 20,
      height: 20
    }
    const candidateSlotKeys = this.slotSpatialIndex.query(searchArea)

    // Check precise bounds for candidates
    for (const key of candidateSlotKeys) {
      const slotLayout = this.slotLayouts.get(key)
      if (slotLayout && pointInBounds(point, slotLayout.bounds)) {
        return slotLayout
      }
    }
    return null
  }

  /**
   * Query reroute at point
   */
  queryRerouteAtPoint(point: Point): RerouteLayout | null {
    // Use spatial index to get candidate reroutes
    const maxRadius = 20 // Maximum expected reroute radius
    const searchArea = {
      x: point.x - maxRadius,
      y: point.y - maxRadius,
      width: maxRadius * 2,
      height: maxRadius * 2
    }
    const candidateRerouteKeys = this.rerouteSpatialIndex.query(searchArea)

    if (candidateRerouteKeys.length > 0) {
      logger.debug('Checking reroutes at point:', {
        point,
        candidateCount: candidateRerouteKeys.length
      })
    }

    // Check precise distance for candidates
    for (const rerouteKey of candidateRerouteKeys) {
      const rerouteId = asRerouteId(rerouteKey)
      const rerouteLayout = this.rerouteLayouts.get(rerouteId)
      if (rerouteLayout) {
        const dx = point.x - rerouteLayout.position.x
        const dy = point.y - rerouteLayout.position.y
        const distance = Math.sqrt(dx * dx + dy * dy)

        if (distance <= rerouteLayout.radius) {
          logger.debug('Reroute hit:', {
            rerouteId: rerouteLayout.id,
            position: rerouteLayout.position,
            distance
          })
          return rerouteLayout
        }
      }
    }
    return null
  }

  /**
   * Query all items in bounds
   */
  queryItemsInBounds(bounds: Bounds): {
    nodes: NodeId[]
    links: LinkId[]
    slots: string[]
    reroutes: RerouteId[]
  } {
    // Query segments and union their linkIds
    const segmentKeys = this.linkSegmentSpatialIndex.query(bounds)
    const linkIds = new Set<LinkId>()
    for (const key of segmentKeys) {
      const segment = this.linkSegmentLayouts.get(key)
      if (segment) {
        linkIds.add(segment.linkId)
      }
    }

    return {
      nodes: this.queryNodesInBounds(bounds),
      links: Array.from(linkIds),
      slots: this.slotSpatialIndex.query(bounds),
      reroutes: this.rerouteSpatialIndex
        .query(bounds)
        .map((key) => asRerouteId(key))
    }
  }

  /**
   * Apply a layout operation using Yjs transactions
   */
  applyOperation(operation: LayoutOperation): void {
    // Create change object outside transaction so we can use it after
    const change: LayoutChange = {
      type: 'update',
      nodeIds: [],
      timestamp: operation.timestamp,
      source: operation.source,
      operation
    }

    // Use Yjs transaction for atomic updates
    this.ydoc.transact(() => {
      // Add operation to log
      this.yoperations.push([operation])

      // Apply the operation
      this.applyOperationInTransaction(operation, change)
    }, this.currentActor)

    // Post-transaction updates
    this.finalizeOperation(change)
  }

  /**
   * Apply operation within a transaction
   */
  private applyOperationInTransaction(
    operation: LayoutOperation,
    change: LayoutChange
  ): void {
    switch (operation.type) {
      case 'moveNode':
        this.handleMoveNode(operation as MoveNodeOperation, change)
        break
      case 'resizeNode':
        this.handleResizeNode(operation as ResizeNodeOperation, change)
        break
      case 'setNodeZIndex':
        this.handleSetNodeZIndex(operation as SetNodeZIndexOperation, change)
        break
      case 'createNode':
        this.handleCreateNode(operation as CreateNodeOperation, change)
        break
      case 'deleteNode':
        this.handleDeleteNode(operation as DeleteNodeOperation, change)
        break
      case 'batchUpdateBounds':
        this.handleBatchUpdateBounds(
          operation as BatchUpdateBoundsOperation,
          change
        )
        break
      case 'createLink':
        this.handleCreateLink(operation as CreateLinkOperation, change)
        break
      case 'deleteLink':
        this.handleDeleteLink(operation as DeleteLinkOperation, change)
        break
      case 'createReroute':
        this.handleCreateReroute(operation as CreateRerouteOperation, change)
        break
      case 'deleteReroute':
        this.handleDeleteReroute(operation as DeleteRerouteOperation, change)
        break
      case 'moveReroute':
        this.handleMoveReroute(operation as MoveRerouteOperation, change)
        break
    }
  }

  /**
   * Finalize operation after transaction
   */
  private finalizeOperation(change: LayoutChange): void {
    // Update version
    this.version++

    // Manually trigger affected node refs after transaction
    // This is needed because Yjs observers don't fire for property changes
    change.nodeIds.forEach((nodeId) => {
      const trigger = this.nodeTriggers.get(nodeId)
      if (trigger) {
        trigger()
      }
    })

    // Notify listeners (after transaction completes)
    setTimeout(() => this.notifyChange(change), 0)
  }

  /**
   * Subscribe to layout changes
   */
  onChange(callback: (change: LayoutChange) => void): () => void {
    this.changeListeners.add(callback)
    return () => this.changeListeners.delete(callback)
  }

  /**
   * Set the current operation source
   */
  setSource(source: LayoutSource): void {
    this.currentSource = source
  }

  /**
   * Set the current actor (for CRDT)
   */
  setActor(actor: string): void {
    this.currentActor = actor
  }

  /**
   * Get the current operation source
   */
  getCurrentSource(): LayoutSource {
    return this.currentSource
  }

  /**
   * Get the current actor
   */
  getCurrentActor(): string {
    return this.currentActor
  }

  /**
   * Clean up refs and triggers for a node when its Vue component unmounts.
   * This should be called from the component's onUnmounted hook.
   */
  cleanupNodeRef(nodeId: NodeId): void {
    this.nodeRefs.delete(nodeId)
    this.nodeTriggers.delete(nodeId)
  }

  /**
   * Initialize store with existing nodes
   */
  initializeFromLiteGraph(
    nodes: Array<{ id: string; pos: [number, number]; size: [number, number] }>
  ): void {
    this.ydoc.transact(() => {
      this.ynodes.clear()
      // Note: We intentionally do NOT clear nodeRefs and nodeTriggers here.
      // Vue components may already hold references to these refs, and clearing
      // them would break the reactivity chain. The refs will be reused when
      // nodes are recreated, and stale refs will be cleaned up over time.
      this.spatialIndex.clear()
      this.linkSegmentSpatialIndex.clear()
      this.slotSpatialIndex.clear()
      this.rerouteSpatialIndex.clear()
      this.linkLayouts.clear()
      this.linkSegmentLayouts.clear()
      this.slotLayouts.clear()
      this.rerouteLayouts.clear()

      nodes.forEach((node, index) => {
        const layout: NodeLayout = {
          id: node.id.toString(),
          position: { x: node.pos[0], y: node.pos[1] },
          size: { width: node.size[0], height: node.size[1] },
          zIndex: index,
          visible: true,
          bounds: {
            x: node.pos[0],
            y: node.pos[1],
            width: node.size[0],
            height: node.size[1]
          }
        }

        this.ynodes.set(layout.id, layoutToYNode(layout))

        // Add to spatial index
        this.spatialIndex.insert(layout.id, layout.bounds)
      })

      // Trigger all existing refs to notify Vue of the new data
      this.nodeTriggers.forEach((trigger) => trigger())
    }, 'initialization')
  }

  // Operation handlers
  private handleMoveNode(
    operation: MoveNodeOperation,
    change: LayoutChange
  ): void {
    const ynode = this.ynodes.get(operation.nodeId)
    if (!ynode) {
      return
    }

    const size = yNodeToLayout(ynode).size
    const newBounds = {
      x: operation.position.x,
      y: operation.position.y,
      width: size.width,
      height: size.height
    }

    // Update spatial index FIRST, synchronously to prevent race conditions
    // Hit detection queries can run before CRDT updates complete
    this.spatialIndex.update(operation.nodeId, newBounds)

    // Then update CRDT
    ynode.set('position', operation.position)
    this.updateNodeBounds(ynode, operation.position, size)

    change.nodeIds.push(operation.nodeId)
  }

  private handleResizeNode(
    operation: ResizeNodeOperation,
    change: LayoutChange
  ): void {
    const ynode = this.ynodes.get(operation.nodeId)
    if (!ynode) return

    const position = yNodeToLayout(ynode).position
    const newBounds = {
      x: position.x,
      y: position.y,
      width: operation.size.width,
      height: operation.size.height
    }

    // Update spatial index FIRST, synchronously to prevent race conditions
    // Hit detection queries can run before CRDT updates complete
    this.spatialIndex.update(operation.nodeId, newBounds)

    // Then update CRDT
    ynode.set('size', operation.size)
    this.updateNodeBounds(ynode, position, operation.size)

    change.nodeIds.push(operation.nodeId)
  }

  private handleSetNodeZIndex(
    operation: SetNodeZIndexOperation,
    change: LayoutChange
  ): void {
    const ynode = this.ynodes.get(operation.nodeId)
    if (!ynode) return

    ynode.set('zIndex', operation.zIndex)
    change.nodeIds.push(operation.nodeId)
  }

  private handleCreateNode(
    operation: CreateNodeOperation,
    change: LayoutChange
  ): void {
    const ynode = layoutToYNode(operation.layout)
    this.ynodes.set(operation.nodeId, ynode)

    // Add to spatial index
    this.spatialIndex.insert(operation.nodeId, operation.layout.bounds)

    change.type = 'create'
    change.nodeIds.push(operation.nodeId)
  }

  private handleDeleteNode(
    operation: DeleteNodeOperation,
    change: LayoutChange
  ): void {
    if (!this.ynodes.has(operation.nodeId)) return

    this.ynodes.delete(operation.nodeId)
    // Note: We intentionally do NOT delete nodeRefs and nodeTriggers here.
    // During undo/redo, Vue components may still hold references to the old ref.
    // If we delete the trigger, Vue won't be notified when the node is re-created.
    // The trigger will be called in finalizeOperation to notify Vue of the change.
    // We also intentionally do NOT delete slot layouts here for the same reason,
    // and cleanup is handled by onUnmounted in useSlotElementTracking.
    // Remove from spatial index
    this.spatialIndex.remove(operation.nodeId)
    // Clean up associated links
    const linksToDelete = this.findLinksConnectedToNode(operation.nodeId)

    // Delete the associated links
    for (const linkId of linksToDelete) {
      this.ylinks.delete(String(linkId))
      this.linkLayouts.delete(linkId)

      // Clean up link segment layouts
      this.cleanupLinkSegments(linkId)
    }

    change.type = 'delete'
    change.nodeIds.push(operation.nodeId)
  }

  private handleBatchUpdateBounds(
    operation: BatchUpdateBoundsOperation,
    change: LayoutChange
  ): void {
    const spatialUpdates: Array<{ nodeId: NodeId; bounds: Bounds }> = []

    for (const nodeId of operation.nodeIds) {
      const data = operation.bounds[nodeId]
      const ynode = this.ynodes.get(nodeId)
      if (!ynode || !data) continue

      ynode.set('position', { x: data.bounds.x, y: data.bounds.y })
      ynode.set('size', {
        width: data.bounds.width,
        height: data.bounds.height
      })
      ynode.set('bounds', data.bounds)

      spatialUpdates.push({ nodeId, bounds: data.bounds })
      change.nodeIds.push(nodeId)
    }

    // Batch update spatial index for better performance
    if (spatialUpdates.length > 0) {
      this.spatialIndex.batchUpdate(spatialUpdates)
    }

    if (change.nodeIds.length) {
      change.type = 'update'
    }
  }

  private handleCreateLink(
    operation: CreateLinkOperation,
    change: LayoutChange
  ): void {
    const linkData = new Y.Map<unknown>()
    linkData.set('id', operation.linkId)
    linkData.set('sourceNodeId', operation.sourceNodeId)
    linkData.set('sourceSlot', operation.sourceSlot)
    linkData.set('targetNodeId', operation.targetNodeId)
    linkData.set('targetSlot', operation.targetSlot)

    this.ylinks.set(String(operation.linkId), linkData)

    // Link geometry will be computed separately when nodes move
    // This just tracks that the link exists
    change.type = 'create'
  }

  private handleDeleteLink(
    operation: DeleteLinkOperation,
    change: LayoutChange
  ): void {
    if (!this.ylinks.has(String(operation.linkId))) return

    this.ylinks.delete(String(operation.linkId))
    this.linkLayouts.delete(operation.linkId)
    // Clean up any segment layouts for this link
    this.cleanupLinkSegments(operation.linkId)

    change.type = 'delete'
  }

  private handleCreateReroute(
    operation: CreateRerouteOperation,
    change: LayoutChange
  ): void {
    const rerouteData = new Y.Map<unknown>()
    rerouteData.set('id', operation.rerouteId)
    rerouteData.set('position', operation.position)
    rerouteData.set('parentId', operation.parentId)
    rerouteData.set('linkIds', operation.linkIds)

    this.yreroutes.set(String(operation.rerouteId), rerouteData) // Yjs Map keys must be strings

    // The observer will automatically update the spatial index
    change.type = 'create'
  }

  private handleDeleteReroute(
    operation: DeleteRerouteOperation,
    change: LayoutChange
  ): void {
    if (!this.yreroutes.has(String(operation.rerouteId))) return // Yjs Map keys are strings

    this.yreroutes.delete(String(operation.rerouteId)) // Yjs Map keys are strings
    this.rerouteLayouts.delete(operation.rerouteId) // Layout map uses numeric ID
    this.rerouteSpatialIndex.remove(String(operation.rerouteId)) // Spatial index uses strings

    change.type = 'delete'
  }

  private handleMoveReroute(
    operation: MoveRerouteOperation,
    change: LayoutChange
  ): void {
    const yreroute = this.yreroutes.get(String(operation.rerouteId)) // Yjs Map keys are strings
    if (!yreroute) return

    yreroute.set('position', operation.position)

    const pos = operation.position
    const layout: RerouteLayout = {
      id: operation.rerouteId,
      position: pos,
      radius: 8,
      bounds: {
        x: pos.x - 8,
        y: pos.y - 8,
        width: 16,
        height: 16
      }
    }
    this.updateRerouteLayout(operation.rerouteId, layout)

    // Mark as update for listeners
    change.type = 'update'
  }

  /**
   * Update node bounds helper
   */
  private updateNodeBounds(
    ynode: NodeLayoutMap,
    position: Point,
    size: { width: number; height: number }
  ): void {
    ynode.set('bounds', {
      x: position.x,
      y: position.y,
      width: size.width,
      height: size.height
    })
  }

  /**
   * Find all links connected to a specific node
   */
  private findLinksConnectedToNode(nodeId: NodeId): LinkId[] {
    const connectedLinks: LinkId[] = []
    this.ylinks.forEach((linkData: Y.Map<unknown>, linkIdStr: string) => {
      const linkId = asLinkId(linkIdStr)
      const sourceNodeId = this.getLinkField(linkData, 'sourceNodeId')
      const targetNodeId = this.getLinkField(linkData, 'targetNodeId')

      if (sourceNodeId === nodeId || targetNodeId === nodeId) {
        connectedLinks.push(linkId)
      }
    })
    return connectedLinks
  }

  /**
   * Handle link change events
   */
  private handleLinkChange(change: YEventChange, linkIdStr: string): void {
    if (change.action === 'delete') {
      const linkId = asLinkId(linkIdStr)
      this.cleanupLinkData(linkId)
    }
    // Link was added or updated - geometry will be computed separately
    // This just tracks that the link exists in CRDT
  }

  /**
   * Clean up all data associated with a link
   */
  private cleanupLinkData(linkId: LinkId): void {
    this.linkLayouts.delete(linkId)
    this.cleanupLinkSegments(linkId)
  }

  /**
   * Clean up all segment layouts for a link
   */
  private cleanupLinkSegments(linkId: LinkId): void {
    const keysToDelete: string[] = []
    for (const [key] of this.linkSegmentLayouts) {
      if (key.startsWith(`${linkId}:`)) {
        keysToDelete.push(key)
      }
    }

    for (const key of keysToDelete) {
      this.linkSegmentLayouts.delete(key)
      this.linkSegmentSpatialIndex.remove(key)
    }
  }

  /**
   * Handle reroute change events
   */
  private handleRerouteChange(
    change: YEventChange,
    rerouteIdStr: string
  ): void {
    const rerouteId = asRerouteId(rerouteIdStr)

    if (change.action === 'delete') {
      this.handleRerouteDelete(rerouteId)
    } else {
      this.handleRerouteUpsert(rerouteId)
    }
  }

  /**
   * Handle reroute deletion
   */
  private handleRerouteDelete(rerouteId: RerouteId): void {
    this.rerouteLayouts.delete(rerouteId)
    this.rerouteSpatialIndex.remove(String(rerouteId))
  }

  /**
   * Handle reroute upsert (update if exists, create if not)
   */
  private handleRerouteUpsert(rerouteId: RerouteId): void {
    const rerouteData = this.yreroutes.get(String(rerouteId))
    if (!rerouteData) return

    const position = this.getRerouteField(rerouteData, 'position')
    if (!position) return

    const layout = this.createRerouteLayout(rerouteId, position)
    this.updateRerouteLayout(rerouteId, layout)
  }

  /**
   * Create reroute layout from position
   */
  private createRerouteLayout(
    rerouteId: RerouteId,
    position: Point
  ): RerouteLayout {
    return {
      id: rerouteId,
      position,
      radius: REROUTE_RADIUS,
      bounds: {
        x: position.x - REROUTE_RADIUS,
        y: position.y - REROUTE_RADIUS,
        width: REROUTE_RADIUS * 2,
        height: REROUTE_RADIUS * 2
      }
    }
  }

  // Helper methods

  private notifyChange(change: LayoutChange): void {
    this.changeListeners.forEach((listener) => {
      try {
        listener(change)
      } catch (error) {
        console.error('Error in layout change listener:', error)
      }
    })
  }

  // CRDT-specific methods
  getOperationsSince(timestamp: number): LayoutOperation[] {
    const operations: LayoutOperation[] = []
    this.yoperations.forEach((op: LayoutOperation) => {
      if (op && op.timestamp > timestamp) {
        operations.push(op)
      }
    })
    return operations
  }

  getOperationsByActor(actor: string): LayoutOperation[] {
    const operations: LayoutOperation[] = []
    this.yoperations.forEach((op: LayoutOperation) => {
      if (op && op.actor === actor) {
        operations.push(op)
      }
    })
    return operations
  }

  /**
   * Get the Yjs document for network sync (future feature)
   */
  getYDoc(): Y.Doc {
    return this.ydoc
  }

  /**
   * Apply updates from remote peers (future feature)
   */
  applyUpdate(update: Uint8Array): void {
    Y.applyUpdate(this.ydoc, update)
  }

  /**
   * Get state as update for sending to peers (future feature)
   */
  getStateAsUpdate(): Uint8Array {
    return Y.encodeStateAsUpdate(this.ydoc)
  }

  /**
   * Batch update node bounds using Yjs transaction for atomicity.
   */
  batchUpdateNodeBounds(updates: NodeBoundsUpdate[]): void {
    if (updates.length === 0) return

    const originalSource = this.currentSource
    const shouldNormalizeHeights = originalSource === LayoutSource.DOM
    this.currentSource = LayoutSource.Vue

    const nodeIds: NodeId[] = []
    const boundsRecord: BatchUpdateBoundsOperation['bounds'] = {}

    for (const { nodeId, bounds } of updates) {
      const ynode = this.ynodes.get(nodeId)
      if (!ynode) continue
      const currentLayout = yNodeToLayout(ynode)

      const normalizedBounds = shouldNormalizeHeights
        ? {
            ...bounds,
            height: removeNodeTitleHeight(bounds.height)
          }
        : bounds

      boundsRecord[nodeId] = {
        bounds: normalizedBounds,
        previousBounds: currentLayout.bounds
      }
      nodeIds.push(nodeId)
    }

    if (!nodeIds.length) {
      this.currentSource = originalSource
      return
    }

    const operation: BatchUpdateBoundsOperation = {
      type: 'batchUpdateBounds',
      entity: 'node',
      nodeIds,
      bounds: boundsRecord,
      timestamp: Date.now(),
      source: this.currentSource,
      actor: this.currentActor
    }

    this.applyOperation(operation)

    this.currentSource = originalSource
  }
}

// Create singleton instance
export const layoutStore = new LayoutStoreImpl()
