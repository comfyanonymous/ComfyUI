/**
 * Layout System - Type Definitions
 *
 * This file contains all type definitions for the layout system
 * that manages node positions, bounds, spatial data, and operations.
 */
import type { ComputedRef, Ref } from 'vue'

// Enum for layout source types
export enum LayoutSource {
  Canvas = 'canvas',
  Vue = 'vue',
  DOM = 'dom',
  External = 'external'
}

// Basic geometric types
export interface Point {
  x: number
  y: number
}

export interface Size {
  width: number
  height: number
}

export interface Bounds {
  x: number
  y: number
  width: number
  height: number
}

export interface NodeBoundsUpdate {
  nodeId: NodeId
  bounds: Bounds
}

export type NodeId = string
export type LinkId = number
export type RerouteId = number

// Layout data structures
export interface NodeLayout {
  id: NodeId
  position: Point
  size: Size
  zIndex: number
  visible: boolean
  // Computed bounds for hit testing
  bounds: Bounds
}

export interface SlotLayout {
  nodeId: NodeId
  index: number
  type: 'input' | 'output'
  position: Point
  bounds: Bounds
}

export interface LinkLayout {
  id: LinkId
  path: Path2D
  bounds: Bounds
  centerPos: Point
  sourceNodeId: NodeId
  targetNodeId: NodeId
  sourceSlot: number
  targetSlot: number
}

// Layout for individual link segments (for precise hit-testing)
export interface LinkSegmentLayout {
  linkId: LinkId
  rerouteId: RerouteId | null // null for final segment to target
  path: Path2D
  bounds: Bounds
  centerPos: Point
}

export interface RerouteLayout {
  id: RerouteId
  position: Point
  radius: number
  bounds: Bounds
}

/**
 * Meta-only base for all operations - contains common fields
 */
interface OperationMeta {
  /** Unique operation ID for deduplication */
  id?: string
  /** Timestamp for ordering operations */
  timestamp: number
  /** Actor who performed the operation (for CRDT) */
  actor: string
  /** Source system that initiated the operation */
  source: LayoutSource
  /** Operation type discriminator */
  type: OperationType
}

/**
 * Entity-specific base types for proper type discrimination
 */
type NodeOpBase = OperationMeta & { entity: 'node'; nodeId: NodeId }
type LinkOpBase = OperationMeta & { entity: 'link'; linkId: LinkId }
type RerouteOpBase = OperationMeta & {
  entity: 'reroute'
  rerouteId: RerouteId
}

/**
 * Operation type discriminator for type narrowing
 */
type OperationType =
  | 'moveNode'
  | 'resizeNode'
  | 'setNodeZIndex'
  | 'createNode'
  | 'deleteNode'
  | 'setNodeVisibility'
  | 'batchUpdateBounds'
  | 'createLink'
  | 'deleteLink'
  | 'createReroute'
  | 'deleteReroute'
  | 'moveReroute'

/**
 * Move node operation
 */
export interface MoveNodeOperation extends NodeOpBase {
  type: 'moveNode'
  position: Point
  previousPosition: Point
}

/**
 * Resize node operation
 */
export interface ResizeNodeOperation extends NodeOpBase {
  type: 'resizeNode'
  size: { width: number; height: number }
  previousSize: { width: number; height: number }
}

/**
 * Set node z-index operation
 */
export interface SetNodeZIndexOperation extends NodeOpBase {
  type: 'setNodeZIndex'
  zIndex: number
  previousZIndex: number
}

/**
 * Create node operation
 */
export interface CreateNodeOperation extends NodeOpBase {
  type: 'createNode'
  layout: NodeLayout
}

/**
 * Delete node operation
 */
export interface DeleteNodeOperation extends NodeOpBase {
  type: 'deleteNode'
  previousLayout: NodeLayout
}

/**
 * Set node visibility operation
 */
interface SetNodeVisibilityOperation extends NodeOpBase {
  type: 'setNodeVisibility'
  visible: boolean
  previousVisible: boolean
}

/**
 * Batch update operation for atomic multi-property changes
 */
export interface BatchUpdateBoundsOperation extends OperationMeta {
  entity: 'node'
  type: 'batchUpdateBounds'
  nodeIds: NodeId[]
  bounds: Record<NodeId, { bounds: Bounds; previousBounds: Bounds }>
}

/**
 * Create link operation
 */
export interface CreateLinkOperation extends LinkOpBase {
  type: 'createLink'
  sourceNodeId: NodeId
  sourceSlot: number
  targetNodeId: NodeId
  targetSlot: number
}

/**
 * Delete link operation
 */
export interface DeleteLinkOperation extends LinkOpBase {
  type: 'deleteLink'
}

/**
 * Create reroute operation
 */
export interface CreateRerouteOperation extends RerouteOpBase {
  type: 'createReroute'
  position: Point
  parentId?: RerouteId
  linkIds: LinkId[]
}

/**
 * Delete reroute operation
 */
export interface DeleteRerouteOperation extends RerouteOpBase {
  type: 'deleteReroute'
}

/**
 * Move reroute operation
 */
export interface MoveRerouteOperation extends RerouteOpBase {
  type: 'moveReroute'
  position: Point
  previousPosition: Point
}

/**
 * Union of all operation types
 */
export type LayoutOperation =
  | MoveNodeOperation
  | ResizeNodeOperation
  | SetNodeZIndexOperation
  | CreateNodeOperation
  | DeleteNodeOperation
  | SetNodeVisibilityOperation
  | BatchUpdateBoundsOperation
  | CreateLinkOperation
  | DeleteLinkOperation
  | CreateRerouteOperation
  | DeleteRerouteOperation
  | MoveRerouteOperation

export interface LayoutChange {
  type: 'create' | 'update' | 'delete'
  nodeIds: NodeId[]
  timestamp: number
  source: LayoutSource
  operation: LayoutOperation
}

// Store interfaces
export interface LayoutStore {
  // CustomRef accessors for shared write access
  getNodeLayoutRef(nodeId: NodeId): Ref<NodeLayout | null>
  getNodesInBounds(bounds: Bounds): ComputedRef<NodeId[]>
  getAllNodes(): ComputedRef<ReadonlyMap<NodeId, NodeLayout>>
  getVersion(): ComputedRef<number>

  // Spatial queries (non-reactive)
  queryNodeAtPoint(point: Point): NodeId | null
  queryNodesInBounds(bounds: Bounds): NodeId[]

  // Hit testing queries for links, slots, and reroutes
  queryLinkAtPoint(point: Point, ctx?: CanvasRenderingContext2D): LinkId | null
  queryLinkSegmentAtPoint(
    point: Point,
    ctx?: CanvasRenderingContext2D
  ): { linkId: LinkId; rerouteId: RerouteId | null } | null
  querySlotAtPoint(point: Point): SlotLayout | null
  queryRerouteAtPoint(point: Point): RerouteLayout | null
  queryItemsInBounds(bounds: Bounds): {
    nodes: NodeId[]
    links: LinkId[]
    slots: string[]
    reroutes: RerouteId[]
  }

  // Update methods for link, slot, and reroute layouts
  updateLinkLayout(linkId: LinkId, layout: LinkLayout): void
  updateLinkSegmentLayout(
    linkId: LinkId,
    rerouteId: RerouteId | null,
    layout: Omit<LinkSegmentLayout, 'linkId' | 'rerouteId'>
  ): void
  updateSlotLayout(key: string, layout: SlotLayout): void
  updateRerouteLayout(rerouteId: RerouteId, layout: RerouteLayout): void

  // Delete methods for cleanup
  deleteLinkLayout(linkId: LinkId): void
  deleteLinkSegmentLayout(linkId: LinkId, rerouteId: RerouteId | null): void
  deleteSlotLayout(key: string): void
  deleteRerouteLayout(rerouteId: RerouteId): void
  clearAllSlotLayouts(): void

  // Get layout data
  getLinkLayout(linkId: LinkId): LinkLayout | null
  getSlotLayout(key: string): SlotLayout | null
  getRerouteLayout(rerouteId: RerouteId): RerouteLayout | null

  // Returns all slot layout keys currently tracked by the store
  getAllSlotKeys(): string[]

  // Direct mutation API (CRDT-ready)
  applyOperation(operation: LayoutOperation): void

  // Change subscription
  onChange(callback: (change: LayoutChange) => void): () => void

  // Initialization
  initializeFromLiteGraph(
    nodes: Array<{ id: string; pos: [number, number]; size: [number, number] }>
  ): void

  // Source and actor management
  setSource(source: LayoutSource): void
  setActor(actor: string): void
  getCurrentSource(): LayoutSource
  getCurrentActor(): string

  // Batch updates
  batchUpdateNodeBounds(
    updates: Array<{ nodeId: NodeId; bounds: Bounds }>
  ): void

  batchUpdateSlotLayouts(
    updates: Array<{ key: string; layout: SlotLayout }>
  ): void
}
