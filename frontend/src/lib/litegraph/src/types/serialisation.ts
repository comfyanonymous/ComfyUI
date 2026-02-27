import type { UUID } from '@/lib/litegraph/src/utils/uuid'

import type { LGraphConfig, LGraphExtra, LGraphState } from '../LGraph'
import type { IGraphGroupFlags } from '../LGraphGroup'
import type { NodeId, NodeProperty } from '../LGraphNode'
import type { LinkId, SerialisedLLinkArray } from '../LLink'
import type { FloatingRerouteSlot, RerouteId } from '../Reroute'
import type {
  Dictionary,
  INodeFlags,
  INodeInputSlot,
  INodeOutputSlot,
  INodeSlot,
  ISlotType,
  Point,
  Size
} from '../interfaces'
import type { LiteGraph } from '../litegraph'
import type { RenderShape } from './globalEnums'
import type { TWidgetValue } from './widgets'

/**
 * An object that implements custom pre-serialization logic via {@link Serialisable.asSerialisable}.
 */
export interface Serialisable<SerialisableObject> {
  /**
   * Prepares this object for serialization.
   * Creates a partial shallow copy of itself, with only the properties that should be serialised.
   * @returns An object that can immediately be serialized to JSON.
   */
  asSerialisable(): SerialisableObject
}

interface BaseExportedGraph {
  /** Unique graph ID.  Automatically generated if not provided. */
  id: UUID
  /** The revision number of this graph. Not automatically incremented; intended for use by a downstream save function. */
  revision: number
  config?: LGraphConfig
  /** Details of the appearance and location of subgraphs shown in this graph. Similar to */
  subgraphs?: ExportedSubgraphInstance[]
  /** Definitions of re-usable objects that are referenced elsewhere in this exported graph. */
  definitions?: {
    /** The base definition of subgraphs used in this workflow. That is, what you see when you open / edit a subgraph. */
    subgraphs?: ExportedSubgraph[]
  }
}

export interface SerialisableGraph extends BaseExportedGraph {
  /** Schema version.  @remarks Version bump should add to const union, which is used to narrow type during deserialise. */
  version: 0 | 1
  state: LGraphState
  groups?: ISerialisedGroup[]
  nodes?: ISerialisedNode[]
  links?: SerialisableLLink[]
  floatingLinks?: SerialisableLLink[]
  reroutes?: SerialisableReroute[]
  extra?: LGraphExtra
}

export type ISerialisableNodeInput = Omit<
  INodeInputSlot,
  'boundingRect' | 'widget'
> & {
  widget?: { name: string }
}
export type ISerialisableNodeOutput = Omit<
  INodeOutputSlot,
  'boundingRect' | '_data'
> & {
  widget?: { name: string }
}

/** Serialised LGraphNode */
export interface ISerialisedNode {
  title?: string
  id: NodeId
  type: string
  pos: Point
  size: Size
  flags: INodeFlags
  order: number
  mode: number
  outputs?: ISerialisableNodeOutput[]
  inputs?: ISerialisableNodeInput[]
  properties?: Dictionary<NodeProperty | undefined>
  shape?: RenderShape
  boxcolor?: string
  color?: string
  bgcolor?: string
  showAdvanced?: boolean
  /**
   * Note: Some custom nodes overrides the `widgets_values` property to an
   * object that has `length` property and index access. It is not safe to call
   * any array methods on it.
   * See example in https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/8629188458dc6cb832f871ece3bd273507e8a766/web/js/VHS.core.js#L59-L84
   */
  widgets_values?: TWidgetValue[]
}

/** Properties of nodes that are used by subgraph instances. */
type NodeSubgraphSharedProps = Omit<
  ISerialisedNode,
  'properties' | 'showAdvanced'
>

/** A single instance of a subgraph; where it is used on a graph, any customisation to shape / colour etc. */
export interface ExportedSubgraphInstance extends NodeSubgraphSharedProps {
  /**
   * The ID of the actual subgraph definition.
   * @see {@link ExportedSubgraph.subgraphs}
   */
  type: UUID
  /** Custom properties for this subgraph instance */
  properties?: Dictionary<NodeProperty | undefined>
}

/**
 * Original implementation from static litegraph.d.ts
 * Maintained for backwards compat
 */
export interface ISerialisedGraph extends BaseExportedGraph {
  last_node_id: NodeId
  last_link_id: number
  nodes: ISerialisedNode[]
  links: SerialisedLLinkArray[]
  floatingLinks?: SerialisableLLink[]
  groups: ISerialisedGroup[]
  version: typeof LiteGraph.VERSION
  extra?: LGraphExtra
}

/**
 * Defines a subgraph and its contents.
 * Can be referenced multiple times in a schema.
 */
export interface ExportedSubgraph extends SerialisableGraph {
  /** The display name of the subgraph. */
  name: string
  /** Optional category for organizing subgraph blueprints in the node library. */
  category?: string
  /** Optional description shown as tooltip when hovering over the subgraph node. */
  description?: string
  inputNode: ExportedSubgraphIONode
  outputNode: ExportedSubgraphIONode
  /** Ordered list of inputs to the subgraph itself. Similar to a reroute, with the input side in the graph, and the output side in the subgraph. */
  inputs?: SubgraphIO[]
  /** Ordered list of outputs from the subgraph itself. Similar to a reroute, with the input side in the subgraph, and the output side in the graph. */
  outputs?: SubgraphIO[]
  /** A list of node widgets displayed in the parent graph, on the subgraph object. */
  widgets?: ExposedWidget[]
}

/** Properties shared by subgraph and node I/O slots. */
type SubgraphIOShared = Omit<
  INodeSlot,
  'boundingRect' | 'nameLocked' | 'locked' | 'removable' | '_floatingLinks'
>

/** Subgraph I/O slots */
export interface SubgraphIO extends SubgraphIOShared {
  /** Slot ID (internal; never changes once instantiated). */
  id: UUID
  /** The data type this slot uses. Unlike nodes, this does not support legacy numeric types. */
  type: string
  /** Links connected to this slot, or `undefined` if not connected. An output slot should only ever have one link. */
  linkIds?: LinkId[]
}

/** A reference to a node widget shown in the parent graph */
export interface ExposedWidget {
  /** The ID of the node (inside the subgraph) that the widget belongs to. */
  id: NodeId
  /** The name of the widget to show in the parent graph. */
  name: string
}

/** Serialised LGraphGroup */
export interface ISerialisedGroup {
  id: number
  title: string
  bounding: number[]
  color?: string
  font_size?: number
  flags?: IGraphGroupFlags
}

/** Items copied from the canvas */
export interface ClipboardItems {
  nodes?: ISerialisedNode[]
  groups?: ISerialisedGroup[]
  reroutes?: SerialisableReroute[]
  links?: SerialisableLLink[]
  subgraphs?: ExportedSubgraph[]
}

export interface SerialisableReroute {
  id: RerouteId
  parentId?: RerouteId
  pos: Point
  linkIds: LinkId[]
  floating?: FloatingRerouteSlot
}

export interface SerialisableLLink {
  /** Link ID */
  id: LinkId
  /** Output node ID */
  origin_id: NodeId
  /** Output slot index */
  origin_slot: number
  /** Input node ID */
  target_id: NodeId
  /** Input slot index */
  target_slot: number
  /** Data type of the link */
  type: ISlotType
  /** ID of the last reroute (from input to output) that this link passes through, otherwise `undefined` */
  parentId?: RerouteId
}

export interface ExportedSubgraphIONode {
  id: NodeId
  bounding: [number, number, number, number]
  pinned?: boolean
}
