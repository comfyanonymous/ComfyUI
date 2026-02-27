import type { ContextMenu } from './ContextMenu'
import type { LGraphNode } from './LGraphNode'
import { LiteGraphGlobal } from './LiteGraphGlobal'
import type {
  ConnectingLink,
  IContextMenuOptions,
  Point,
  Size
} from './interfaces'
import { loadPolyfills } from './polyfills'
import type { CanvasEventDetail } from './types/events'
import type { RenderShape, TitleMode } from './types/globalEnums'

// Must remain above LiteGraphGlobal (circular dependency due to abstract factory behaviour in `configure`)
export { Subgraph } from './subgraph/Subgraph'

export const LiteGraph = new LiteGraphGlobal()

// Load legacy polyfills
loadPolyfills()

// Backwards compat

// Type definitions for litegraph.js 0.7.0
// Project: litegraph.js
// Definitions by: NateScarlet <https://github.com/NateScarlet>
/** @deprecated Use {@link Point} instead. */
export type Vector2 = Point

interface IContextMenuItem {
  content: string
  callback?: ContextMenuEventListener
  /** Used as innerHTML for extra child element */
  title?: string
  disabled?: boolean
  has_submenu?: boolean
  submenu?: {
    options: IContextMenuItem[]
  } & IContextMenuOptions
  className?: string
}

type ContextMenuEventListener = (
  value: IContextMenuItem,
  options: IContextMenuOptions,
  event: MouseEvent,
  parentMenu: ContextMenu<unknown> | undefined,
  node: LGraphNode
) => boolean | void

export interface LinkReleaseContextExtended {
  links: ConnectingLink[]
}

export interface LiteGraphCanvasEvent extends CustomEvent<CanvasEventDetail> {}

export interface LGraphNodeConstructor<T extends LGraphNode = LGraphNode> {
  new (title: string, type?: string): T

  title: string
  type?: string // TODO: to be, or not to be--that is the question
  size?: Size
  min_height?: number
  slot_start_y?: number
  widgets_info?: Record<string, unknown>
  collapsable?: boolean
  color?: string
  bgcolor?: string
  shape?: RenderShape
  title_mode?: TitleMode
  title_color?: string
  title_text_color?: string
  keepAllLinksOnBypass: boolean
  resizeHandleSize?: number
  resizeEdgeSize?: number
}

// End backwards compat

export { LinkConnector } from './canvas/LinkConnector'
export { isOverNodeInput, isOverNodeOutput } from './canvas/measureSlots'
export { CanvasPointer } from './CanvasPointer'
export * as Constants from './constants'
export { SUBGRAPH_INPUT_ID } from './constants'
export { ContextMenu } from './ContextMenu'

export { DragAndScale } from './DragAndScale'

export { Rectangle } from './infrastructure/Rectangle'
export { RecursionError } from './infrastructure/RecursionError'
export type {
  CanvasColour,
  ColorOption,
  IContextMenuOptions,
  IContextMenuValue,
  INodeInputSlot,
  INodeOutputSlot,
  INodeSlot,
  ISlotType,
  LinkNetwork,
  Point,
  Positionable,
  Size
} from './interfaces'
export {
  LGraph,
  type GroupNodeConfigEntry,
  type GroupNodeWorkflowData,
  type LGraphTriggerAction,
  type LGraphTriggerParam,
  type GraphAddOptions
} from './LGraph'
export type { LGraphTriggerEvent } from './types/graphTriggers'
export { BadgePosition, LGraphBadge } from './LGraphBadge'
export { LGraphCanvas } from './LGraphCanvas'
export { LGraphGroup } from './LGraphGroup'
export { LGraphNode, type NodeId } from './LGraphNode'
export { LLink } from './LLink'
export { createBounds } from './measure'
export { Reroute, type RerouteId } from './Reroute'
export {
  type ExecutableLGraphNode,
  ExecutableNodeDTO,
  type ExecutionId
} from './subgraph/ExecutableNodeDTO'
export { SubgraphNode } from './subgraph/SubgraphNode'
export type { CanvasPointerEvent } from './types/events'
export {
  EaseFunction,
  LGraphEventMode,
  LinkDirection,
  LinkMarkerShape,
  RenderShape
} from './types/globalEnums'
export type {
  ExportedSubgraph,
  ExportedSubgraphInstance,
  ISerialisedGraph,
  ISerialisedNode,
  SerialisableGraph
} from './types/serialisation'
export type { IWidget } from './types/widgets'
export { isColorable } from './utils/type'
export { createUuidv4 } from './utils/uuid'
export type { UUID } from './utils/uuid'
export { truncateText } from './utils/textUtils'
export { getWidgetStep, resolveNodeRootGraphId } from './utils/widget'
export { distributeSpace, type SpaceRequest } from './utils/spaceDistribution'

export { BaseWidget } from './widgets/BaseWidget'

export { LegacyWidget } from './widgets/LegacyWidget'

export { isComboWidget } from './widgets/widgetMap'
/** @knipIgnoreUnusedButUsedByCustomNodes */
export { isAssetWidget } from './widgets/widgetMap'
// Additional test-specific exports
export { LGraphButton } from './LGraphButton'
export { MovingOutputLink } from './canvas/MovingOutputLink'
export { ToOutputRenderLink } from './canvas/ToOutputRenderLink'
export { ToInputFromIoNodeLink } from './canvas/ToInputFromIoNodeLink'
export type { TWidgetType, TWidgetValue, IWidgetOptions } from './types/widgets'
export {
  findUsedSubgraphIds,
  getDirectSubgraphIds,
  isSubgraphInput,
  isSubgraphOutput
} from './subgraph/subgraphUtils'
export { NodeInputSlot } from './node/NodeInputSlot'
export { NodeOutputSlot } from './node/NodeOutputSlot'
export { inputAsSerialisable, outputAsSerialisable } from './node/slotUtils'
export { MovingInputLink } from './canvas/MovingInputLink'
export { ToInputRenderLink } from './canvas/ToInputRenderLink'
export { LiteGraphGlobal } from './LiteGraphGlobal'
