import { toValue } from 'vue'

import { LGraphNodeProperties } from '@/lib/litegraph/src/LGraphNodeProperties'
import {
  calculateInputSlotPosFromSlot,
  getSlotPosition
} from '@/renderer/core/canvas/litegraph/slotCalculations'
import type { SlotPositionContext } from '@/renderer/core/canvas/litegraph/slotCalculations'
import { useLayoutMutations } from '@/renderer/core/layout/operations/layoutMutations'
import { LayoutSource } from '@/renderer/core/layout/types'
import { adjustColor } from '@/utils/colorUtil'
import type { ColorAdjustOptions } from '@/utils/colorUtil'
import {
  commonType,
  isNodeBindable,
  toClass
} from '@/lib/litegraph/src/utils/type'

import { SUBGRAPH_OUTPUT_ID } from '@/lib/litegraph/src/constants'
import type { DragAndScale } from './DragAndScale'
import type { LGraph } from './LGraph'
import { BadgePosition, LGraphBadge } from './LGraphBadge'
import { LGraphButton } from './LGraphButton'
import type { LGraphButtonOptions } from './LGraphButton'
import { LGraphCanvas } from './LGraphCanvas'
import { LLink } from './LLink'
import type { Reroute, RerouteId } from './Reroute'
import { getNodeInputOnPos, getNodeOutputOnPos } from './canvas/measureSlots'
import type { IDrawBoundingOptions } from './draw'
import { NullGraphError } from './infrastructure/NullGraphError'
import type { ReadOnlyRectangle } from './infrastructure/Rectangle'
import { Rectangle } from './infrastructure/Rectangle'
import type {
  ColorOption,
  CompassCorners,
  DefaultConnectionColors,
  Dictionary,
  IColorable,
  IContextMenuValue,
  IFoundSlot,
  INodeFlags,
  INodeInputSlot,
  INodeOutputSlot,
  INodeSlot,
  INodeSlotContextItem,
  IPinnable,
  ISlotType,
  Panel,
  Point,
  Positionable,
  ReadOnlyRect,
  Rect,
  Size
} from './interfaces'
import { LiteGraph, Subgraph } from './litegraph'
import type { LGraphNodeConstructor, SubgraphNode } from './litegraph'
import {
  createBounds,
  isInRect,
  isInRectangle,
  isPointInRect,
  snapPoint
} from './measure'
import { NodeInputSlot } from './node/NodeInputSlot'
import { NodeOutputSlot } from './node/NodeOutputSlot'
import {
  inputAsSerialisable,
  isINodeInputSlot,
  isWidgetInputSlot,
  outputAsSerialisable
} from './node/slotUtils'
import type { SubgraphInputNode } from './subgraph/SubgraphInputNode'
import type { SubgraphOutputNode } from './subgraph/SubgraphOutputNode'
import type { NodeLike } from './types/NodeLike'
import type { CanvasPointerEvent } from './types/events'
import {
  LGraphEventMode,
  NodeSlotType,
  RenderShape,
  TitleMode
} from './types/globalEnums'
import type { ISerialisedNode, SubgraphIO } from './types/serialisation'
import type {
  IBaseWidget,
  IWidgetOptions,
  TWidgetType,
  TWidgetValue
} from './types/widgets'
import { findFreeSlotOfType } from './utils/collections'
import { warnDeprecated } from './utils/feedback'
import { distributeSpace } from './utils/spaceDistribution'
import { truncateText } from './utils/textUtils'
import { BaseWidget } from './widgets/BaseWidget'
import { toConcreteWidget } from './widgets/widgetMap'
import type { WidgetTypeMap } from './widgets/widgetMap'

// #region Types

export type NodeId = number | string

export type NodeProperty = string | number | boolean | object

interface INodePropertyInfo {
  name?: string
  type?: string
  default_value?: NodeProperty
  widget?: string
  label?: string
  values?: TWidgetValue[]
}

interface IMouseOverData {
  inputId?: number
  outputId?: number
  overWidget?: IBaseWidget
}

interface ConnectByTypeOptions {
  /** @deprecated Events */
  createEventInCase?: boolean
  /** Allow our wildcard slot to connect to typed slots on remote node. Default: true */
  wildcardToTyped?: boolean
  /** Allow our typed slot to connect to wildcard slots on remote node. Default: true */
  typedToWildcard?: boolean
  /** The {@link Reroute.id} that the connection is being dragged from. */
  afterRerouteId?: RerouteId
}

/** Internal type used for type safety when implementing generic checks for inputs & outputs */
interface IGenericLinkOrLinks {
  links?: INodeOutputSlot['links']
  link?: INodeInputSlot['link']
}

interface FindFreeSlotOptions {
  /** Slots matching these types will be ignored.  Default: [] */
  typesNotAccepted?: ISlotType[]
  /** If true, the slot itself is returned instead of the index.  Default: false */
  returnObj?: boolean
}

interface DrawSlotsOptions {
  fromSlot?: INodeInputSlot | INodeOutputSlot
  colorContext: DefaultConnectionColors
  editorAlpha: number
  lowQuality: boolean
}

interface DrawWidgetsOptions {
  lowQuality?: boolean
  editorAlpha?: number
}

interface DrawTitleOptions {
  scale: number
  title_height?: number
  low_quality?: boolean
}

interface DrawTitleTextOptions extends DrawTitleOptions {
  default_title_color: string
}

export interface DrawTitleBoxOptions extends DrawTitleOptions {
  box_size?: number
}

/*
title: string
pos: [x,y]
size: [x,y]

input|output: every connection
    +  { name:string, type:string, pos: [x,y]=Optional, direction: "input"|"output", links: Array });

general properties:
    + clip_area: if you render outside the node, it will be clipped
    + unsafe_execution: not allowed for safe execution
    + skip_repeated_outputs: when adding new outputs, it won't show if there is one already connected
    + resizable: if set to false it won't be resizable with the mouse
    + widgets_start_y: widgets start at y distance from the top of the node

flags object:
    + collapsed: if it is collapsed

supported callbacks:
    + onAdded: when added to graph (warning: this is called BEFORE the node is configured when loading)
    + onRemoved: when removed from graph
    + onStart: when the graph starts playing
    + onStop: when the graph stops playing
    + onDrawForeground: render the inside widgets inside the node
    + onDrawBackground: render the background area inside the node (only in edit mode)
    + onMouseDown
    + onMouseMove
    + onMouseUp
    + onMouseEnter
    + onMouseLeave
    + onExecute: execute the node
    + onPropertyChanged: when a property is changed in the panel (return true to skip default behaviour)
    + onGetInputs: returns an array of possible inputs
    + onGetOutputs: returns an array of possible outputs
    + onBounding: in case this node has a bigger bounding than the node itself (the callback receives the bounding as [x,y,w,h])
    + onDblClick: double clicked in the node
    + onNodeTitleDblClick: double clicked in the node title
    + onInputDblClick: input slot double clicked (can be used to automatically create a node connected)
    + onOutputDblClick: output slot double clicked (can be used to automatically create a node connected)
    + onConfigure: called after the node has been configured
    + onSerialize: to add extra info when serializing (the callback receives the object that should be filled with the data)
    + onSelected
    + onDeselected
    + onDropItem : DOM item dropped over the node
    + onDropFile : file dropped over the node
    + onConnectInput : if returns false the incoming connection will be canceled
    + onConnectionsChange : a connection changed (new one or removed) (NodeSlotType.INPUT or NodeSlotType.OUTPUT, slot, true if connected, link_info, input_info )
    + onAction: action slot triggered
    + getExtraMenuOptions: to add option to context menu
*/

export interface LGraphNode {
  constructor: LGraphNodeConstructor
}

// #endregion Types

/**
 * Base class for all nodes
 * @param title a name for the node
 * @param type a type for the node
 */

export class LGraphNode
  implements NodeLike, Positionable, IPinnable, IColorable
{
  // Static properties used by dynamic child classes
  static title?: string
  static MAX_CONSOLE?: number
  static type?: string
  static category?: string
  static description?: string
  static filter?: string
  static skip_list?: boolean
  static nodeData?: {
    dev_only?: boolean
    deprecated?: boolean
    experimental?: boolean
    output_node?: boolean
    api_node?: boolean
    name?: string
  }

  static resizeHandleSize = 15
  static resizeEdgeSize = 5

  /** Default setting for {@link LGraphNode.connectInputToOutput}. @see {@link INodeFlags.keepAllLinksOnBypass} */
  static keepAllLinksOnBypass: boolean = false

  /** The title text of the node. */
  title: string
  /**
   * The font style used to render the node's title text.
   */
  get titleFontStyle(): string {
    return `${LiteGraph.NODE_TEXT_SIZE}px ${LiteGraph.NODE_FONT}`
  }

  get innerFontStyle(): string {
    return `normal ${LiteGraph.NODE_SUBTEXT_SIZE}px ${LiteGraph.NODE_FONT}`
  }

  get displayType(): string {
    return this.type
  }

  graph: LGraph | Subgraph | null = null
  id: NodeId
  type: string = ''
  inputs: INodeInputSlot[] = []
  outputs: INodeOutputSlot[] = []

  private _concreteInputs: NodeInputSlot[] = []
  private _concreteOutputs: NodeOutputSlot[] = []

  properties: Dictionary<NodeProperty | undefined> = {}
  properties_info: INodePropertyInfo[] = []
  flags: INodeFlags = {}
  widgets?: IBaseWidget[]

  /** Property manager for this node */
  changeTracker: LGraphNodeProperties

  /**
   * The amount of space available for widgets to grow into.
   * @see {@link layoutWidgets}
   */
  freeWidgetSpace?: number

  locked?: boolean

  /** Execution order, automatically computed during run @see {@link LGraph.computeExecutionOrder} */
  order: number = 0
  mode: LGraphEventMode = LGraphEventMode.ALWAYS
  last_serialization?: ISerialisedNode
  serialize_widgets?: boolean
  /**
   * The overridden fg color used to render the node.
   * @see {@link renderingColor}
   */
  color?: string
  /**
   * The overridden bg color used to render the node.
   * @see {@link renderingBgColor}
   */
  bgcolor?: string
  /**
   * The overridden box color used to render the node.
   * @see {@link renderingBoxColor}
   */
  boxcolor?: string

  /** The fg color used to render the node. */
  get renderingColor(): string {
    const baseColor =
      this.color || this.constructor.color || LiteGraph.NODE_DEFAULT_COLOR
    return adjustColor(baseColor, { lightness: LiteGraph.nodeLightness })
  }

  /** The bg color used to render the node. */
  get renderingBgColor(): string {
    const baseBgColor =
      this.bgcolor || this.constructor.bgcolor || LiteGraph.NODE_DEFAULT_BGCOLOR
    const adjustments: ColorAdjustOptions = {
      opacity: LiteGraph.nodeOpacity,
      lightness: LiteGraph.nodeLightness
    }

    return adjustColor(
      this.mode === LGraphEventMode.BYPASS
        ? LiteGraph.NODE_DEFAULT_BYPASS_COLOR
        : baseBgColor,
      adjustments
    )
  }

  /** The box color used to render the node. */
  get renderingBoxColor(): string {
    if (this.boxcolor) return this.boxcolor

    if (LiteGraph.node_box_coloured_when_on) {
      if (this.action_triggered) return '#FFF'
      if (this.execute_triggered) return '#AAA'
    }

    if (LiteGraph.node_box_coloured_by_mode) {
      const modeColour =
        LiteGraph.NODE_MODES_COLORS[this.mode ?? LGraphEventMode.ALWAYS]
      if (modeColour) return modeColour
    }
    return LiteGraph.NODE_DEFAULT_BOXCOLOR
  }

  /** @inheritdoc {@link IColorable.setColorOption} */
  setColorOption(colorOption: ColorOption | null): void {
    if (colorOption == null) {
      this.color = undefined
      this.bgcolor = undefined
    } else {
      this.color = colorOption.color
      this.bgcolor = colorOption.bgcolor
    }
  }

  /** @inheritdoc {@link IColorable.getColorOption} */
  getColorOption(): ColorOption | null {
    return (
      Object.values(LGraphCanvas.node_colors).find(
        (colorOption) =>
          colorOption.color === this.color &&
          colorOption.bgcolor === this.bgcolor
      ) ?? null
    )
  }

  /**
   * The stroke styles that should be applied to the node.
   */
  strokeStyles: Record<
    string,
    (this: LGraphNode) => IDrawBoundingOptions | undefined
  >

  /**
   * The progress of node execution. Used to render a progress bar. Value between 0 and 1.
   */
  progress?: number

  exec_version?: number
  action_call?: string
  execute_triggered?: number
  action_triggered?: number
  /**
   * @deprecated This property is unsupported and will be removed in a future release.
   * Use `widgets_start_y` or a custom `arrange()` override instead.
   */
  widgets_up?: boolean
  widgets_start_y?: number
  lostFocusAt?: number
  gotFocusAt?: number
  badges: (LGraphBadge | (() => LGraphBadge))[] = []
  title_buttons: LGraphButton[] = []
  badgePosition: BadgePosition = BadgePosition.TopLeft
  onOutputRemoved?(this: LGraphNode, slot: number): void
  onInputRemoved?(this: LGraphNode, slot: number, input: INodeInputSlot): void
  /**
   * The width of the node when collapsed.
   * Updated by {@link LGraphCanvas.drawNode}
   */
  _collapsed_width?: number
  /**
   * Called once at the start of every frame.  Caller may change the values in {@link out}, which will be reflected in {@link boundingRect}.
   * WARNING: Making changes to boundingRect via onBounding is poorly supported, and will likely result in strange behaviour.
   */
  onBounding?(this: LGraphNode, out: Rect): void
  console?: string[]
  _level?: number
  _shape?: RenderShape
  mouseOver?: IMouseOverData
  redraw_on_mouse?: boolean
  resizable?: boolean
  clonable?: boolean
  _relative_id?: number
  clip_area?: boolean
  ignore_remove?: boolean
  has_errors?: boolean
  removable?: boolean
  block_delete?: boolean
  selected?: boolean
  showAdvanced?: boolean

  declare comfyDynamic?: Record<string, object>
  declare comfyClass?: string
  declare isVirtualNode?: boolean
  applyToGraph?(extraLinks?: LLink[]): void

  isSubgraphNode(): this is SubgraphNode {
    return false
  }

  /** @inheritdoc {@link renderArea} */
  private _renderArea = new Rectangle()
  /**
   * Rect describing the node area, including shadows and any protrusions.
   * Determines if the node is visible.  Calculated once at the start of every frame.
   */
  get renderArea(): ReadOnlyRect {
    return this._renderArea
  }

  /** @inheritdoc {@link boundingRect} */
  private _boundingRect: Rectangle = new Rectangle()
  /**
   * Cached node position & area as `x, y, width, height`.  Includes changes made by {@link onBounding}, if present.
   *
   * Determines the node hitbox and other rendering effects.  Calculated once at the start of every frame.
   */
  get boundingRect(): ReadOnlyRectangle {
    return this._boundingRect
  }

  /** The offset from {@link pos} to the top-left of {@link boundingRect}. */
  get boundingOffset(): Readonly<Point> {
    const {
      pos: [posX, posY],
      boundingRect: [bX, bY]
    } = this
    return [posX - bX, posY - bY]
  }

  /** {@link pos} and {@link size} values are backed by this {@link Rectangle}. */
  _posSize = new Rectangle()
  _pos: Point = this._posSize.pos
  _size: Size = this._posSize.size

  public get pos() {
    return this._pos
  }

  /** Node position does not necessarily correlate to the top-left corner. */
  public set pos(value) {
    if (!value || value.length < 2) return

    this._pos[0] = value[0]
    this._pos[1] = value[1]

    const mutations = useLayoutMutations()
    mutations.setSource(LayoutSource.Canvas)
    mutations.moveNode(String(this.id), { x: value[0], y: value[1] })
  }

  /**
   * Set the node position to an absolute location.
   */
  setPos(x: number, y: number): void {
    this.pos = [x, y]
  }

  public get size() {
    return this._size
  }

  public set size(value) {
    if (!value || value.length < 2) return

    this._size[0] = value[0]
    this._size[1] = value[1]

    const mutations = useLayoutMutations()
    mutations.setSource(LayoutSource.Canvas)
    mutations.resizeNode(String(this.id), {
      width: value[0],
      height: value[1]
    })
  }

  /**
   * The size of the node used for rendering.
   */
  get renderingSize(): Size {
    return this.flags.collapsed ? [this._collapsed_width ?? 0, 0] : this._size
  }

  get shape(): RenderShape | undefined {
    return this._shape
  }

  set shape(v: RenderShape | 'default' | 'box' | 'round' | 'circle' | 'card') {
    const oldValue = this._shape
    switch (v) {
      case 'default':
        this._shape = undefined
        break
      case 'box':
        this._shape = RenderShape.BOX
        break
      case 'round':
        this._shape = RenderShape.ROUND
        break
      case 'circle':
        this._shape = RenderShape.CIRCLE
        break
      case 'card':
        this._shape = RenderShape.CARD
        break
      default:
        this._shape = v
    }
    if (oldValue !== this._shape) {
      this.graph?.trigger('node:property:changed', {
        nodeId: this.id,
        property: 'shape',
        oldValue,
        newValue: this._shape
      })
    }
  }

  /**
   * The shape of the node used for rendering. @see {@link RenderShape}
   */
  get renderingShape(): RenderShape {
    return this._shape || this.constructor.shape || LiteGraph.NODE_DEFAULT_SHAPE
  }

  public get is_selected(): boolean | undefined {
    return this.selected
  }

  public set is_selected(value: boolean) {
    this.selected = value
  }

  public get title_mode(): TitleMode {
    return this.constructor.title_mode ?? TitleMode.NORMAL_TITLE
  }

  onConnectInput?(
    this: LGraphNode,
    target_slot: number,
    type: unknown,
    output: INodeOutputSlot | SubgraphIO,
    node: LGraphNode | SubgraphInputNode,
    slot: number
  ): boolean
  onConnectOutput?(
    this: LGraphNode,
    slot: number,
    type: unknown,
    input: INodeInputSlot | SubgraphIO,
    target_node: number | LGraphNode | SubgraphOutputNode,
    target_slot: number
  ): boolean
  onResize?(this: LGraphNode, size: Size): void
  onPropertyChanged?(
    this: LGraphNode,
    name: string,
    value: unknown,
    prev_value?: unknown
  ): boolean
  /** Called for each connection that is created, updated, or removed. This includes "restored" connections when deserialising. */
  onConnectionsChange?(
    this: LGraphNode,
    type: ISlotType,
    index: number,
    isConnected: boolean,
    link_info: LLink | null | undefined,
    inputOrOutput: INodeInputSlot | INodeOutputSlot | SubgraphIO
  ): void
  onInputAdded?(this: LGraphNode, input: INodeInputSlot): void
  onOutputAdded?(this: LGraphNode, output: INodeOutputSlot): void
  onConfigure?(this: LGraphNode, serialisedNode: ISerialisedNode): void
  onSerialize?(this: LGraphNode, serialised: ISerialisedNode): void
  onExecute?(
    this: LGraphNode,
    param?: unknown,
    options?: { action_call?: string }
  ): void
  onAction?(
    this: LGraphNode,
    action: string,
    param: unknown,
    options: { action_call?: string }
  ): void
  onDrawBackground?(this: LGraphNode, ctx: CanvasRenderingContext2D): void
  onNodeCreated?(this: LGraphNode): void
  /**
   * Callback invoked by {@link connect} to override the target slot index.
   * Its return value overrides the target index selection.
   * @param target_slot The current input slot index
   * @param requested_slot The originally requested slot index - could be negative, or if using (deprecated) name search, a string
   * @returns {number | null} If a number is returned, the connection will be made to that input index.
   * If an invalid index or non-number (false, null, NaN etc) is returned, the connection will be cancelled.
   */
  onBeforeConnectInput?(
    this: LGraphNode,
    target_slot: number,
    requested_slot?: number | string
  ): number | false | null
  onShowCustomPanelInfo?(this: LGraphNode, panel: Panel): void
  onAddPropertyToPanel?(this: LGraphNode, pName: string, panel: Panel): boolean
  onWidgetChanged?(
    this: LGraphNode,
    name: string,
    value: unknown,
    old_value: unknown,
    w: IBaseWidget
  ): void
  onDeselected?(this: LGraphNode): void
  onKeyUp?(this: LGraphNode, e: KeyboardEvent): void
  onKeyDown?(this: LGraphNode, e: KeyboardEvent): void
  onSelected?(this: LGraphNode): void
  getExtraMenuOptions?(
    this: LGraphNode,
    canvas: LGraphCanvas,
    options: (IContextMenuValue<unknown> | null)[]
  ): (IContextMenuValue<unknown> | null)[]
  getMenuOptions?(this: LGraphNode, canvas: LGraphCanvas): IContextMenuValue[]
  onAdded?(this: LGraphNode, graph: LGraph): void
  onDrawCollapsed?(
    this: LGraphNode,
    ctx: CanvasRenderingContext2D,
    cavnas: LGraphCanvas
  ): boolean
  onDrawForeground?(
    this: LGraphNode,
    ctx: CanvasRenderingContext2D,
    canvas: LGraphCanvas,
    canvasElement: HTMLCanvasElement
  ): void
  onMouseLeave?(this: LGraphNode, e: CanvasPointerEvent): void
  /**
   * Override the default slot menu options.
   */
  getSlotMenuOptions?(this: LGraphNode, slot: IFoundSlot): IContextMenuValue[]
  /**
   * Add extra menu options to the slot context menu.
   */
  getExtraSlotMenuOptions?(
    this: LGraphNode,
    slot: IFoundSlot
  ): IContextMenuValue[]

  // FIXME: Re-typing
  onDropItem?(this: LGraphNode, event: Event): boolean
  onDropData?(
    this: LGraphNode,
    data: string | ArrayBuffer,
    filename: string,
    file: File
  ): void
  onDropFile?(this: LGraphNode, file: File): void
  onInputClick?(this: LGraphNode, index: number, e: CanvasPointerEvent): void
  onInputDblClick?(this: LGraphNode, index: number, e: CanvasPointerEvent): void
  onOutputClick?(this: LGraphNode, index: number, e: CanvasPointerEvent): void
  onOutputDblClick?(
    this: LGraphNode,
    index: number,
    e: CanvasPointerEvent
  ): void
  onGetPropertyInfo?(this: LGraphNode, property: string): INodePropertyInfo
  onNodeOutputAdd?(this: LGraphNode, value: unknown): void
  onNodeInputAdd?(this: LGraphNode, value: unknown): void
  onMenuNodeInputs?(
    this: LGraphNode,
    entries: (IContextMenuValue<INodeSlotContextItem> | null)[]
  ): (IContextMenuValue<INodeSlotContextItem> | null)[]
  onMenuNodeOutputs?(
    this: LGraphNode,
    entries: (IContextMenuValue<INodeSlotContextItem> | null)[]
  ): (IContextMenuValue<INodeSlotContextItem> | null)[]
  onMouseUp?(
    this: LGraphNode,
    e: CanvasPointerEvent,
    pos: Point,
    canvas: LGraphCanvas
  ): void
  onMouseEnter?(this: LGraphNode, e: CanvasPointerEvent): void
  /** Blocks drag if return value is truthy. @param pos Offset from {@link LGraphNode.pos}. */
  onMouseDown?(
    this: LGraphNode,
    e: CanvasPointerEvent,
    pos: Point,
    canvas: LGraphCanvas
  ): boolean
  /** @param pos Offset from {@link LGraphNode.pos}. */
  onDblClick?(
    this: LGraphNode,
    e: CanvasPointerEvent,
    pos: Point,
    canvas: LGraphCanvas
  ): void
  /** @param pos Offset from {@link LGraphNode.pos}. */
  onNodeTitleDblClick?(
    this: LGraphNode,
    e: CanvasPointerEvent,
    pos: Point,
    canvas: LGraphCanvas
  ): void
  onDrawTitle?(this: LGraphNode, ctx: CanvasRenderingContext2D): void
  onDrawTitleText?(
    this: LGraphNode,
    ctx: CanvasRenderingContext2D,
    title_height: number,
    size: Size,
    scale: number,
    title_text_font: string,
    selected?: boolean
  ): void
  onDrawTitleBox?(
    this: LGraphNode,
    ctx: CanvasRenderingContext2D,
    title_height: number,
    size: Size,
    scale: number
  ): void
  onDrawTitleBar?(
    this: LGraphNode,
    ctx: CanvasRenderingContext2D,
    title_height: number,
    size: Size,
    scale: number,
    fgcolor: string
  ): void
  onRemoved?(this: LGraphNode): void
  onMouseMove?(
    this: LGraphNode,
    e: CanvasPointerEvent,
    pos: Point,
    arg2: LGraphCanvas
  ): void
  onPropertyChange?(this: LGraphNode): void
  updateOutputData?(this: LGraphNode, origin_slot: number): void

  private _getErrorStrokeStyle(
    this: LGraphNode
  ): IDrawBoundingOptions | undefined {
    if (this.has_errors) {
      return {
        padding: 12,
        lineWidth: 10,
        color: LiteGraph.NODE_ERROR_COLOUR
      }
    }
  }

  private _getSelectedStrokeStyle(
    this: LGraphNode
  ): IDrawBoundingOptions | undefined {
    if (this.selected) {
      return {
        padding: this.has_errors ? 20 : undefined
      }
    }
  }

  constructor(title: string, type?: string) {
    this.id = LiteGraph.use_uuids ? LiteGraph.uuidv4() : -1
    this.title = title || 'Unnamed'
    this.type = type ?? ''
    this.size = [LiteGraph.NODE_WIDTH, 60]
    this.pos = [10, 10]
    this.strokeStyles = {
      error: this._getErrorStrokeStyle,
      selected: this._getSelectedStrokeStyle
    }
    // Initialize property manager with tracked properties
    this.changeTracker = new LGraphNodeProperties(this)
  }

  /** Internal callback for subgraph nodes. Do not implement externally. */
  _internalConfigureAfterSlots?(): void

  /**
   * configure a node from an object containing the serialized info
   */
  configure(info: ISerialisedNode): void {
    if (this.graph) {
      this.graph._version++
    }
    if (info.id === -1) info.id = this.id
    for (const j in info) {
      if (j == 'properties') {
        // i don't want to clone properties, I want to reuse the old container
        for (const k in info.properties) {
          this.properties[k] = info.properties[k]
          this.onPropertyChanged?.(k, info.properties[k])
        }
        continue
      }

      // @ts-expect-error #594
      if (info[j] == null) {
        continue
        // @ts-expect-error #594
      } else if (typeof info[j] == 'object') {
        // @ts-expect-error #594
        if (this[j]?.configure) {
          // @ts-expect-error #594
          this[j]?.configure(info[j])
        } else {
          // @ts-expect-error #594
          this[j] = LiteGraph.cloneObject(info[j], this[j])
        }
      } else {
        // value
        // @ts-expect-error #594
        this[j] = info[j]
      }
    }

    if (!info.title) {
      this.title = this.constructor.title
    }

    this.inputs ??= []
    this.inputs = this.inputs.map((input) =>
      toClass(NodeInputSlot, input, this)
    )
    for (const [i, input] of this.inputs.entries()) {
      const link =
        this.graph && input.link != null
          ? this.graph._links.get(input.link)
          : null
      this.onConnectionsChange?.(NodeSlotType.INPUT, i, true, link, input)
      this.onInputAdded?.(input)
    }

    this.outputs ??= []
    this.outputs = this.outputs.map((output) =>
      toClass(NodeOutputSlot, output, this)
    )
    for (const [i, output] of this.outputs.entries()) {
      if (!output.links) continue

      for (const linkId of output.links) {
        const link = this.graph ? this.graph._links.get(linkId) : null
        this.onConnectionsChange?.(NodeSlotType.OUTPUT, i, true, link, output)
      }
      this.onOutputAdded?.(output)
    }

    // SubgraphNode callback.
    this._internalConfigureAfterSlots?.()

    if (this.widgets) {
      for (const w of this.widgets) {
        if (!w) continue

        const input = this.inputs.find((i) => i.widget?.name === w.name)
        if (input?.label) w.label = input.label

        if (
          w.options?.property &&
          this.properties[w.options.property] != undefined
        )
          w.value = JSON.parse(
            JSON.stringify(this.properties[w.options.property])
          )
      }

      if (info.widgets_values) {
        let i = 0
        for (const widget of this.widgets ?? []) {
          if (widget.serialize === false) continue
          if (i >= info.widgets_values.length) break
          widget.value = info.widgets_values[i++]
        }
      }
    }

    // Sync the state of this.resizable.
    if (this.pinned) this.resizable = false

    if (this.widgets_up) {
      console.warn(
        `[LiteGraph] Node type "${this.type}" uses deprecated property "widgets_up". ` +
          'This property is unsupported and will be removed. ' +
          'Use "widgets_start_y" or a custom arrange() override instead.'
      )
    }

    this.onConfigure?.(info)
  }

  /**
   * serialize the content
   */
  serialize(): ISerialisedNode {
    // create serialization object
    const o: ISerialisedNode = {
      id: this.id,
      type: this.type,
      pos: [this.pos[0], this.pos[1]],
      size: [this.size[0], this.size[1]],
      flags: LiteGraph.cloneObject(this.flags),
      order: this.order,
      mode: this.mode,
      showAdvanced: this.showAdvanced
    }

    // special case for when there were errors
    if (this.constructor === LGraphNode && this.last_serialization)
      return { ...this.last_serialization, mode: o.mode, pos: o.pos }

    if (this.inputs)
      o.inputs = this.inputs.map((input) => inputAsSerialisable(input))
    if (this.outputs)
      // @ts-expect-error - Output serialization type mismatch
      o.outputs = this.outputs.map((output) => outputAsSerialisable(output))

    if (this.title && this.title != this.constructor.title) o.title = this.title

    if (this.properties) o.properties = LiteGraph.cloneObject(this.properties)

    const { widgets } = this
    if (widgets && this.serialize_widgets) {
      o.widgets_values = []
      for (const [i, widget] of widgets.entries()) {
        if (widget.serialize === false) continue
        const val = widget?.value
        // Ensure object values are plain (not reactive proxies) for structuredClone compatibility.
        o.widgets_values[i] =
          val != null && typeof val === 'object'
            ? JSON.parse(JSON.stringify(val))
            : (val ?? null)
      }
    }

    if (!o.type && this.constructor.type) o.type = this.constructor.type

    if (this.color) o.color = this.color
    if (this.bgcolor) o.bgcolor = this.bgcolor
    if (this.boxcolor) o.boxcolor = this.boxcolor
    if (this.shape) o.shape = this.shape

    if (this.onSerialize?.(o))
      console.warn(
        "node onSerialize shouldn't return anything, data should be stored in the object pass in the first parameter"
      )

    return o
  }

  /* Creates a clone of this node */
  clone(): LGraphNode | null {
    if (this.type == null) return null
    const node = LiteGraph.createNode(this.type)
    if (!node) return null

    // we clone it because serialize returns shared containers
    const data = LiteGraph.cloneObject(this.serialize())
    const { inputs, outputs } = data

    // remove links
    if (inputs) {
      for (const input of inputs) {
        input.link = null
      }
    }

    if (outputs) {
      for (const { links } of outputs) {
        if (links) links.length = 0
      }
    }

    // @ts-expect-error Exceptional case: id is removed so that the graph can assign a new one on add.
    data.id = undefined

    if (LiteGraph.use_uuids) data.id = LiteGraph.uuidv4()

    node.configure(data)

    return node
  }

  /**
   * serialize and stringify
   */
  toString(): string {
    return JSON.stringify(this.serialize())
  }

  /**
   * get the title string
   */
  getTitle(): string {
    return this.title || this.constructor.title
  }

  /**
   * sets the value of a property
   * @param name
   * @param value
   */
  setProperty(name: string, value: TWidgetValue): void {
    this.properties ||= {}
    if (value === this.properties[name]) return

    const prev_value = this.properties[name]
    this.properties[name] = value
    // abort change
    if (this.onPropertyChanged?.(name, value, prev_value) === false)
      this.properties[name] = prev_value

    if (this.widgets) {
      for (const w of this.widgets) {
        if (!w) continue

        if (w.options.property == name) {
          w.value = value
          break
        }
      }
    }
  }

  /**
   * sets the output data
   * @param slot
   * @param data
   */
  setOutputData(
    slot: number,
    data: number | string | boolean | { toToolTip?(): string }
  ): void {
    const { outputs } = this
    if (!outputs) return

    // this maybe slow and a niche case
    if (slot == -1 || slot >= outputs.length) return

    const output_info = outputs[slot]
    if (!output_info) return

    // store data in the output itself in case we want to debug
    output_info._data = data

    if (!this.graph) throw new NullGraphError()

    // if there are connections, pass the data to the connections
    const { links } = outputs[slot]
    if (links) {
      for (const id of links) {
        const link = this.graph._links.get(id)
        if (link) link.data = data
      }
    }
  }

  /**
   * sets the output data type, useful when you want to be able to overwrite the data type
   */
  setOutputDataType(slot: number, type: ISlotType): void {
    const { outputs } = this
    if (!outputs || slot == -1 || slot >= outputs.length) return

    const output_info = outputs[slot]
    if (!output_info) return
    // store data in the output itself in case we want to debug
    output_info.type = type

    if (!this.graph) throw new NullGraphError()

    // if there are connections, pass the data to the connections
    const { links } = outputs[slot]
    if (links) {
      for (const id of links) {
        const link = this.graph._links.get(id)
        if (link) link.type = type
      }
    }
  }

  /**
   * Retrieves the input data (data traveling through the connection) from one slot
   * @param slot
   * @param force_update if set to true it will force the connected node of this slot to output data into this link
   * @returns data or if it is not connected returns undefined
   */
  getInputData(slot: number, force_update?: boolean): unknown {
    if (!this.inputs) return

    if (slot >= this.inputs.length || this.inputs[slot].link == null) return
    if (!this.graph) throw new NullGraphError()

    const link_id = this.inputs[slot].link
    const link = this.graph._links.get(link_id)
    // bug: weird case but it happens sometimes
    if (!link) return null

    if (!force_update) return link.data

    // special case: used to extract data from the incoming connection before the graph has been executed
    const node = this.graph.getNodeById(link.origin_id)
    if (!node) return link.data

    if (node.updateOutputData) {
      node.updateOutputData(link.origin_slot)
    } else {
      node.onExecute?.()
    }

    return link.data
  }

  /**
   * Retrieves the input data type (in case this supports multiple input types)
   * @param slot
   * @returns datatype in string format
   */
  getInputDataType(slot: number): ISlotType | null {
    if (!this.inputs) return null
    if (slot >= this.inputs.length || this.inputs[slot].link == null)
      return null
    if (!this.graph) throw new NullGraphError()

    const link_id = this.inputs[slot].link
    const link = this.graph._links.get(link_id)
    // bug: weird case but it happens sometimes
    if (!link) return null

    const node = this.graph.getNodeById(link.origin_id)
    if (!node) return link.type

    const output_info = node.outputs[link.origin_slot]
    return output_info ? output_info.type : null
  }

  /**
   * Retrieves the input data from one slot using its name instead of slot number
   * @param slot_name
   * @param force_update if set to true it will force the connected node of this slot to output data into this link
   * @returns data or if it is not connected returns null
   */
  getInputDataByName(slot_name: string, force_update: boolean): unknown {
    const slot = this.findInputSlot(slot_name)
    return slot == -1 ? null : this.getInputData(slot, force_update)
  }

  /**
   * tells you if there is a connection in one input slot
   * @param slot The 0-based index of the input to check
   * @returns `true` if the input slot has a link ID (does not perform validation)
   */
  isInputConnected(slot: number): boolean {
    if (!this.inputs) return false
    return slot < this.inputs.length && this.inputs[slot].link != null
  }

  /**
   * tells you info about an input connection (which node, type, etc)
   * @returns object or null { link: id, name: string, type: string or 0 }
   */
  getInputInfo(slot: number): INodeInputSlot | null {
    return !this.inputs || !(slot < this.inputs.length)
      ? null
      : this.inputs[slot]
  }

  /**
   * Returns the link info in the connection of an input slot
   * @returns object or null
   */
  getInputLink(slot: number): LLink | null {
    if (!this.inputs) return null

    if (slot < this.inputs.length) {
      if (!this.graph) throw new NullGraphError()

      const input = this.inputs[slot]
      if (input.link != null) {
        return this.graph._links.get(input.link) ?? null
      }
    }
    return null
  }

  /**
   * returns the node connected in the input slot
   * @returns node or null
   */
  getInputNode(slot: number): LGraphNode | null {
    if (!this.inputs) return null
    if (slot >= this.inputs.length) return null

    const input = this.inputs[slot]
    if (!input || input.link === null) return null
    if (!this.graph) throw new NullGraphError()

    const link_info = this.graph._links.get(input.link)
    if (!link_info) return null

    return this.graph.getNodeById(link_info.origin_id)
  }

  /**
   * returns the value of an input with this name, otherwise checks if there is a property with that name
   * @returns value
   */
  getInputOrProperty(name: string): unknown {
    const { inputs } = this
    if (!inputs?.length) {
      return this.properties ? this.properties[name] : null
    }
    if (!this.graph) throw new NullGraphError()

    for (const input of inputs) {
      if (name == input.name && input.link != null) {
        const link = this.graph._links.get(input.link)
        if (link) return link.data
      }
    }
    return this.properties[name]
  }

  /**
   * tells you the last output data that went in that slot
   * @returns object or null
   */
  getOutputData(slot: number): unknown {
    if (!this.outputs) return null
    if (slot >= this.outputs.length) return null

    const info = this.outputs[slot]
    return info._data
  }

  /**
   * tells you info about an output connection (which node, type, etc)
   * @returns object or null { name: string, type: string, links: [ ids of links in number ] }
   */
  getOutputInfo(slot: number): INodeOutputSlot | null {
    return !this.outputs || !(slot < this.outputs.length)
      ? null
      : this.outputs[slot]
  }

  /**
   * tells you if there is a connection in one output slot
   */
  isOutputConnected(slot: number): boolean {
    if (!this.outputs) return false
    return (
      slot < this.outputs.length && Number(this.outputs[slot].links?.length) > 0
    )
  }

  /**
   * tells you if there is any connection in the output slots
   */
  isAnyOutputConnected(): boolean {
    const { outputs } = this
    if (!outputs) return false

    for (const output of outputs) {
      if (output.links?.length) return true
    }
    return false
  }

  /**
   * retrieves all the nodes connected to this output slot
   */
  getOutputNodes(slot: number): LGraphNode[] | null {
    const { outputs } = this
    if (!outputs || outputs.length == 0) return null

    if (slot >= outputs.length) return null

    const { links } = outputs[slot]
    if (!links || links.length == 0) return null
    if (!this.graph) throw new NullGraphError()

    const r: LGraphNode[] = []
    for (const id of links) {
      const link = this.graph._links.get(id)
      if (link) {
        const target_node = this.graph.getNodeById(link.target_id)
        if (target_node) {
          r.push(target_node)
        }
      }
    }
    return r
  }

  addOnTriggerInput(): number {
    const trigS = this.findInputSlot('onTrigger')
    if (trigS == -1) {
      this.addInput('onTrigger', LiteGraph.EVENT, {
        nameLocked: true
      })
      return this.findInputSlot('onTrigger')
    }
    return trigS
  }

  addOnExecutedOutput(): number {
    const trigS = this.findOutputSlot('onExecuted')
    if (trigS == -1) {
      this.addOutput('onExecuted', LiteGraph.ACTION, {
        nameLocked: true
      })
      return this.findOutputSlot('onExecuted')
    }
    return trigS
  }

  onAfterExecuteNode(param: unknown, options?: { action_call?: string }) {
    const trigS = this.findOutputSlot('onExecuted')
    if (trigS != -1) {
      this.triggerSlot(trigS, param, null, options)
    }
  }

  changeMode(modeTo: number): boolean {
    switch (modeTo) {
      case LGraphEventMode.ON_EVENT:
        break

      case LGraphEventMode.ON_TRIGGER:
        this.addOnTriggerInput()
        this.addOnExecutedOutput()
        break

      case LGraphEventMode.NEVER:
        break

      case LGraphEventMode.ALWAYS:
        break

      // @ts-expect-error Not impl.
      case LiteGraph.ON_REQUEST:
        break

      default:
        return false
        break
    }
    this.mode = modeTo
    return true
  }

  /**
   * Triggers the node code execution, place a boolean/counter to mark the node as being executed
   */
  doExecute(param?: unknown, options?: { action_call?: string }): void {
    options = options || {}
    if (this.onExecute) {
      // enable this to give the event an ID
      options.action_call ||= `${this.id}_exec_${Math.floor(Math.random() * 9999)}`
      if (!this.graph) throw new NullGraphError()

      // @ts-expect-error Technically it works when id is a string. Array gets props.
      this.graph.nodes_executing[this.id] = true
      this.onExecute(param, options)
      // @ts-expect-error deprecated
      this.graph.nodes_executing[this.id] = false

      // save execution/action ref
      this.exec_version = this.graph.iteration
      if (options?.action_call) {
        this.action_call = options.action_call
        // @ts-expect-error deprecated
        this.graph.nodes_executedAction[this.id] = options.action_call
      }
    }
    // the nFrames it will be used (-- each step), means "how old" is the event
    this.execute_triggered = 2
    this.onAfterExecuteNode?.(param, options)
  }

  /**
   * Triggers an action, wrapped by logics to control execution flow
   * @param action name
   */
  actionDo(
    action: string,
    param: unknown,
    options: { action_call?: string }
  ): void {
    options = options || {}
    if (this.onAction) {
      // enable this to give the event an ID
      options.action_call ||= `${this.id}_${action || 'action'}_${Math.floor(Math.random() * 9999)}`
      if (!this.graph) throw new NullGraphError()

      // @ts-expect-error deprecated
      this.graph.nodes_actioning[this.id] = action || 'actioning'
      this.onAction(action, param, options)
      // @ts-expect-error deprecated
      this.graph.nodes_actioning[this.id] = false

      // save execution/action ref
      if (options?.action_call) {
        this.action_call = options.action_call
        // @ts-expect-error deprecated
        this.graph.nodes_executedAction[this.id] = options.action_call
      }
    }
    // the nFrames it will be used (-- each step), means "how old" is the event
    this.action_triggered = 2
    this.onAfterExecuteNode?.(param, options)
  }

  /**
   * Triggers an event in this node, this will trigger any output with the same name
   * @param action name ( "on_play", ... ) if action is equivalent to false then the event is send to all
   */
  trigger(
    action: string,
    param: unknown,
    options: { action_call?: string }
  ): void {
    const { outputs } = this
    if (!outputs || !outputs.length) {
      return
    }

    if (this.graph) this.graph._last_trigger_time = LiteGraph.getTime()

    for (const [i, output] of outputs.entries()) {
      if (
        !output ||
        output.type !== LiteGraph.EVENT ||
        (action && output.name != action)
      ) {
        continue
      }
      this.triggerSlot(i, param, null, options)
    }
  }

  /**
   * Triggers a slot event in this node: cycle output slots and launch execute/action on connected nodes
   * @param slot the index of the output slot
   * @param link_id [optional] in case you want to trigger and specific output link in a slot
   */
  triggerSlot(
    slot: number,
    param: unknown,
    link_id: number | null,
    options?: { action_call?: string }
  ): void {
    options = options || {}
    if (!this.outputs) return

    if (slot == null) {
      console.error('slot must be a number')
      return
    }

    if (typeof slot !== 'number')
      console.warn(
        "slot must be a number, use node.trigger('name') if you want to use a string"
      )

    const output = this.outputs[slot]
    if (!output) return

    const links = output.links
    if (!links || !links.length) return

    if (!this.graph) throw new NullGraphError()
    this.graph._last_trigger_time = LiteGraph.getTime()

    // for every link attached here
    for (const id of links) {
      // to skip links
      if (link_id != null && link_id != id) continue

      const link_info = this.graph._links.get(id)
      // not connected
      if (!link_info) continue

      link_info._last_time = LiteGraph.getTime()
      const node = this.graph.getNodeById(link_info.target_id)
      // node not found?
      if (!node) continue

      if (node.mode === LGraphEventMode.ON_TRIGGER) {
        // generate unique trigger ID if not present
        if (!options.action_call)
          options.action_call = `${this.id}_trigg_${Math.floor(Math.random() * 9999)}`
        // -- wrapping node.onExecute(param); --
        node.doExecute?.(param, options)
      } else if (node.onAction) {
        // generate unique action ID if not present
        if (!options.action_call)
          options.action_call = `${this.id}_act_${Math.floor(Math.random() * 9999)}`
        // pass the action name
        const target_connection = node.inputs[link_info.target_slot]
        node.actionDo(target_connection.name, param, options)
      }
    }
  }

  /**
   * clears the trigger slot animation
   * @param slot the index of the output slot
   * @param link_id [optional] in case you want to trigger and specific output link in a slot
   */
  clearTriggeredSlot(slot: number, link_id: number): void {
    if (!this.outputs) return

    const output = this.outputs[slot]
    if (!output) return

    const links = output.links
    if (!links || !links.length) return

    if (!this.graph) throw new NullGraphError()

    // for every link attached here
    for (const id of links) {
      // to skip links
      if (link_id != null && link_id != id) continue

      const link_info = this.graph._links.get(id)
      // not connected
      if (!link_info) continue

      link_info._last_time = 0
    }
  }

  /**
   * changes node size and triggers callback
   */
  setSize(size: Size): void {
    this.size = size
    this.onResize?.(this.size)
  }

  /**
   * Expands the node size to fit its content.
   */
  expandToFitContent(): void {
    const newSize = this.computeSize()
    this.setSize([
      Math.max(this.size[0], newSize[0]),
      Math.max(this.size[1], newSize[1])
    ])
  }

  /**
   * add a new property to this node
   * @param type string defining the output type ("vec3","number",...)
   * @param extra_info this can be used to have special properties of the property (like values, etc)
   */
  addProperty(
    name: string,
    default_value: NodeProperty | undefined,
    type?: string,
    extra_info?: Partial<INodePropertyInfo>
  ): INodePropertyInfo {
    const o: INodePropertyInfo = { name, type, default_value }
    if (extra_info) Object.assign(o, extra_info)

    this.properties_info ||= []
    this.properties_info.push(o)
    this.properties ||= {}
    this.properties[name] = default_value
    return o
  }

  /**
   * add a new output slot to use in this node
   * @param type string defining the output type ("vec3","number",...)
   * @param extra_info this can be used to have special properties of an output (label, special color, position, etc)
   */
  addOutput<TProperties extends Partial<INodeOutputSlot>>(
    name: string,
    type: ISlotType,
    extra_info?: TProperties
  ): INodeOutputSlot & TProperties {
    const output = Object.assign(
      new NodeOutputSlot({ name, type, links: null }, this),
      extra_info
    )

    this.outputs ||= []
    this.outputs.push(output)
    this.onOutputAdded?.(output)

    if (LiteGraph.auto_load_slot_types)
      LiteGraph.registerNodeAndSlotType(this, type, true)

    this.expandToFitContent()
    this.setDirtyCanvas(true, true)
    return output
  }

  /**
   * remove an existing output slot
   */
  removeOutput(slot: number): void {
    // Only disconnect if node is part of a graph
    if (this.graph) {
      this.disconnectOutput(slot)
    }
    const { outputs } = this
    outputs.splice(slot, 1)

    for (let i = slot; i < outputs.length; ++i) {
      const output = outputs[i]
      if (!output || !output.links) continue

      // Only update link indices if node is part of a graph
      if (this.graph) {
        for (const linkId of output.links) {
          const link = this.graph._links.get(linkId)
          if (link) link.origin_slot--
        }
      }
    }

    this.onOutputRemoved?.(slot)
    this.setDirtyCanvas(true, true)
  }

  /**
   * add a new input slot to use in this node
   * @param type string defining the input type ("vec3","number",...), it its a generic one use 0
   * @param extra_info this can be used to have special properties of an input (label, color, position, etc)
   */
  addInput<TProperties extends Partial<INodeInputSlot>>(
    name: string,
    type: ISlotType,
    extra_info?: TProperties
  ): INodeInputSlot & TProperties {
    type ||= 0

    const input = Object.assign(
      new NodeInputSlot({ name, type, link: null }, this),
      extra_info
    )

    this.inputs ||= []
    this.inputs.push(input)
    this.expandToFitContent()

    this.onInputAdded?.(input)
    LiteGraph.registerNodeAndSlotType(this, type)

    this.setDirtyCanvas(true, true)
    return input
  }

  /**
   * remove an existing input slot
   */
  removeInput(slot: number): void {
    // Only disconnect if node is part of a graph
    if (this.graph) {
      this.disconnectInput(slot, true)
    }
    const { inputs } = this
    const slot_info = inputs.splice(slot, 1)

    for (let i = slot; i < inputs.length; ++i) {
      const input = inputs[i]
      if (!input?.link) continue

      // Only update link indices if node is part of a graph
      if (this.graph) {
        const link = this.graph._links.get(input.link)
        if (link) link.target_slot--
      }
    }
    this.onInputRemoved?.(slot, slot_info[0])
    this.setDirtyCanvas(true, true)
  }

  /**
   * computes the minimum size of a node according to its inputs and output slots
   * @returns the total size
   */
  computeSize(out?: Size): Size {
    const ctorSize = this.constructor.size
    if (ctorSize) return [ctorSize[0], ctorSize[1]]

    const { inputs, outputs, widgets } = this
    let rows = Math.max(
      inputs ? inputs.filter((input) => !isWidgetInputSlot(input)).length : 1,
      outputs ? outputs.length : 1
    )
    const size = out ?? [0, 0]
    rows = Math.max(rows, 1)
    // although it should be graphcanvas.inner_text_font size
    const font_size = LiteGraph.NODE_TEXT_SIZE

    const padLeft = LiteGraph.NODE_TITLE_HEIGHT
    const padRight = padLeft * 0.33
    const title_width =
      padLeft + compute_text_size(this.title, this.titleFontStyle) + padRight
    let input_width = 0
    let widgetWidth = 0
    let output_width = 0

    if (inputs) {
      for (const input of inputs) {
        const text = input.label || input.localized_name || input.name || ''
        const text_width = compute_text_size(text, this.innerFontStyle)
        if (isWidgetInputSlot(input)) {
          const widget = this.getWidgetFromSlot(input)
          if (widget && !this.isWidgetVisible(widget)) continue

          if (text_width > widgetWidth) widgetWidth = text_width
        } else {
          if (text_width > input_width) input_width = text_width
        }
      }
    }

    if (outputs) {
      for (const output of outputs) {
        const text = output.label || output.localized_name || output.name || ''
        const text_width = compute_text_size(text, this.innerFontStyle)
        if (output_width < text_width) output_width = text_width
      }
    }

    const minWidth = LiteGraph.NODE_WIDTH * (widgets?.length ? 1.5 : 1)
    // Text + slot width + centre padding
    const centrePadding = input_width && output_width ? 5 : 0
    const slotsWidth =
      input_width +
      output_width +
      2 * LiteGraph.NODE_SLOT_HEIGHT +
      centrePadding

    // Total distance from edge of node to the inner edge of the widget 'previous' arrow button
    const widgetMargin =
      BaseWidget.margin + BaseWidget.arrowMargin + BaseWidget.arrowWidth
    const widgetPadding = BaseWidget.minValueWidth + 2 * widgetMargin
    if (widgetWidth) widgetWidth += widgetPadding

    size[0] = Math.max(slotsWidth, widgetWidth, title_width, minWidth)
    size[1] =
      (this.constructor.slot_start_y || 0) + rows * LiteGraph.NODE_SLOT_HEIGHT

    // Get widget height & expand size if necessary
    let widgets_height = 0
    if (widgets?.length) {
      for (const widget of widgets) {
        if (!this.isWidgetVisible(widget)) continue

        let widget_height = 0
        if (widget.computeSize) {
          widget_height += widget.computeSize(size[0])[1]
        } else if (widget.computeLayoutSize) {
          // Expand widget width if necessary
          const { minHeight, minWidth } = widget.computeLayoutSize(this)
          const widgetWidth = minWidth + widgetPadding
          if (widgetWidth > size[0]) size[0] = widgetWidth

          widget_height += minHeight
        } else {
          widget_height += LiteGraph.NODE_WIDGET_HEIGHT
        }
        widgets_height += widget_height + 4
      }
      widgets_height += 8
    }

    // compute height using widgets height
    if (this.widgets_up) size[1] = Math.max(size[1], widgets_height)
    else if (this.widgets_start_y != null)
      size[1] = Math.max(size[1], widgets_height + this.widgets_start_y)
    else size[1] += widgets_height

    function compute_text_size(text: string, fontStyle: string) {
      return (
        LGraphCanvas._measureText?.(text, fontStyle) ??
        font_size * (text?.length ?? 0) * 0.6
      )
    }

    if (this.constructor.min_height && size[1] < this.constructor.min_height) {
      size[1] = this.constructor.min_height
    }

    // margin
    size[1] += 6

    return size
  }

  inResizeCorner(canvasX: number, canvasY: number): boolean {
    const rows = this.outputs ? this.outputs.length : 1
    const outputs_offset =
      (this.constructor.slot_start_y || 0) + rows * LiteGraph.NODE_SLOT_HEIGHT
    return isInRectangle(
      canvasX,
      canvasY,
      this.pos[0] + this.size[0] - 15,
      this.pos[1] + Math.max(this.size[1] - 15, outputs_offset),
      20,
      20
    )
  }

  /**
   * Returns which resize corner the point is over, if any.
   * @param canvasX X position in canvas coordinates
   * @param canvasY Y position in canvas coordinates
   * @returns The compass corner the point is in, otherwise `undefined`.
   */
  findResizeDirection(
    canvasX: number,
    canvasY: number
  ): CompassCorners | undefined {
    if (this.resizable === false) return

    const { boundingRect } = this
    if (!boundingRect.containsXy(canvasX, canvasY)) return

    // Check corners first (they take priority over edges)
    return boundingRect.findContainingCorner(
      canvasX,
      canvasY,
      LGraphNode.resizeHandleSize
    )
  }

  /**
   * returns all the info available about a property of this node.
   * @param property name of the property
   * @returns the object with all the available info
   */
  getPropertyInfo(property: string) {
    let info = null

    // there are several ways to define info about a property
    // legacy mode
    const { properties_info } = this
    if (properties_info) {
      for (const propInfo of properties_info) {
        if (propInfo.name == property) {
          info = propInfo
          break
        }
      }
    }
    // litescene mode using the constructor
    // @ts-expect-error deprecated https://github.com/Comfy-Org/litegraph.js/issues/639
    if (this.constructor[`@${property}`])
      // @ts-expect-error deprecated https://github.com/Comfy-Org/litegraph.js/issues/639
      info = this.constructor[`@${property}`]

    if (this.constructor.widgets_info?.[property])
      info = this.constructor.widgets_info[property]

    // litescene mode using the constructor
    if (!info && this.onGetPropertyInfo) {
      info = this.onGetPropertyInfo(property)
    }

    info ||= {}
    info.type ||= typeof this.properties[property]
    if (info.widget == 'combo') info.type = 'enum'

    return info
  }

  /**
   * Defines a widget inside the node, it will be rendered on top of the node, you can control lots of properties
   * @param type the widget type
   * @param name the text to show on the widget
   * @param value the default value
   * @param callback function to call when it changes (optionally, it can be the name of the property to modify)
   * @param options the object that contains special properties of this widget
   * @returns the created widget object
   */
  addWidget<
    Type extends TWidgetType,
    TValue extends WidgetTypeMap[Type]['value']
  >(
    type: Type,
    name: string,
    value: TValue,
    callback: IBaseWidget['callback'] | string | null,
    options?: IWidgetOptions | string
  ): WidgetTypeMap[Type] | IBaseWidget {
    this.widgets ||= []

    if (!options && callback && typeof callback === 'object') {
      options = callback
      callback = null
    }

    // options can be the property name
    options ||= {}
    if (typeof options === 'string') options = { property: options }

    // callback can be the property name
    if (callback && typeof callback === 'string') {
      options.property = callback
      callback = null
    }

    const w: IBaseWidget & { type: Type } = {
      // @ts-expect-error - Type casting for widget type property
      type: type.toLowerCase(),
      name: name,
      value: value,
      callback: typeof callback !== 'function' ? undefined : callback,
      options,
      y: 0
    }

    if (w.options.y !== undefined) {
      w.y = w.options.y
    }

    if (!callback && !w.options.callback && !w.options.property) {
      console.warn(
        'LiteGraph addWidget(...) without a callback or property assigned'
      )
    }
    if (type == 'combo' && !w.options.values) {
      throw "LiteGraph addWidget('combo',...) requires to pass values in options: { values:['red','blue'] }"
    }

    const widget = this.addCustomWidget(w)
    this.expandToFitContent()
    return widget
  }

  addCustomWidget<TPlainWidget extends IBaseWidget>(
    custom_widget: TPlainWidget
  ): TPlainWidget | WidgetTypeMap[TPlainWidget['type']] {
    this.widgets ||= []
    const widget = toConcreteWidget(custom_widget, this, false) ?? custom_widget
    this.widgets.push(widget)

    // Only register with store if node has a valid ID (is already in a graph).
    // If the node isn't in a graph yet (id === -1), registration happens
    // when the node is added via LGraph.add() -> node.onAdded.
    if (this.id !== -1 && isNodeBindable(widget)) {
      widget.setNodeId(this.id)
    }

    return widget
  }

  addTitleButton(options: LGraphButtonOptions): LGraphButton {
    this.title_buttons ||= []
    const button = new LGraphButton(options)
    this.title_buttons.push(button)
    return button
  }

  onTitleButtonClick(button: LGraphButton, canvas: LGraphCanvas): void {
    // Dispatch event for button click
    canvas.dispatch('litegraph:node-title-button-clicked', {
      node: this,
      button: button
    })
  }

  removeWidgetByName(name: string): void {
    const widget = this.widgets?.find((x) => x.name === name)
    if (widget) this.removeWidget(widget)
  }

  removeWidget(widget: IBaseWidget): void {
    if (!this.widgets)
      throw new Error('removeWidget called on node without widgets')

    const widgetIndex = this.widgets.indexOf(widget)
    if (widgetIndex === -1) throw new Error('Widget not found on this node')

    // Clean up slot references to prevent memory leaks
    if (this.inputs) {
      for (const input of this.inputs) {
        if (input._widget === widget) {
          input._widget = undefined
          input.widget = undefined
        }
      }
    }

    widget.onRemove?.()
    this.widgets.splice(widgetIndex, 1)
  }

  ensureWidgetRemoved(widget: IBaseWidget): void {
    try {
      this.removeWidget(widget)
    } catch (error) {
      console.error('Failed to remove widget', error)
    }
  }

  move(deltaX: number, deltaY: number): void {
    if (this.pinned) return

    // If Vue nodes mode is enabled, skip LiteGraph's direct position update
    // The layout store will handle the movement and sync back to LiteGraph
    if (LiteGraph.vueNodesMode) {
      // Vue nodes handle their own dragging through the layout store
      // This prevents the snap-back issue from conflicting position updates
      return
    }

    this.pos = [this._pos[0] + deltaX, this._pos[1] + deltaY]
  }

  /**
   * Internal method to measure the node for rendering.  Prefer {@link boundingRect} where possible.
   *
   * Populates {@link out} with the results in graph space.
   * Populates {@link _collapsed_width} with the collapsed width if the node is collapsed.
   * Adjusts for title and collapsed status, but does not call {@link onBounding}.
   * @param out `x, y, width, height` are written to this array.
   * @param ctx The canvas context to use for measuring text.
   */
  measure(out: Rect, ctx?: CanvasRenderingContext2D): void {
    const titleMode = this.title_mode
    const renderTitle =
      titleMode != TitleMode.TRANSPARENT_TITLE &&
      titleMode != TitleMode.NO_TITLE
    const titleHeight = renderTitle ? LiteGraph.NODE_TITLE_HEIGHT : 0

    out[0] = this.pos[0]
    out[1] = this.pos[1] + -titleHeight
    if (!this.flags?.collapsed) {
      out[2] = this.size[0]
      out[3] = this.size[1] + titleHeight
    } else {
      if (ctx) ctx.font = this.innerFontStyle
      this._collapsed_width = Math.min(
        this.size[0],
        ctx
          ? ctx.measureText(this.getTitle() ?? '').width +
              LiteGraph.NODE_TITLE_HEIGHT * 2
          : 0
      )
      out[2] = this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH
      out[3] = LiteGraph.NODE_TITLE_HEIGHT
    }
  }

  /**
   * returns the bounding of the object, used for rendering purposes
   * @param out {Rect?} [optional] a place to store the output, to free garbage
   * @param includeExternal {boolean?} [optional] set to true to
   * include the shadow and connection points in the bounding calculation
   * @returns the bounding box in format of [topleft_cornerx, topleft_cornery, width, height]
   */
  getBounding(out?: Rect, includeExternal?: boolean): Rect {
    out ||= [0, 0, 0, 0]

    const rect = includeExternal ? this.renderArea : this.boundingRect
    out[0] = rect[0]
    out[1] = rect[1]
    out[2] = rect[2]
    out[3] = rect[3]

    return out
  }

  /**
   * Calculates the render area of this node, populating both {@link boundingRect} and {@link renderArea}.
   * Called automatically at the start of every frame.
   */
  updateArea(ctx?: CanvasRenderingContext2D): void {
    const bounds = this._boundingRect
    this.measure(bounds, ctx)
    this.onBounding?.(bounds)

    const renderArea = this._renderArea
    renderArea.set(bounds)
    // 4 offset for collapsed node connection points
    renderArea[0] -= 4
    renderArea[1] -= 4
    // Add shadow & left offset
    renderArea[2] += 6 + 4
    // Add shadow & top offsets
    renderArea[3] += 5 + 4
  }

  /**
   * checks if a point is inside the shape of a node
   */
  isPointInside(x: number, y: number): boolean {
    if (isInRect(x, y, this.boundingRect)) return true

    for (const badge of this.badges.map(toValue).filter((b) => b.onClick)) {
      if (isInRect(x - this.pos[0], y - this.pos[1], badge.boundingRect))
        return true
    }
    return false
  }

  /**
   * Checks if the provided point is inside this node's collapse button area.
   * @param x X co-ordinate to check
   * @param y Y co-ordinate to check
   * @returns true if the x,y point is in the collapse button area, otherwise false
   */
  isPointInCollapse(x: number, y: number): boolean {
    const squareLength = LiteGraph.NODE_TITLE_HEIGHT
    return isInRectangle(
      x,
      y,
      this.pos[0],
      this.pos[1] - squareLength,
      squareLength,
      squareLength
    )
  }

  /**
   * Returns the input slot at the given position. Uses full 20 height, and approximates the label length.
   * @param pos The graph co-ordinates to check
   * @returns The input slot at the given position if found, otherwise `undefined`.
   */
  getInputOnPos(pos: Point): INodeInputSlot | undefined {
    return getNodeInputOnPos(this, pos[0], pos[1])?.input
  }

  /**
   * Returns the output slot at the given position. Uses full 20x20 box for the slot.
   * @param pos The graph co-ordinates to check
   * @returns The output slot at the given position if found, otherwise `undefined`.
   */
  getOutputOnPos(pos: Point): INodeOutputSlot | undefined {
    return getNodeOutputOnPos(this, pos[0], pos[1])?.output
  }

  /**
   * Returns the input or output slot at the given position.
   *
   * Tries {@link getNodeInputOnPos} first, then {@link getNodeOutputOnPos}.
   * @param pos The graph co-ordinates to check
   * @returns The input or output slot at the given position if found, otherwise `undefined`.
   */
  getSlotOnPos(pos: Point): INodeInputSlot | INodeOutputSlot | undefined {
    if (!isPointInRect(pos, this.boundingRect)) return

    return this.getInputOnPos(pos) ?? this.getOutputOnPos(pos)
  }

  /**
   * @deprecated Use {@link getSlotOnPos} instead.
   * checks if a point is inside a node slot, and returns info about which slot
   * @param x
   * @param y
   * @returns if found the object contains { input|output: slot object, slot: number, link_pos: [x,y] }
   */
  getSlotInPosition(x: number, y: number): IFoundSlot | null {
    // search for inputs
    const { inputs, outputs } = this

    if (inputs) {
      for (const [i, input] of inputs.entries()) {
        const pos = this.getInputPos(i)
        if (isInRectangle(x, y, pos[0] - 10, pos[1] - 10, 20, 20)) {
          return { input, slot: i, link_pos: pos }
        }
      }
    }

    if (outputs) {
      for (const [i, output] of outputs.entries()) {
        const pos = this.getOutputPos(i)
        if (isInRectangle(x, y, pos[0] - 10, pos[1] - 10, 20, 20)) {
          return { output, slot: i, link_pos: pos }
        }
      }
    }

    return null
  }

  /**
   * Gets the widget on this node at the given co-ordinates.
   * @param canvasX X co-ordinate in graph space
   * @param canvasY Y co-ordinate in graph space
   * @returns The widget found, otherwise `null`
   */
  getWidgetOnPos(
    canvasX: number,
    canvasY: number,
    includeDisabled = false
  ): IBaseWidget | undefined {
    const { widgets, pos, size } = this
    if (!widgets?.length) return

    const x = canvasX - pos[0]
    const y = canvasY - pos[1]
    const nodeWidth = size[0]

    for (const widget of widgets) {
      if (
        (widget.computedDisabled && !includeDisabled) ||
        !this.isWidgetVisible(widget)
      ) {
        continue
      }

      const h =
        widget.computedHeight ??
        widget.computeSize?.(nodeWidth)[1] ??
        LiteGraph.NODE_WIDGET_HEIGHT

      const maybeDOMWidget = widget as { margin?: number }
      const mtop = maybeDOMWidget.margin ?? -2
      const mbot = maybeDOMWidget.margin ?? 2
      const mx = maybeDOMWidget.margin ?? 6

      const w = widget.width || nodeWidth
      if (
        widget.last_y !== undefined &&
        isInRectangle(
          x,
          y,
          mx,
          widget.last_y + mtop,
          w - 2 * mx,
          h - mtop - mbot
        )
      ) {
        return widget
      }
    }
  }

  /**
   * Returns the input slot with a given name (used for dynamic slots), -1 if not found
   * @param name the name of the slot to find
   * @param returnObj if the obj itself wanted
   * @returns the slot (-1 if not found)
   */
  findInputSlot<TReturn extends false>(
    name: string,
    returnObj?: TReturn
  ): number
  findInputSlot<TReturn extends true>(
    name: string,
    returnObj?: TReturn
  ): INodeInputSlot
  findInputSlot(name: string, returnObj: boolean = false) {
    const { inputs } = this
    if (!inputs) return -1

    for (const [i, input] of inputs.entries()) {
      if (name == input.name) {
        return !returnObj ? i : input
      }
    }
    return -1
  }

  /**
   * returns the output slot with a given name (used for dynamic slots), -1 if not found
   * @param name the name of the slot to find
   * @param returnObj if the obj itself wanted
   * @returns the slot (-1 if not found)
   */
  findOutputSlot<TReturn extends false>(
    name: string,
    returnObj?: TReturn
  ): number
  findOutputSlot<TReturn extends true>(
    name: string,
    returnObj?: TReturn
  ): INodeOutputSlot
  findOutputSlot(name: string, returnObj: boolean = false) {
    const { outputs } = this
    if (!outputs) return -1

    for (const [i, output] of outputs.entries()) {
      if (name == output.name) {
        return !returnObj ? i : output
      }
    }
    return -1
  }

  /**
   * Finds the first free input slot.
   * @param optsIn
   * @returns The index of the first matching slot, the slot itself if returnObj is true, or -1 if not found.
   */
  findInputSlotFree<TReturn extends false>(
    optsIn?: FindFreeSlotOptions & { returnObj?: TReturn }
  ): number
  findInputSlotFree<TReturn extends true>(
    optsIn?: FindFreeSlotOptions & { returnObj?: TReturn }
  ): INodeInputSlot | -1
  findInputSlotFree(optsIn?: FindFreeSlotOptions) {
    return this._findFreeSlot(this.inputs, optsIn)
  }

  /**
   * Finds the first free output slot.
   * @param optsIn
   * @returns The index of the first matching slot, the slot itself if returnObj is true, or -1 if not found.
   */
  findOutputSlotFree<TReturn extends false>(
    optsIn?: FindFreeSlotOptions & { returnObj?: TReturn }
  ): number
  findOutputSlotFree<TReturn extends true>(
    optsIn?: FindFreeSlotOptions & { returnObj?: TReturn }
  ): INodeOutputSlot | -1
  findOutputSlotFree(optsIn?: FindFreeSlotOptions) {
    return this._findFreeSlot(this.outputs, optsIn)
  }

  /**
   * Finds the next free slot
   * @param slots The slots to search, i.e. this.inputs or this.outputs
   */
  private _findFreeSlot<TSlot extends INodeInputSlot | INodeOutputSlot>(
    slots: TSlot[],
    options?: FindFreeSlotOptions
  ): TSlot | number {
    const defaults = {
      returnObj: false,
      typesNotAccepted: []
    }
    const opts = Object.assign(defaults, options || {})
    const length = slots?.length
    if (!(length > 0)) return -1

    for (let i = 0; i < length; ++i) {
      const slot: TSlot & IGenericLinkOrLinks = slots[i]
      if (!slot || slot.link || slot.links?.length) continue
      if (opts.typesNotAccepted?.includes?.(slot.type)) continue
      return !opts.returnObj ? i : slot
    }
    return -1
  }

  /**
   * findSlotByType for INPUTS
   */
  findInputSlotByType<TReturn extends false>(
    type: ISlotType,
    returnObj?: TReturn,
    preferFreeSlot?: boolean,
    doNotUseOccupied?: boolean
  ): number
  findInputSlotByType<TReturn extends true>(
    type: ISlotType,
    returnObj?: TReturn,
    preferFreeSlot?: boolean,
    doNotUseOccupied?: boolean
  ): INodeInputSlot
  findInputSlotByType(
    type: ISlotType,
    returnObj?: boolean,
    preferFreeSlot?: boolean,
    doNotUseOccupied?: boolean
  ) {
    return this._findSlotByType(
      this.inputs,
      type,
      returnObj,
      preferFreeSlot,
      doNotUseOccupied
    )
  }

  /**
   * findSlotByType for OUTPUTS
   */
  findOutputSlotByType<TReturn extends false>(
    type: ISlotType,
    returnObj?: TReturn,
    preferFreeSlot?: boolean,
    doNotUseOccupied?: boolean
  ): number
  findOutputSlotByType<TReturn extends true>(
    type: ISlotType,
    returnObj?: TReturn,
    preferFreeSlot?: boolean,
    doNotUseOccupied?: boolean
  ): INodeOutputSlot
  findOutputSlotByType(
    type: ISlotType,
    returnObj?: boolean,
    preferFreeSlot?: boolean,
    doNotUseOccupied?: boolean
  ) {
    return this._findSlotByType(
      this.outputs,
      type,
      returnObj,
      preferFreeSlot,
      doNotUseOccupied
    )
  }

  /**
   * returns the output (or input) slot with a given type, -1 if not found
   * @param input use inputs instead of outputs
   * @param type the type of the slot to find
   * @param returnObj if the obj itself wanted
   * @param preferFreeSlot if we want a free slot (if not found, will return the first of the type anyway)
   * @returns the slot (-1 if not found)
   */
  findSlotByType<TSlot extends true | false, TReturn extends false>(
    input: TSlot,
    type: ISlotType,
    returnObj?: TReturn,
    preferFreeSlot?: boolean,
    doNotUseOccupied?: boolean
  ): number
  findSlotByType<TSlot extends true, TReturn extends true>(
    input: TSlot,
    type: ISlotType,
    returnObj?: TReturn,
    preferFreeSlot?: boolean,
    doNotUseOccupied?: boolean
  ): INodeInputSlot | -1
  findSlotByType<TSlot extends false, TReturn extends true>(
    input: TSlot,
    type: ISlotType,
    returnObj?: TReturn,
    preferFreeSlot?: boolean,
    doNotUseOccupied?: boolean
  ): INodeOutputSlot | -1
  findSlotByType(
    input: boolean,
    type: ISlotType,
    returnObj?: boolean,
    preferFreeSlot?: boolean,
    doNotUseOccupied?: boolean
  ): number | INodeOutputSlot | INodeInputSlot {
    return input
      ? this._findSlotByType(
          this.inputs,
          type,
          returnObj,
          preferFreeSlot,
          doNotUseOccupied
        )
      : this._findSlotByType(
          this.outputs,
          type,
          returnObj,
          preferFreeSlot,
          doNotUseOccupied
        )
  }

  /**
   * Finds a matching slot from those provided, returning the slot itself or its index in {@link slots}.
   * @param slots Slots to search (this.inputs or this.outputs)
   * @param type Type of slot to look for
   * @param returnObj If true, returns the slot itself.  Otherwise, the index.
   * @param preferFreeSlot Prefer a free slot, but if none are found, fall back to an occupied slot.
   * @param doNotUseOccupied Do not fall back to occupied slots.
   * @see {findSlotByType}
   * @see {findOutputSlotByType}
   * @see {findInputSlotByType}
   * @returns If a match is found, the slot if returnObj is true, otherwise the index.  If no matches are found, -1
   */
  private _findSlotByType<TSlot extends INodeInputSlot | INodeOutputSlot>(
    slots: TSlot[],
    type: ISlotType,
    returnObj?: boolean,
    preferFreeSlot?: boolean,
    doNotUseOccupied?: boolean
  ): TSlot | number {
    const length = slots?.length
    if (!length) return -1

    // Empty string and * match anything (type:  0)
    if (type == '' || type == '*') type = 0
    const sourceTypes = String(type).toLowerCase().split(',')

    // Run the search
    let occupiedSlot: number | TSlot | null = null
    for (let i = 0; i < length; ++i) {
      const slot: TSlot & IGenericLinkOrLinks = slots[i]
      const destTypes =
        slot.type == '0' || slot.type == '*'
          ? ['0']
          : String(slot.type).toLowerCase().split(',')

      for (const sourceType of sourceTypes) {
        // TODO: Remove _event_ entirely.
        const source = sourceType == '_event_' ? LiteGraph.EVENT : sourceType

        for (const destType of destTypes) {
          const dest = destType == '_event_' ? LiteGraph.EVENT : destType

          if (source == dest || source === '*' || dest === '*') {
            if (preferFreeSlot && (slot.links?.length || slot.link != null)) {
              // In case we can't find a free slot.
              occupiedSlot ??= returnObj ? slot : i
              continue
            }
            return returnObj ? slot : i
          }
        }
      }
    }

    return doNotUseOccupied ? -1 : (occupiedSlot ?? -1)
  }

  /**
   * Determines the slot index to connect to when attempting to connect by type.
   * @param findInputs If true, searches for an input.  Otherwise, an output.
   * @param node The node at the other end of the connection.
   * @param slotType The type of slot at the other end of the connection.
   * @param options Search restrictions to adhere to.
   * @see {connectByType}
   * @see {connectByTypeOutput}
   */
  findConnectByTypeSlot(
    findInputs: boolean,
    node: LGraphNode,
    slotType: ISlotType,
    options?: ConnectByTypeOptions
  ): number | undefined {
    // LEGACY: Old options names
    if (options && typeof options === 'object') {
      if ('firstFreeIfInputGeneralInCase' in options)
        options.wildcardToTyped = !!options.firstFreeIfInputGeneralInCase
      if ('firstFreeIfOutputGeneralInCase' in options)
        options.wildcardToTyped = !!options.firstFreeIfOutputGeneralInCase
      if ('generalTypeInCase' in options)
        options.typedToWildcard = !!options.generalTypeInCase
    }
    const optsDef: ConnectByTypeOptions = {
      createEventInCase: true,
      wildcardToTyped: true,
      typedToWildcard: true
    }
    const opts = Object.assign(optsDef, options)

    if (!this.graph) throw new NullGraphError()

    if (node && typeof node === 'number') {
      const nodeById = this.graph.getNodeById(node)
      if (!nodeById) return

      node = nodeById
    }
    const slot = node.findSlotByType(findInputs, slotType, false, true)
    if (slot >= 0 && slot !== null) return slot

    // TODO: Remove or reimpl. events.  WILL CREATE THE onTrigger IN SLOT
    if (opts.createEventInCase && slotType == LiteGraph.EVENT) {
      if (findInputs) return -1
      if (LiteGraph.do_add_triggers_slots) return node.addOnExecutedOutput()
    }

    // connect to the first general output slot if not found a specific type and
    if (opts.typedToWildcard) {
      const generalSlot = node.findSlotByType(findInputs, 0, false, true, true)
      if (generalSlot >= 0) return generalSlot
    }
    // connect to the first free input slot if not found a specific type and this output is general
    if (
      opts.wildcardToTyped &&
      (slotType == 0 || slotType == '*' || slotType == '')
    ) {
      const opt = { typesNotAccepted: [LiteGraph.EVENT] }
      const nonEventSlot = findInputs
        ? node.findInputSlotFree(opt)
        : node.findOutputSlotFree(opt)
      if (nonEventSlot >= 0) return nonEventSlot
    }
  }

  /**
   * Finds the first free output slot with any of the comma-delimited types in {@link type}.
   *
   * If no slots are free, falls back in order to:
   * - The first free wildcard slot
   * - The first occupied slot
   * - The first occupied wildcard slot
   * @param type The {@link ISlotType type} of slot to find
   * @returns The index and slot if found, otherwise `undefined`.
   */
  findOutputByType(
    type: ISlotType
  ): { index: number; slot: INodeOutputSlot } | undefined {
    return findFreeSlotOfType(
      this.outputs,
      type,
      (output) => !output.links?.length
    )
  }

  /**
   * Finds the first free input slot with any of the comma-delimited types in {@link type}.
   *
   * If no slots are free, falls back in order to:
   * - The first free wildcard slot
   * - The first occupied slot
   * - The first occupied wildcard slot
   * @param type The {@link ISlotType type} of slot to find
   * @returns The index and slot if found, otherwise `undefined`.
   */
  findInputByType(
    type: ISlotType
  ): { index: number; slot: INodeInputSlot } | undefined {
    return findFreeSlotOfType(
      this.inputs,
      type,
      (input) =>
        input.link == null || !!this.graph?.getLink(input.link)?._dragging
    )
  }

  /**
   * connect this node output to the input of another node BY TYPE
   * @param slot (could be the number of the slot or the string with the name of the slot)
   * @param target_node the target node
   * @param target_slotType the input slot type of the target node
   * @returns the link_info is created, otherwise null
   */
  connectByType(
    slot: number | string,
    target_node: LGraphNode,
    target_slotType: ISlotType,
    optsIn?: ConnectByTypeOptions
  ): LLink | null {
    const slotIndex = this.findConnectByTypeSlot(
      true,
      target_node,
      target_slotType,
      optsIn
    )
    if (slotIndex !== undefined)
      return this.connect(slot, target_node, slotIndex, optsIn?.afterRerouteId)

    // No compatible slot found - connection not possible
    return null
  }

  /**
   * connect this node input to the output of another node BY TYPE
   * @param slot (could be the number of the slot or the string with the name of the slot)
   * @param source_node the target node
   * @param source_slotType the output slot type of the target node
   * @returns the link_info is created, otherwise null
   */
  connectByTypeOutput(
    slot: number | string,
    source_node: LGraphNode,
    source_slotType: ISlotType,
    optsIn?: ConnectByTypeOptions
  ): LLink | null {
    // LEGACY: Old options names
    if (typeof optsIn === 'object') {
      if ('firstFreeIfInputGeneralInCase' in optsIn)
        optsIn.wildcardToTyped = !!optsIn.firstFreeIfInputGeneralInCase
      if ('generalTypeInCase' in optsIn)
        optsIn.typedToWildcard = !!optsIn.generalTypeInCase
    }
    const slotIndex = this.findConnectByTypeSlot(
      false,
      source_node,
      source_slotType,
      optsIn
    )
    if (slotIndex !== undefined)
      return source_node.connect(slotIndex, this, slot, optsIn?.afterRerouteId)

    console.error(
      '[connectByType]: no way to connect type:',
      source_slotType,
      'to node:',
      source_node
    )
    return null
  }

  canConnectTo(
    node: NodeLike,
    toSlot: INodeInputSlot | SubgraphIO,
    fromSlot: INodeOutputSlot | SubgraphIO
  ) {
    return (
      this.id !== node.id &&
      LiteGraph.isValidConnection(fromSlot.type, toSlot.type)
    )
  }

  /**
   * Connect an output of this node to an input of another node
   * @param slot (could be the number of the slot or the string with the name of the slot)
   * @param target_node the target node
   * @param target_slot the input slot of the target node (could be the number of the slot or the string with the name of the slot, or -1 to connect a trigger)
   * @returns the link_info is created, otherwise null
   */
  connect(
    slot: number | string,
    target_node: LGraphNode,
    target_slot: ISlotType,
    afterRerouteId?: RerouteId
  ): LLink | null {
    // Allow legacy API support for searching target_slot by string, without mutating the input variables
    let targetIndex: number | null

    const { graph, outputs } = this
    if (!graph) {
      // could be connected before adding it to a graph
      // due to link ids being associated with graphs
      console.error(
        "Connect: Error, node doesn't belong to any graph. Nodes must be added first to a graph before connecting them."
      )
      return null
    }

    // seek for the output slot
    if (typeof slot === 'string') {
      slot = this.findOutputSlot(slot)
      if (slot == -1) {
        if (LiteGraph.debug)
          console.error(`Connect: Error, no slot of name ${slot}`)
        return null
      }
    } else if (!outputs || slot >= outputs.length) {
      if (LiteGraph.debug)
        console.error('Connect: Error, slot number not found')
      return null
    }

    if (target_node && typeof target_node === 'number') {
      const nodeById = graph.getNodeById(target_node)
      if (!nodeById) throw 'target node is null'

      target_node = nodeById
    }
    if (!target_node) throw 'target node is null'

    // avoid loopback
    if (target_node == this) return null

    // you can specify the slot by name
    if (typeof target_slot === 'string') {
      targetIndex = target_node.findInputSlot(target_slot)
      if (targetIndex == -1) {
        if (LiteGraph.debug)
          console.error(`Connect: Error, no slot of name ${targetIndex}`)
        return null
      }
    } else if (target_slot === LiteGraph.EVENT) {
      // TODO: Events
      if (LiteGraph.do_add_triggers_slots) {
        target_node.changeMode(LGraphEventMode.ON_TRIGGER)
        targetIndex = target_node.findInputSlot('onTrigger')
      } else {
        return null
      }
    } else if (typeof target_slot === 'number') {
      targetIndex = target_slot
    } else {
      targetIndex = 0
    }

    // Allow target node to change slot
    if (target_node.onBeforeConnectInput) {
      // This way node can choose another slot (or make a new one?)
      const requestedIndex = target_node.onBeforeConnectInput(
        targetIndex,
        target_slot
      )
      targetIndex = typeof requestedIndex === 'number' ? requestedIndex : null
    }

    if (
      targetIndex === null ||
      !target_node.inputs ||
      targetIndex >= target_node.inputs.length
    ) {
      if (LiteGraph.debug)
        console.error('Connect: Error, slot number not found')
      return null
    }

    const input = target_node.inputs[targetIndex]
    const output = outputs[slot]

    if (!output) return null

    if (output.links?.length) {
      if (
        output.type === LiteGraph.EVENT &&
        !LiteGraph.allow_multi_output_for_events
      ) {
        graph.beforeChange()
        this.disconnectOutput(slot)
      }
    }

    const link = this.connectSlots(output, target_node, input, afterRerouteId)
    return link ?? null
  }

  /**
   * Connect two slots between two nodes
   * @param output The output slot to connect
   * @param inputNode The node that the input slot is on
   * @param input The input slot to connect
   * @param afterRerouteId The reroute ID to use for the link
   * @returns The link that was created, or null if the connection was blocked
   */
  connectSlots(
    output: INodeOutputSlot,
    inputNode: LGraphNode,
    input: INodeInputSlot,
    afterRerouteId: RerouteId | undefined
  ): LLink | null | undefined {
    const { graph } = this
    if (!graph) throw new NullGraphError()

    const layoutMutations = useLayoutMutations()

    const outputIndex = this.outputs.indexOf(output)
    if (outputIndex === -1) {
      console.warn('connectSlots: output not found')
      return
    }
    const inputIndex = inputNode.inputs.indexOf(input)
    if (inputIndex === -1) {
      console.warn('connectSlots: input not found')
      return
    }

    // check targetSlot and check connection types
    if (!LiteGraph.isValidConnection(output.type, input.type)) {
      this.setDirtyCanvas(false, true)
      return null
    }

    // Allow nodes to block connection
    if (
      inputNode.onConnectInput?.(
        inputIndex,
        output.type,
        output,
        this,
        outputIndex
      ) === false
    )
      return null
    if (
      this.onConnectOutput?.(
        outputIndex,
        input.type,
        input,
        inputNode,
        inputIndex
      ) === false
    )
      return null

    // if there is something already plugged there, disconnect
    if (inputNode.inputs[inputIndex]?.link != null) {
      graph.beforeChange()
      inputNode.disconnectInput(inputIndex, true)
    }

    const maybeCommonType =
      input.type && output.type && commonType(input.type, output.type)

    const link = new LLink(
      ++graph.state.lastLinkId,
      maybeCommonType || input.type || output.type,
      this.id,
      outputIndex,
      inputNode.id,
      inputIndex,
      afterRerouteId
    )

    // add to graph links list
    graph._links.set(link.id, link)

    // Register link in Layout Store for spatial tracking
    layoutMutations.setSource(LayoutSource.Canvas)
    layoutMutations.createLink(
      link.id,
      this.id,
      outputIndex,
      inputNode.id,
      inputIndex
    )

    // connect in output
    output.links ??= []
    output.links.push(link.id)
    // connect in input
    const targetInput = inputNode.inputs[inputIndex]
    targetInput.link = link.id
    if (targetInput.widget) {
      graph.trigger('node:slot-links:changed', {
        nodeId: inputNode.id,
        slotType: NodeSlotType.INPUT,
        slotIndex: inputIndex,
        connected: true,
        linkId: link.id
      })
    }

    // Reroutes
    const reroutes = LLink.getReroutes(graph, link)
    for (const reroute of reroutes) {
      reroute.linkIds.add(link.id)
      if (reroute.floating) reroute.floating = undefined
      reroute._dragging = undefined
    }

    // If this is the terminus of a floating link, remove it
    const lastReroute = reroutes.at(-1)
    if (lastReroute) {
      for (const linkId of lastReroute.floatingLinkIds) {
        const link = graph.floatingLinks.get(linkId)
        if (link?.parentId === lastReroute.id) {
          graph.removeFloatingLink(link)
        }
      }
    }
    graph._version++

    // link has been created now, so its updated
    this.onConnectionsChange?.(
      NodeSlotType.OUTPUT,
      outputIndex,
      true,
      link,
      output
    )

    inputNode.onConnectionsChange?.(
      NodeSlotType.INPUT,
      inputIndex,
      true,
      link,
      input
    )

    this.setDirtyCanvas(false, true)
    graph.afterChange()

    return link
  }

  connectFloatingReroute(
    pos: Point,
    slot: INodeInputSlot | INodeOutputSlot,
    afterRerouteId?: RerouteId
  ): Reroute {
    const { graph, id } = this
    if (!graph) throw new NullGraphError()

    // Assertion: It's either there or it isn't.
    const inputIndex = this.inputs.indexOf(slot as INodeInputSlot)
    const outputIndex = this.outputs.indexOf(slot as INodeOutputSlot)
    if (inputIndex === -1 && outputIndex === -1) throw new Error('Invalid slot')

    const slotType = outputIndex === -1 ? 'input' : 'output'

    const reroute = graph.setReroute({
      pos,
      parentId: afterRerouteId,
      linkIds: [],
      floating: { slotType }
    })

    const parentReroute = graph.getReroute(afterRerouteId)
    const fromLastFloatingReroute =
      parentReroute?.floating?.slotType === 'output'

    // Adding from an output, or a floating reroute that is NOT the tip of an existing floating chain
    if (afterRerouteId == null || !fromLastFloatingReroute) {
      const link = new LLink(
        -1,
        slot.type,
        outputIndex === -1 ? -1 : id,
        outputIndex,
        inputIndex === -1 ? -1 : id,
        inputIndex
      )
      link.parentId = reroute.id
      graph.addFloatingLink(link)
      return reroute
    }

    // Adding a new floating reroute from the tip of a floating chain.
    if (!parentReroute)
      throw new Error('[connectFloatingReroute] Parent reroute not found')

    const link = parentReroute.getFloatingLinks('output')?.[0]
    if (!link)
      throw new Error('[connectFloatingReroute] Floating link not found')

    reroute.floatingLinkIds.add(link.id)
    link.parentId = reroute.id
    parentReroute.floating = undefined
    return reroute
  }

  /**
   * disconnect one output to an specific node
   * @param slot (could be the number of the slot or the string with the name of the slot)
   * @param target_node the target node to which this slot is connected [Optional,
   * if not target_node is specified all nodes will be disconnected]
   * @returns if it was disconnected successfully
   */
  disconnectOutput(slot: string | number, target_node?: LGraphNode): boolean {
    if (typeof slot === 'string') {
      slot = this.findOutputSlot(slot)
      if (slot == -1) {
        if (LiteGraph.debug)
          console.error(`Connect: Error, no slot of name ${slot}`)
        return false
      }
    } else if (!this.outputs || slot >= this.outputs.length) {
      if (LiteGraph.debug)
        console.error('Connect: Error, slot number not found')
      return false
    }

    // get output slot
    const output = this.outputs[slot]
    if (!output) return false

    if (output._floatingLinks) {
      for (const link of output._floatingLinks) {
        if (link.hasOrigin(this.id, slot)) {
          this.graph?.removeFloatingLink(link)
        }
      }
    }

    if (!output.links || output.links.length == 0) return false
    const { links } = output

    // one of the output links in this slot
    const graph = this.graph
    if (!graph) throw new NullGraphError()

    if (target_node) {
      const target =
        typeof target_node === 'number'
          ? graph.getNodeById(target_node)
          : target_node
      if (!target) throw 'Target Node not found'

      for (const [i, link_id] of links.entries()) {
        const link_info = graph._links.get(link_id)
        if (link_info?.target_id != target.id) continue

        // is the link we are searching for...
        // remove here
        links.splice(i, 1)
        const input = target.inputs[link_info.target_slot]
        // remove there
        input.link = null
        if (input.widget) {
          graph.trigger('node:slot-links:changed', {
            nodeId: target.id,
            slotType: NodeSlotType.INPUT,
            slotIndex: link_info.target_slot,
            connected: false,
            linkId: link_info.id
          })
        }

        // remove the link from the links pool
        link_info.disconnect(graph, 'input')
        graph._version++

        // link_info hasn't been modified so its ok
        target.onConnectionsChange?.(
          NodeSlotType.INPUT,
          link_info.target_slot,
          false,
          link_info,
          input
        )
        this.onConnectionsChange?.(
          NodeSlotType.OUTPUT,
          slot,
          false,
          link_info,
          output
        )

        break
      }
    } else {
      // all the links in this output slot
      for (const link_id of links) {
        const link_info = graph._links.get(link_id)
        if (!link_info) continue
        if (
          link_info.target_id === SUBGRAPH_OUTPUT_ID &&
          graph instanceof Subgraph
        ) {
          const targetSlot = graph.outputNode.slots[link_info.target_slot]
          if (targetSlot) {
            targetSlot.linkIds.length = 0
          } else {
            console.error('Missing subgraphOutput slot when disconnecting link')
          }
        }

        const target = graph.getNodeById(link_info.target_id)
        graph._version++

        if (target) {
          const input = target.inputs[link_info.target_slot]
          // remove other side link
          input.link = null
          if (input.widget) {
            graph.trigger('node:slot-links:changed', {
              nodeId: target.id,
              slotType: NodeSlotType.INPUT,
              slotIndex: link_info.target_slot,
              connected: false,
              linkId: link_info.id
            })
          }

          // link_info hasn't been modified so its ok
          target.onConnectionsChange?.(
            NodeSlotType.INPUT,
            link_info.target_slot,
            false,
            link_info,
            input
          )
        }
        // remove the link from the links pool
        link_info.disconnect(graph, 'input')

        this.onConnectionsChange?.(
          NodeSlotType.OUTPUT,
          slot,
          false,
          link_info,
          output
        )
      }
      output.links = null
    }

    this.setDirtyCanvas(false, true)
    return true
  }

  /**
   * Disconnect one input
   * @param slot Input slot index, or the name of the slot
   * @param keepReroutes If `true`, reroutes will not be garbage collected.
   * @returns true if disconnected successfully or already disconnected, otherwise false
   */
  disconnectInput(slot: number | string, keepReroutes?: boolean): boolean {
    // Allow search by string
    if (typeof slot === 'string') {
      slot = this.findInputSlot(slot)
      if (slot == -1) {
        if (LiteGraph.debug)
          console.error(`Connect: Error, no slot of name ${slot}`)
        return false
      }
    } else if (!this.inputs || slot >= this.inputs.length) {
      if (LiteGraph.debug) {
        console.error('Connect: Error, slot number not found')
      }
      return false
    }

    const input = this.inputs[slot]
    if (!input) {
      console.error('disconnectInput: input not found', slot, this.inputs)
      return false
    }

    const { graph } = this
    if (!graph) throw new NullGraphError()

    // Break floating links
    if (input._floatingLinks?.size) {
      for (const link of input._floatingLinks) {
        graph.removeFloatingLink(link)
      }
    }

    const link_id = this.inputs[slot].link
    if (link_id != null) {
      this.inputs[slot].link = null
      if (input.widget) {
        graph.trigger('node:slot-links:changed', {
          nodeId: this.id,
          slotType: NodeSlotType.INPUT,
          slotIndex: slot,
          connected: false,
          linkId: link_id
        })
      }

      // remove other side
      const link_info = graph._links.get(link_id)
      if (link_info) {
        // Let SubgraphInput do the disconnect.
        if (link_info.origin_id === -10 && 'inputNode' in graph) {
          graph.inputNode._disconnectNodeInput(this, input, link_info)
          return true
        }

        const target_node = graph.getNodeById(link_info.origin_id)
        if (!target_node) {
          console.error(
            'disconnectInput: output not found',
            link_info.origin_slot
          )
          return false
        }

        const output = target_node.outputs[link_info.origin_slot]
        if (!output?.links?.length) {
          // Output not found - may have been removed
          return false
        }

        // search in the inputs list for this link
        let i = 0
        for (const l = output.links.length; i < l; i++) {
          if (output.links[i] == link_id) {
            output.links.splice(i, 1)
            break
          }
        }

        link_info.disconnect(graph, keepReroutes ? 'output' : undefined)
        if (graph) graph._version++

        this.onConnectionsChange?.(
          NodeSlotType.INPUT,
          slot,
          false,
          link_info,
          input
        )
        target_node.onConnectionsChange?.(
          NodeSlotType.OUTPUT,
          i,
          false,
          link_info,
          output
        )
      }
    }

    this.setDirtyCanvas(false, true)
    return true
  }

  /**
   * @deprecated Use {@link getInputPos} or {@link getOutputPos} instead.
   * returns the center of a connection point in canvas coords
   * @param is_input true if if a input slot, false if it is an output
   * @param slot_number (could be the number of the slot or the string with the name of the slot)
   * @param out [optional] a place to store the output, to free garbage
   * @returns the position
   */
  getConnectionPos(is_input: boolean, slot_number: number, out?: Point): Point {
    out ||= [0, 0]

    const {
      pos: [nodeX, nodeY],
      inputs,
      outputs
    } = this

    if (this.flags.collapsed) {
      const w = this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH
      out[0] = is_input ? nodeX : nodeX + w
      out[1] = nodeY - LiteGraph.NODE_TITLE_HEIGHT * 0.5
      return out
    }

    // weird feature that never got finished
    if (is_input && slot_number == -1) {
      out[0] = nodeX + LiteGraph.NODE_TITLE_HEIGHT * 0.5
      out[1] = nodeY + LiteGraph.NODE_TITLE_HEIGHT * 0.5
      return out
    }

    // hard-coded pos
    const inputPos = inputs?.[slot_number]?.pos
    const outputPos = outputs?.[slot_number]?.pos

    if (is_input && inputPos) {
      out[0] = nodeX + inputPos[0]
      out[1] = nodeY + inputPos[1]
      return out
    } else if (!is_input && outputPos) {
      out[0] = nodeX + outputPos[0]
      out[1] = nodeY + outputPos[1]
      return out
    }

    // default vertical slots
    const offset = LiteGraph.NODE_SLOT_HEIGHT * 0.5
    const slotIndex = is_input
      ? this._defaultVerticalInputs.indexOf(this.inputs[slot_number])
      : this._defaultVerticalOutputs.indexOf(this.outputs[slot_number])

    out[0] = is_input ? nodeX + offset : nodeX + this.size[0] + 1 - offset
    out[1] =
      nodeY +
      (slotIndex + 0.7) * LiteGraph.NODE_SLOT_HEIGHT +
      (this.constructor.slot_start_y || 0)
    return out
  }

  /**
   * @internal The inputs that are not positioned with absolute coordinates.
   */
  private get _defaultVerticalInputs() {
    return this.inputs.filter(
      (slot) => !slot.pos && !(this.widgets?.length && isWidgetInputSlot(slot))
    )
  }

  /**
   * @internal The outputs that are not positioned with absolute coordinates.
   */
  private get _defaultVerticalOutputs() {
    return this.outputs.filter((slot: INodeOutputSlot) => !slot.pos)
  }

  /**
   * Get the context needed for slot position calculations
   * @internal
   */
  private _getSlotPositionContext(): SlotPositionContext {
    return {
      nodeX: this.pos[0],
      nodeY: this.pos[1],
      nodeWidth: this.size[0],
      nodeHeight: this.size[1],
      collapsed: this.flags.collapsed ?? false,
      collapsedWidth: this._collapsed_width,
      slotStartY: this.constructor.slot_start_y,
      inputs: this.inputs,
      outputs: this.outputs,
      widgets: this.widgets
    }
  }

  /**
   * Gets the position of an input slot, in graph co-ordinates.
   *
   * This method is preferred over the legacy {@link getConnectionPos} method.
   * @param slot Input slot index
   * @returns Position of the input slot
   */
  getInputPos(slot: number): Point {
    return getSlotPosition(this, slot, true)
  }

  /**
   * Gets the position of an input slot, in graph co-ordinates.
   * @param input The actual node input object
   * @returns Position of the centre of the input slot in graph co-ordinates.
   */
  getInputSlotPos(input: INodeInputSlot): Point {
    return calculateInputSlotPosFromSlot(this._getSlotPositionContext(), input)
  }

  /**
   * Gets the position of an output slot, in graph co-ordinates.
   *
   * This method is preferred over the legacy {@link getConnectionPos} method.
   * @param outputSlotIndex Output slot index
   * @returns Position of the output slot
   */
  getOutputPos(outputSlotIndex: number): Point {
    return getSlotPosition(this, outputSlotIndex, false)
  }

  /**
   * Get slot position using layout tree if available, fallback to node's position * Unified implementation used by both LitegraphLinkAdapter and useLinkLayoutSync
   * @param slotIndex The slot index
   * @param isInput Whether this is an input slot
   * @returns Position of the slot center in graph coordinates
   */
  getSlotPosition(slotIndex: number, isInput: boolean): Point {
    return getSlotPosition(this, slotIndex, isInput)
  }

  /** @inheritdoc */
  snapToGrid(snapTo: number): boolean {
    return this.pinned ? false : snapPoint(this.pos, snapTo)
  }

  /** @see {@link snapToGrid} */
  alignToGrid(): void {
    this.snapToGrid(LiteGraph.CANVAS_GRID_SIZE)
  }

  /* Console output */
  trace(msg: string): void {
    this.console ||= []
    this.console.push(msg)
    // @ts-expect-error deprecated
    if (this.console.length > LGraphNode.MAX_CONSOLE) this.console.shift()
  }

  /* Forces to redraw or the main canvas (LGraphNode) or the bg canvas (links) */
  setDirtyCanvas(dirty_foreground: boolean, dirty_background?: boolean): void {
    this.graph?.canvasAction((c) =>
      c.setDirty(dirty_foreground, dirty_background)
    )
  }

  loadImage(url: string): HTMLImageElement {
    interface AsyncImageElement extends HTMLImageElement {
      ready?: boolean
    }

    const img: AsyncImageElement = new Image()
    img.src = LiteGraph.node_images_path + url
    img.ready = false

    const dirty = () => this.setDirtyCanvas(true)
    img.addEventListener('load', function (this: AsyncImageElement) {
      this.ready = true
      dirty()
    })
    return img
  }

  /**
   * Allows to get onMouseMove and onMouseUp events even if the mouse is out of focus
   * @deprecated Use {@link LGraphCanvas.pointer} instead.
   */
  captureInput(v: boolean): void {
    warnDeprecated(
      '[DEPRECATED] captureInput will be removed in a future version. Please use LGraphCanvas.pointer (CanvasPointer) instead.'
    )
    if (!this.graph || !this.graph.list_of_graphcanvas) return

    const list = this.graph.list_of_graphcanvas

    for (const c of list) {
      // releasing somebody elses capture?!
      if (!v && c.node_capturing_input != this) continue

      // change
      c.node_capturing_input = v ? this : null
    }
  }

  get collapsed() {
    return !!this.flags.collapsed
  }

  get collapsible() {
    return !this.pinned && this.constructor.collapsable !== false
  }

  /**
   * Toggle node collapse (makes it smaller on the canvas)
   */
  collapse(force?: boolean): void {
    if (!this.collapsible && !force) return
    if (!this.graph) throw new NullGraphError()
    this.graph._version++
    this.flags.collapsed = !this.flags.collapsed
    this.setDirtyCanvas(true, true)
  }

  /**
   * Toggles advanced mode of the node, showing advanced widgets
   */
  toggleAdvanced() {
    if (!this.hasAdvancedWidgets()) return
    if (!this.graph) throw new NullGraphError()
    this.graph._version++
    this.showAdvanced = !this.showAdvanced
    this.expandToFitContent()
    this.setDirtyCanvas(true, true)
  }

  get pinned() {
    return !!this.flags.pinned
  }

  /**
   * Prevents the node being accidentally moved or resized by mouse interaction.
   * Toggles pinned state if no value is provided.
   */
  pin(v?: boolean): void {
    if (!this.graph) throw new NullGraphError()

    this.graph._version++
    this.flags.pinned = v ?? !this.flags.pinned
    this.resizable = !this.pinned
    if (!this.pinned) this.flags.pinned = undefined
  }

  unpin(): void {
    this.pin(false)
  }

  localToScreen(x: number, y: number, dragAndScale: DragAndScale): Point {
    return [
      (x + this.pos[0]) * dragAndScale.scale + dragAndScale.offset[0],
      (y + this.pos[1]) * dragAndScale.scale + dragAndScale.offset[1]
    ]
  }

  get width() {
    return this.collapsed
      ? this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH
      : this.size[0]
  }

  /**
   * Returns the height of the node, including the title bar.
   */
  get height() {
    return LiteGraph.NODE_TITLE_HEIGHT + this.bodyHeight
  }

  /**
   * Returns the height of the node, excluding the title bar.
   */
  get bodyHeight() {
    return this.collapsed ? 0 : this.size[1]
  }

  drawBadges(ctx: CanvasRenderingContext2D, { gap = 2 } = {}): void {
    const badgeInstances = this.badges.map((badge) =>
      badge instanceof LGraphBadge ? badge : badge()
    )
    const isLeftAligned = this.badgePosition === BadgePosition.TopLeft

    let currentX = isLeftAligned
      ? 0
      : this.width -
        badgeInstances.reduce(
          (acc, badge) => acc + badge.getWidth(ctx) + gap,
          0
        )
    const y = -(LiteGraph.NODE_TITLE_HEIGHT + gap)

    for (const badge of badgeInstances) {
      badge.draw(ctx, currentX, y - badge.height)
      currentX += badge.getWidth(ctx) + gap
    }
  }

  /**
   * Renders the node's title bar background
   */
  drawTitleBarBackground(
    ctx: CanvasRenderingContext2D,
    {
      scale,
      title_height = LiteGraph.NODE_TITLE_HEIGHT,
      low_quality = false
    }: DrawTitleOptions
  ): void {
    const fgcolor = this.renderingColor
    const shape = this.renderingShape
    const size = this.renderingSize

    if (this.onDrawTitleBar) {
      this.onDrawTitleBar(ctx, title_height, size, scale, fgcolor)
      return
    }

    if (this.title_mode === TitleMode.TRANSPARENT_TITLE) {
      return
    }

    if (this.collapsed) {
      ctx.shadowColor = LiteGraph.DEFAULT_SHADOW_COLOR
    }

    ctx.fillStyle = this.constructor.title_color || fgcolor
    ctx.beginPath()

    if (shape == RenderShape.BOX || low_quality) {
      ctx.rect(0, -title_height, size[0], title_height)
    } else if (shape == RenderShape.ROUND || shape == RenderShape.CARD) {
      ctx.roundRect(
        0,
        -title_height,
        size[0],
        title_height,
        this.collapsed
          ? [LiteGraph.ROUND_RADIUS]
          : [LiteGraph.ROUND_RADIUS, LiteGraph.ROUND_RADIUS, 0, 0]
      )
    }
    ctx.fill()
    ctx.shadowColor = 'transparent'
  }

  /**
   * Renders the node's title box, i.e. the dot in front of the title text that
   * when clicked toggles the node's collapsed state. The term `title box` comes
   * from the original LiteGraph implementation.
   */
  drawTitleBox(
    ctx: CanvasRenderingContext2D,
    {
      scale,
      low_quality = false,
      title_height = LiteGraph.NODE_TITLE_HEIGHT,
      box_size = 10
    }: DrawTitleBoxOptions
  ): void {
    const size = this.renderingSize
    const shape = this.renderingShape

    if (this.onDrawTitleBox) {
      this.onDrawTitleBox(ctx, title_height, size, scale)
      return
    }

    if (
      [RenderShape.ROUND, RenderShape.CIRCLE, RenderShape.CARD].includes(shape)
    ) {
      if (low_quality) {
        ctx.fillStyle = 'black'
        ctx.beginPath()
        ctx.arc(
          title_height * 0.5,
          title_height * -0.5,
          box_size * 0.5 + 1,
          0,
          Math.PI * 2
        )
        ctx.fill()
      }

      ctx.fillStyle = this.renderingBoxColor
      if (low_quality) {
        ctx.fillRect(
          title_height * 0.5 - box_size * 0.5,
          title_height * -0.5 - box_size * 0.5,
          box_size,
          box_size
        )
      } else {
        ctx.beginPath()
        ctx.arc(
          title_height * 0.5,
          title_height * -0.5,
          box_size * 0.5,
          0,
          Math.PI * 2
        )
        ctx.fill()
      }
    } else {
      if (low_quality) {
        ctx.fillStyle = 'black'
        ctx.fillRect(
          (title_height - box_size) * 0.5 - 1,
          (title_height + box_size) * -0.5 - 1,
          box_size + 2,
          box_size + 2
        )
      }
      ctx.fillStyle = this.renderingBoxColor
      ctx.fillRect(
        (title_height - box_size) * 0.5,
        (title_height + box_size) * -0.5,
        box_size,
        box_size
      )
    }
  }

  /**
   * Renders the node's title text.
   */
  drawTitleText(
    ctx: CanvasRenderingContext2D,
    {
      scale,
      default_title_color,
      low_quality = false,
      title_height = LiteGraph.NODE_TITLE_HEIGHT
    }: DrawTitleTextOptions
  ): void {
    const size = this.renderingSize
    const selected = this.selected

    if (this.onDrawTitleText) {
      this.onDrawTitleText(
        ctx,
        title_height,
        size,
        scale,
        this.titleFontStyle,
        selected
      )
      return
    }

    // Don't render title text if low quality
    if (low_quality) {
      return
    }

    ctx.font = this.titleFontStyle
    const rawTitle = this.getTitle() ?? `❌ ${this.type}`
    const title = String(rawTitle) + (this.pinned ? '📌' : '')
    if (title) {
      if (selected) {
        ctx.fillStyle = LiteGraph.NODE_SELECTED_TITLE_COLOR
      } else {
        ctx.fillStyle = this.constructor.title_text_color || default_title_color
      }

      // Calculate available width for title
      let availableWidth = size[0] - title_height * 2 // Basic margins

      // Subtract space for title buttons
      if (this.title_buttons?.length > 0) {
        let buttonsWidth = 0
        const savedFont = ctx.font // Save current font
        for (const button of this.title_buttons) {
          if (button.visible) {
            buttonsWidth += button.getWidth(ctx) + 2 // button width + gap
          }
        }
        ctx.font = savedFont // Restore font after button measurements
        if (buttonsWidth > 0) {
          buttonsWidth -= 20 // Reduce by empty padding
          availableWidth -= buttonsWidth
        }
      }

      // Truncate title if needed
      let displayTitle = title

      if (this.collapsed) {
        // For collapsed nodes, limit to 20 chars as before
        displayTitle = title.substr(0, 20)
      } else if (availableWidth > 0) {
        // For regular nodes, truncate based on available width
        displayTitle = truncateText(ctx, title, availableWidth)
      }

      ctx.textAlign = 'left'
      ctx.fillText(
        displayTitle,
        title_height,
        LiteGraph.NODE_TITLE_TEXT_Y - title_height
      )
    }
  }

  /**
   * Attempts to gracefully bypass this node in all of its connections by reconnecting all links.
   *
   * Each input is checked against each output.  This is done on a matching index basis, i.e. input 3 -> output 3.
   * If there are any input links remaining,
   * and {@link flags}.{@link INodeFlags.keepAllLinksOnBypass keepAllLinksOnBypass} is `true`,
   * each input will check for outputs that match, and take the first one that matches
   * `true`: Try the index matching first, then every input to every output.
   * `false`: Only matches indexes, e.g. input 3 to output 3.
   *
   * If {@link flags}.{@link INodeFlags.keepAllLinksOnBypass keepAllLinksOnBypass} is `undefined`, it will fall back to
   * the static {@link keepAllLinksOnBypass}.
   * @returns `true` if any new links were established, otherwise `false`.
   * @todo Decision: Change API to return array of new links instead?
   */
  connectInputToOutput(): boolean | undefined {
    const { inputs, outputs, graph } = this
    if (!inputs || !outputs) return
    if (!graph) throw new NullGraphError()

    const { _links } = graph
    let madeAnyConnections = false

    // First pass: only match exactly index-to-index
    for (const [index, input] of inputs.entries()) {
      if (input.link == null) continue

      const output = outputs[index]
      if (!output || !LiteGraph.isValidConnection(input.type, output.type))
        continue

      const inLink = _links.get(input.link)
      if (!inLink) continue
      const inNode = graph.getNodeById(inLink?.origin_id)
      if (!inNode) continue

      bypassAllLinks(output, inNode, inLink, graph)
    }
    // Configured to only use index-to-index matching
    if (!(this.flags.keepAllLinksOnBypass ?? LGraphNode.keepAllLinksOnBypass))
      return madeAnyConnections

    // Second pass: match any remaining links
    for (const input of inputs) {
      if (input.link == null) continue

      const inLink = _links.get(input.link)
      if (!inLink) continue
      const inNode = graph.getNodeById(inLink?.origin_id)
      if (!inNode) continue

      for (const output of outputs) {
        if (!LiteGraph.isValidConnection(input.type, output.type)) continue

        bypassAllLinks(output, inNode, inLink, graph)
        break
      }
    }
    return madeAnyConnections

    function bypassAllLinks(
      output: INodeOutputSlot,
      inNode: LGraphNode,
      inLink: LLink,
      graph: LGraph
    ) {
      const outLinks = output.links
        ?.map((x) => _links.get(x))
        .filter((x) => !!x)
      if (!outLinks?.length) return

      for (const outLink of outLinks) {
        const outNode = graph.getNodeById(outLink.target_id)
        if (!outNode) continue

        const result = inNode.connect(
          inLink.origin_slot,
          outNode,
          outLink.target_slot,
          inLink.parentId
        )
        madeAnyConnections ||= !!result
      }
    }
  }

  /**
   * Returns `true` if the widget is visible, otherwise `false`.
   */
  isWidgetVisible(widget: IBaseWidget): boolean {
    const isHidden =
      this.collapsed || widget.hidden || (widget.advanced && !this.showAdvanced)
    return !isHidden
  }

  /**
   * Returns all widgets that should participate in layout calculations.
   * Filters out hidden widgets only (not collapsed/advanced).
   */
  getLayoutWidgets(): IBaseWidget[] {
    return this.widgets?.filter((w) => !w.hidden) ?? []
  }

  /**
   * Returns `true` if the node has any advanced widgets.
   */
  hasAdvancedWidgets(): boolean {
    return this.widgets?.some((w) => w.advanced) ?? false
  }

  updateComputedDisabled() {
    if (!this.widgets) return
    for (const widget of this.widgets)
      widget.computedDisabled =
        widget.disabled || this.getSlotFromWidget(widget)?.link != null
  }

  drawWidgets(
    ctx: CanvasRenderingContext2D,
    { lowQuality = false, editorAlpha = 1 }: DrawWidgetsOptions
  ): void {
    if (!this.widgets) return

    const nodeWidth = this.size[0]
    const { widgets } = this
    const H = LiteGraph.NODE_WIDGET_HEIGHT
    const showText = !lowQuality
    ctx.save()
    ctx.globalAlpha = editorAlpha

    this.updateComputedDisabled()
    for (const widget of widgets) {
      if (!this.isWidgetVisible(widget)) continue

      const { y } = widget
      const outlineColour = widget.advanced
        ? LiteGraph.WIDGET_ADVANCED_OUTLINE_COLOR
        : LiteGraph.WIDGET_OUTLINE_COLOR

      widget.last_y = y

      ctx.strokeStyle = outlineColour
      ctx.fillStyle = '#222'
      ctx.textAlign = 'left'
      if (widget.computedDisabled) ctx.globalAlpha *= 0.5
      const width = widget.width || nodeWidth

      if (typeof widget.draw === 'function') {
        widget.draw(ctx, this, width, y, H, lowQuality)
      } else {
        toConcreteWidget(widget, this, false)?.drawWidget(ctx, {
          width,
          showText
        })
      }
      ctx.globalAlpha = editorAlpha
    }
    ctx.restore()
  }

  /**
   * When {@link LGraphNode.collapsed} is `true`, this method draws the node's collapsed slots.
   */
  drawCollapsedSlots(ctx: CanvasRenderingContext2D): void {
    // Render the first connected slot only.
    for (const slot of this._concreteInputs) {
      if (slot.link != null) {
        slot.drawCollapsed(ctx)
        break
      }
    }
    for (const slot of this._concreteOutputs) {
      if (slot.links?.length) {
        slot.drawCollapsed(ctx)
        break
      }
    }
  }

  get slots(): (INodeInputSlot | INodeOutputSlot)[] {
    return [...this.inputs, ...this.outputs]
  }

  private _measureSlot(
    slot: NodeInputSlot | NodeOutputSlot,
    slotIndex: number,
    isInput: boolean
  ): void {
    const pos = isInput
      ? this.getInputPos(slotIndex)
      : this.getOutputPos(slotIndex)

    slot.boundingRect[0] = pos[0] - LiteGraph.NODE_SLOT_HEIGHT * 0.5
    slot.boundingRect[1] = pos[1] - LiteGraph.NODE_SLOT_HEIGHT * 0.5
    slot.boundingRect[2] = slot.isWidgetInputSlot
      ? BaseWidget.margin
      : LiteGraph.NODE_SLOT_HEIGHT
    slot.boundingRect[3] = LiteGraph.NODE_SLOT_HEIGHT
  }

  private _measureSlots(): ReadOnlyRect | null {
    const slots: (NodeInputSlot | NodeOutputSlot)[] = []

    for (const [slotIndex, slot] of this._concreteInputs.entries()) {
      // Unrecognized nodes (Nodes with error) has inputs but no widgets. Treat
      // converted inputs as normal inputs.
      /** Widget input slots are handled in {@link layoutWidgetInputSlots} */
      if (this.widgets?.length && isWidgetInputSlot(slot)) continue

      this._measureSlot(slot, slotIndex, true)
      slots.push(slot)
    }
    for (const [slotIndex, slot] of this._concreteOutputs.entries()) {
      this._measureSlot(slot, slotIndex, false)
      slots.push(slot)
    }

    return slots.length ? createBounds(slots, 0) : null
  }

  private _getMouseOverSlot(slot: INodeSlot): INodeSlot | null {
    const isInput = isINodeInputSlot(slot)
    const mouseOverId = this.mouseOver?.[isInput ? 'inputId' : 'outputId'] ?? -1
    if (mouseOverId === -1) {
      return null
    }
    return isInput ? this.inputs[mouseOverId] : this.outputs[mouseOverId]
  }

  private _isMouseOverSlot(slot: INodeSlot): boolean {
    return this._getMouseOverSlot(slot) === slot
  }

  private _isMouseOverWidget(widget: IBaseWidget | undefined): boolean {
    if (!widget) return false
    return this.mouseOver?.overWidget === widget
  }

  /**
   * Returns the input slot that is associated with the given widget.
   */
  getSlotFromWidget(
    widget: IBaseWidget | undefined
  ): INodeInputSlot | undefined {
    if (widget)
      return this.inputs.find(
        (slot) => isWidgetInputSlot(slot) && slot.widget.name === widget.name
      )
  }

  /**
   * Returns the widget that is associated with the given input slot.
   */
  getWidgetFromSlot(slot: INodeInputSlot): IBaseWidget | undefined {
    if (!isWidgetInputSlot(slot)) return
    return this.widgets?.find((w) => w.name === slot.widget.name)
  }

  /**
   * Draws the node's input and output slots.
   */
  drawSlots(
    ctx: CanvasRenderingContext2D,
    { fromSlot, colorContext, editorAlpha, lowQuality }: DrawSlotsOptions
  ) {
    for (const slot of [...this._concreteInputs, ...this._concreteOutputs]) {
      const isValidTarget = fromSlot && slot.isValidTarget(fromSlot)
      const isMouseOverSlot = this._isMouseOverSlot(slot)

      // change opacity of incompatible slots when dragging a connection
      const isValid = !fromSlot || isValidTarget
      const highlight = isValid && isMouseOverSlot

      // Show slot if it's not a widget input slot
      // or if it's a widget input slot and satisfies one of the following:
      // - the mouse is over the widget
      // - the slot is valid during link drop
      // - the slot is connected
      if (
        isMouseOverSlot ||
        isValidTarget ||
        !slot.isWidgetInputSlot ||
        this._isMouseOverWidget(this.getWidgetFromSlot(slot)) ||
        slot.isConnected ||
        slot.alwaysVisible
      ) {
        ctx.globalAlpha = isValid ? editorAlpha : 0.4 * editorAlpha
        slot.draw(ctx, {
          colorContext,
          lowQuality,
          highlight
        })
      }
    }
  }

  /**
   * Arranges the node's widgets vertically.
   * Sets following properties on each widget:
   * -  {@link IBaseWidget.computedHeight}
   * -  {@link IBaseWidget.y}
   * @param widgetStartY The y-coordinate of the first widget
   */
  private _arrangeWidgets(widgetStartY: number): void {
    if (!this.widgets || !this.widgets.length) return

    const bodyHeight = this.bodyHeight
    const startY =
      this.widgets_start_y ?? (this.widgets_up ? 0 : widgetStartY) + 2

    let freeSpace = bodyHeight - startY

    // Collect fixed height widgets first
    let fixedWidgetHeight = 0
    const growableWidgets: {
      minHeight: number
      prefHeight?: number
      w: IBaseWidget
    }[] = []

    const visibleWidgets = this.getLayoutWidgets()

    for (const w of visibleWidgets) {
      if (w.computeSize) {
        const height = w.computeSize()[1] + 4
        w.computedHeight = height
        fixedWidgetHeight += height
      } else if (w.computeLayoutSize) {
        const { minHeight, maxHeight } = w.computeLayoutSize(this)
        growableWidgets.push({
          minHeight,
          prefHeight: maxHeight,
          w
        })
      } else {
        const height = LiteGraph.NODE_WIDGET_HEIGHT + 4
        w.computedHeight = height
        fixedWidgetHeight += height
      }
    }

    // Calculate remaining space for DOM widgets
    freeSpace -= fixedWidgetHeight
    this.freeWidgetSpace = freeSpace

    // Prepare space requests for distribution
    const spaceRequests = growableWidgets.map((d) => ({
      minSize: d.minHeight,
      maxSize: d.prefHeight
    }))

    // Distribute space among DOM widgets
    const allocations = distributeSpace(Math.max(0, freeSpace), spaceRequests)

    // Apply computed heights
    for (const [i, d] of growableWidgets.entries()) {
      d.w.computedHeight = allocations[i]
    }

    // Position widgets
    let y = startY
    for (const w of visibleWidgets) {
      w.y = y
      y += w.computedHeight ?? 0
    }

    if (!this.graph) throw new NullGraphError()

    // Grow the node if necessary.
    // Ref: https://github.com/Comfy-Org/ComfyUI_frontend/issues/2652
    // TODO: Move the layout logic before drawing of the node shape, so we don't
    // need to trigger extra round of rendering.
    // In Vue mode, the DOM is the source of truth for node sizing — the
    // ResizeObserver feeds measurements back to the layout store.  Allowing
    // LiteGraph to also call setSize() here creates an infinite feedback loop
    // (LG grows node → CSS min-height increases → textarea fills extra space →
    // ResizeObserver reports larger size → LG grows node again).
    if (!LiteGraph.vueNodesMode && y > bodyHeight) {
      this.setSize([this.size[0], y])
      this.graph.setDirtyCanvas(false, true)
    }
  }

  /**
   * Arranges the layout of the node's widget input slots.
   */
  private _arrangeWidgetInputSlots(): void {
    if (!this.widgets) return

    const slotByWidgetName = new Map<
      string,
      INodeInputSlot & { index: number }
    >()

    for (const [i, slot] of this.inputs.entries()) {
      if (!isWidgetInputSlot(slot)) continue

      slotByWidgetName.set(slot.widget.name, { ...slot, index: i })
    }
    if (!slotByWidgetName.size) return

    // Only set custom pos if not using Vue positioning
    // Vue positioning calculates widget slot positions dynamically
    if (!LiteGraph.vueNodesMode) {
      for (const widget of this.widgets) {
        const slot = slotByWidgetName.get(widget.name)
        if (!slot) continue

        const actualSlot = this._concreteInputs[slot.index]
        const offset = LiteGraph.NODE_SLOT_HEIGHT * 0.5
        actualSlot.pos = [offset, widget.y + offset]
        this._measureSlot(actualSlot, slot.index, true)
      }
    } else {
      // For Vue positioning, just measure the slots without setting pos
      for (const widget of this.widgets) {
        const slot = slotByWidgetName.get(widget.name)
        if (!slot) continue

        this._measureSlot(this._concreteInputs[slot.index], slot.index, true)
      }
    }
  }

  /**
   * @internal Sets the internal concrete slot arrays, ensuring they are instances of
   * {@link NodeInputSlot} or {@link NodeOutputSlot}.
   *
   * A temporary workaround until duck-typed inputs and outputs
   * have been removed from the ecosystem.
   */
  _setConcreteSlots(): void {
    this._concreteInputs = this.inputs.map((slot) =>
      toClass(NodeInputSlot, slot, this)
    )
    this._concreteOutputs = this.outputs.map((slot) =>
      toClass(NodeOutputSlot, slot, this)
    )
  }

  /**
   * Arranges node elements in preparation for rendering (slots & widgets).
   */
  arrange(): void {
    const slotsBounds = this._measureSlots()
    const widgetStartY = slotsBounds
      ? slotsBounds[1] + slotsBounds[3] - this.pos[1]
      : 0
    this._arrangeWidgets(widgetStartY)
    this._arrangeWidgetInputSlots()
  }

  /**
   * Draws a progress bar on the node.
   * @param ctx The canvas context to draw on
   */
  drawProgressBar(ctx: CanvasRenderingContext2D): void {
    if (!this.progress) return

    const originalFillStyle = ctx.fillStyle
    ctx.fillStyle = 'green'
    ctx.fillRect(0, 0, this.width * this.progress, 6)
    ctx.fillStyle = originalFillStyle
  }
}
