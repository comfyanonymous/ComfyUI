import { toString } from 'es-toolkit/compat'
import { toValue } from 'vue'

import { PREFIX, SEPARATOR } from '@/constants/groupNodeConstants'
import { MovingInputLink } from '@/lib/litegraph/src/canvas/MovingInputLink'
import { LitegraphLinkAdapter } from '@/renderer/core/canvas/litegraph/litegraphLinkAdapter'
import type { LinkRenderContext } from '@/renderer/core/canvas/litegraph/litegraphLinkAdapter'
import { getSlotPosition } from '@/renderer/core/canvas/litegraph/slotCalculations'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import { LayoutSource } from '@/renderer/core/layout/types'
import { forEachNode } from '@/utils/graphTraversalUtil'

import { CanvasPointer } from './CanvasPointer'
import type { ContextMenu } from './ContextMenu'
import { DragAndScale } from './DragAndScale'
import type { AnimationOptions } from './DragAndScale'
import type { LGraph } from './LGraph'
import { LGraphGroup } from './LGraphGroup'
import { LGraphNode } from './LGraphNode'
import type { NodeId, NodeProperty } from './LGraphNode'
import { LLink } from './LLink'
import type { LinkId } from './LLink'
import { Reroute } from './Reroute'
import type { RerouteId } from './Reroute'
import { LinkConnector } from './canvas/LinkConnector'
import { isOverNodeInput, isOverNodeOutput } from './canvas/measureSlots'
import { strokeShape } from './draw'
import type {
  CustomEventDispatcher,
  ICustomEventTarget
} from './infrastructure/CustomEventTarget'
import type { LGraphCanvasEventMap } from './infrastructure/LGraphCanvasEventMap'
import { NullGraphError } from './infrastructure/NullGraphError'
import { Rectangle } from './infrastructure/Rectangle'
import type {
  CanvasColour,
  ColorOption,
  ConnectingLink,
  ContextMenuDivElement,
  DefaultConnectionColors,
  Dictionary,
  Direction,
  IBoundaryNodes,
  IColorable,
  IContextMenuOptions,
  IContextMenuValue,
  INodeInputSlot,
  INodeOutputSlot,
  INodeSlot,
  INodeSlotContextItem,
  ISlotType,
  LinkNetwork,
  LinkSegment,
  NewNodePosition,
  NullableProperties,
  Panel,
  PanelButton,
  PanelWidget,
  PanelWidgetCallback,
  PanelWidgetOptions,
  Point,
  Positionable,
  ReadOnlyRect,
  Rect,
  Size
} from './interfaces'
import { LiteGraph } from './litegraph'
import {
  containsRect,
  createBounds,
  distance,
  isInRect,
  isInRectangle,
  isPointInRect,
  overlapBounding,
  snapPoint
} from './measure'
import { NodeInputSlot } from './node/NodeInputSlot'
import type { Subgraph } from './subgraph/Subgraph'
import { SubgraphIONodeBase } from './subgraph/SubgraphIONodeBase'
import type { SubgraphInputNode } from './subgraph/SubgraphInputNode'
import { SubgraphNode } from './subgraph/SubgraphNode'
import type { SubgraphOutputNode } from './subgraph/SubgraphOutputNode'
import type {
  CanvasPointerEvent,
  CanvasPointerExtensions
} from './types/events'
import {
  CanvasItem,
  LGraphEventMode,
  LinkDirection,
  LinkMarkerShape,
  LinkRenderType,
  RenderShape,
  TitleMode
} from './types/globalEnums'
import type {
  ClipboardItems,
  ISerialisedNode,
  SubgraphIO
} from './types/serialisation'
import type { NeverNever, PickNevers } from './types/utility'
import type { IBaseWidget, TWidgetValue } from './types/widgets'
import { alignNodes, distributeNodes, getBoundaryNodes } from './utils/arrange'
import { findFirstNode, getAllNestedItems } from './utils/collections'
import { resolveConnectingLinkColor } from './utils/linkColors'
import { createUuidv4 } from './utils/uuid'
import type { UUID } from './utils/uuid'
import { BaseWidget } from './widgets/BaseWidget'
import { toConcreteWidget } from './widgets/widgetMap'

interface IShowSearchOptions {
  node_to?: LGraphNode | null
  node_from?: LGraphNode | null
  slot_from: number | INodeOutputSlot | INodeInputSlot | null | undefined
  type_filter_in?: ISlotType
  type_filter_out?: ISlotType | false

  // TODO check for registered_slot_[in/out]_types not empty // this will be checked for functionality enabled : filter on slot type, in and out
  do_type_filter?: boolean
  show_general_if_none_on_typefilter?: boolean
  show_general_after_typefiltered?: boolean
  hide_on_mouse_leave?: boolean
  show_all_if_empty?: boolean
  show_all_on_open?: boolean
}

interface ICreateNodeOptions {
  /** input */
  nodeFrom?: SubgraphInputNode | LGraphNode | null
  /** input */
  slotFrom?: number | INodeOutputSlot | INodeInputSlot | SubgraphIO | null
  /** output */
  nodeTo?: SubgraphOutputNode | LGraphNode | null
  /** output */
  slotTo?: number | INodeOutputSlot | INodeInputSlot | SubgraphIO | null
  /** pass the event coords */

  /** Create the connection from a reroute */
  afterRerouteId?: RerouteId

  // FIXME: Should not be optional
  /** choose a nodetype to add, AUTO to set at first good */
  nodeType?: string
  e?: CanvasPointerEvent
  allow_searchbox?: boolean
}

interface ICreateDefaultNodeOptions extends ICreateNodeOptions {
  /** Position of new node */
  position: Point
  /** adjust x,y */
  posAdd?: Point
  /** alpha, adjust the position x,y based on the new node size w,h */
  posSizeFix?: Point
}

interface HasShowSearchCallback {
  /** See {@link LGraphCanvas.showSearchBox} */
  showSearchBox: (
    event: MouseEvent | null,
    options?: IShowSearchOptions
  ) => HTMLDivElement | void
}

interface ICloseable {
  close(): void
}

interface IDialogExtensions extends ICloseable {
  modified(): void
  is_modified: boolean
}

interface IDialog extends HTMLDivElement, IDialogExtensions {}
type PromptDialog = Omit<IDialog, 'modified'>

interface IDialogOptions {
  position?: Point
  event?: MouseEvent
  checkForInput?: boolean
  closeOnLeave?: boolean
  onclose?(): void
}

/** @inheritdoc {@link LGraphCanvas.state} */
interface LGraphCanvasState {
  /** {@link Positionable} items are being dragged on the canvas. */
  draggingItems: boolean
  /** The canvas itself is being dragged. */
  draggingCanvas: boolean
  /** The canvas is read-only, preventing changes to nodes, disconnecting links, moving items, etc. */
  readOnly: boolean

  /** Bit flags indicating what is currently below the pointer. */
  hoveringOver: CanvasItem
  /** If `true`, pointer move events will set the canvas cursor style. */
  shouldSetCursor: boolean

  /**
   * Dirty flag indicating that {@link selectedItems} has changed.
   * Downstream consumers may reset to false once actioned.
   */
  selectionChanged: boolean

  /** ID of node currently in ghost placement mode (semi-transparent, following cursor). */
  ghostNodeId: NodeId | null
}

/**
 * The items created by a clipboard paste operation.
 * Includes maps of original copied IDs to newly created items.
 */
interface ClipboardPasteResult {
  /** All successfully created items */
  created: Positionable[]
  /** Map: original node IDs to newly created nodes */
  nodes: Map<NodeId, LGraphNode>
  /** Map: original link IDs to new link IDs */
  links: Map<LinkId, LLink>
  /** Map: original reroute IDs to newly created reroutes */
  reroutes: Map<RerouteId, Reroute>
  /** Map: original subgraph IDs to newly created subgraphs */
  subgraphs: Map<UUID, Subgraph>
}

/** Options for {@link LGraphCanvas.pasteFromClipboard}. */
interface IPasteFromClipboardOptions {
  /** If `true`, always attempt to connect inputs of pasted nodes - including to nodes that were not pasted. */
  connectInputs?: boolean
  /** The position to paste the items at. */
  position?: Point
}

interface ICreatePanelOptions {
  closable?: boolean
  window?: Window
  onOpen?: () => void
  onClose?: () => void
  width?: number | string
  height?: number | string
}

interface SlotTypeDefaultNodeOpts {
  node?: string
  title?: string
  properties?: Record<string, NodeProperty>
  inputs?: [string, string][]
  outputs?: [string, string][]
  json?: Parameters<LGraphNode['configure']>[0]
}

const cursors = {
  NE: 'nesw-resize',
  SE: 'nwse-resize',
  SW: 'nesw-resize',
  NW: 'nwse-resize'
} as const

// Optimised buffers used during rendering
const temp = new Rectangle()
const temp_vec2: Point = [0, 0]
const tmp_area = new Rectangle()
const margin_area = new Rectangle()
const link_bounding = new Rectangle()
/**
 * This class is in charge of rendering one graph inside a canvas. And provides all the interaction required.
 * Valid callbacks are: onNodeSelected, onNodeDeselected, onShowNodePanel, onNodeDblClicked
 */
export class LGraphCanvas implements CustomEventDispatcher<LGraphCanvasEventMap> {
  static DEFAULT_BACKGROUND_IMAGE =
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAQBJREFUeNrs1rEKwjAUhlETUkj3vP9rdmr1Ysammk2w5wdxuLgcMHyptfawuZX4pJSWZTnfnu/lnIe/jNNxHHGNn//HNbbv+4dr6V+11uF527arU7+u63qfa/bnmh8sWLBgwYJlqRf8MEptXPBXJXa37BSl3ixYsGDBMliwFLyCV/DeLIMFCxYsWLBMwSt4Be/NggXLYMGCBUvBK3iNruC9WbBgwYJlsGApeAWv4L1ZBgsWLFiwYJmCV/AK3psFC5bBggULloJX8BpdwXuzYMGCBctgwVLwCl7Be7MMFixYsGDBsu8FH1FaSmExVfAxBa/gvVmwYMGCZbBg/W4vAQYA5tRF9QYlv/QAAAAASUVORK5CYII='

  static DEFAULT_EVENT_LINK_COLOR = '#A86'

  /** Link type to colour dictionary. */
  static link_type_colors: Dictionary<string> = {
    '-1': LGraphCanvas.DEFAULT_EVENT_LINK_COLOR,
    number: '#AAA',
    node: '#DCA'
  }

  static gradients: Record<string, CanvasGradient> = {}

  static search_limit = -1
  static node_colors: Record<string, ColorOption> = {
    red: { color: '#322', bgcolor: '#533', groupcolor: '#A88' },
    brown: { color: '#332922', bgcolor: '#593930', groupcolor: '#b06634' },
    green: { color: '#232', bgcolor: '#353', groupcolor: '#8A8' },
    blue: { color: '#223', bgcolor: '#335', groupcolor: '#88A' },
    pale_blue: {
      color: '#2a363b',
      bgcolor: '#3f5159',
      groupcolor: '#3f789e'
    },
    cyan: { color: '#233', bgcolor: '#355', groupcolor: '#8AA' },
    purple: { color: '#323', bgcolor: '#535', groupcolor: '#a1309b' },
    yellow: { color: '#432', bgcolor: '#653', groupcolor: '#b58b2a' },
    black: { color: '#222', bgcolor: '#000', groupcolor: '#444' }
  }

  /**
   * @internal Exclusively a workaround for design limitation in {@link LGraphNode.computeSize}.
   */
  static _measureText?: (text: string, fontStyle?: string) => number

  /**
   * The state of this canvas, e.g. whether it is being dragged, or read-only.
   *
   * Implemented as a POCO that can be proxied without side-effects.
   */
  state: LGraphCanvasState = {
    draggingItems: false,
    draggingCanvas: false,
    readOnly: false,
    hoveringOver: CanvasItem.Nothing,
    shouldSetCursor: true,
    selectionChanged: false,
    ghostNodeId: null
  }

  private _subgraph?: Subgraph
  get subgraph(): Subgraph | undefined {
    return this._subgraph
  }

  set subgraph(value: Subgraph | undefined) {
    if (value !== this._subgraph) {
      this._subgraph = value
      if (value)
        this.dispatch('litegraph:set-graph', {
          oldGraph: this._subgraph,
          newGraph: value
        })
    }
  }

  /**
   * The location of the fps info widget. Leaving an element unset will use the default position for that element.
   */
  fpsInfoLocation:
    | [x: number | null | undefined, y: number | null | undefined]
    | null
    | undefined

  /** Dispatches a custom event on the canvas. */
  dispatch<T extends keyof NeverNever<LGraphCanvasEventMap>>(
    type: T,
    detail: LGraphCanvasEventMap[T]
  ): boolean
  dispatch<T extends keyof PickNevers<LGraphCanvasEventMap>>(type: T): boolean
  dispatch<T extends keyof LGraphCanvasEventMap>(
    type: T,
    detail?: LGraphCanvasEventMap[T]
  ) {
    const event = new CustomEvent(type as string, { detail, bubbles: true })
    return this.canvas.dispatchEvent(event)
  }

  dispatchEvent<TEvent extends keyof LGraphCanvasEventMap>(
    type: TEvent,
    detail: LGraphCanvasEventMap[TEvent]
  ) {
    this.canvas.dispatchEvent(new CustomEvent(type, { detail }))
  }

  private _updateCursorStyle() {
    if (!this.state.shouldSetCursor) return

    const crosshairItems =
      CanvasItem.Node |
      CanvasItem.RerouteSlot |
      CanvasItem.SubgraphIoNode |
      CanvasItem.SubgraphIoSlot

    let cursor = 'default'
    if (this.state.draggingCanvas) {
      cursor = 'grabbing'
    } else if (this.state.readOnly) {
      cursor = 'grab'
    } else if (this.pointer.resizeDirection) {
      cursor = cursors[this.pointer.resizeDirection] ?? cursors.SE
    } else if (this.state.hoveringOver & crosshairItems) {
      cursor = 'crosshair'
    } else if (this.state.hoveringOver & CanvasItem.Reroute) {
      cursor = 'grab'
    }

    this.canvas.style.cursor = cursor
  }

  // Whether the canvas was previously being dragged prior to pressing space key.
  // null if space key is not pressed.
  private _previously_dragging_canvas: boolean | null = null

  // #region Legacy accessors
  /** @deprecated @inheritdoc {@link LGraphCanvasState.readOnly} */
  get read_only(): boolean {
    return this.state.readOnly
  }

  set read_only(value: boolean) {
    this.state.readOnly = value
    this._updateCursorStyle()
  }

  get isDragging(): boolean {
    return this.state.draggingItems
  }

  set isDragging(value: boolean) {
    this.state.draggingItems = value
  }

  get hoveringOver(): CanvasItem {
    return this.state.hoveringOver
  }

  set hoveringOver(value: CanvasItem) {
    this.state.hoveringOver = value
    this._updateCursorStyle()
  }

  /** @deprecated Replace all references with {@link pointer}.{@link CanvasPointer.isDown isDown}. */
  get pointer_is_down() {
    return this.pointer.isDown
  }

  /** @deprecated Replace all references with {@link pointer}.{@link CanvasPointer.isDouble isDouble}. */
  get pointer_is_double() {
    return this.pointer.isDouble
  }

  /** @deprecated @inheritdoc {@link LGraphCanvasState.draggingCanvas} */
  get dragging_canvas(): boolean {
    return this.state.draggingCanvas
  }

  set dragging_canvas(value: boolean) {
    this.state.draggingCanvas = value
    this._updateCursorStyle()
  }

  /**
   * @deprecated Use {@link LGraphNode.titleFontStyle} instead.
   */
  get title_text_font(): string {
    return `${LiteGraph.NODE_TEXT_SIZE}px ${LiteGraph.NODE_FONT}`
  }
  // #endregion Legacy accessors

  get inner_text_font(): string {
    return `normal ${LiteGraph.NODE_SUBTEXT_SIZE}px ${LiteGraph.NODE_FONT}`
  }

  private _maximumFrameGap = 0
  /** Maximum frames per second to render. 0: unlimited. Default: 0 */
  public get maximumFps() {
    return this._maximumFrameGap > Number.EPSILON
      ? this._maximumFrameGap / 1000
      : 0
  }

  public set maximumFps(value) {
    this._maximumFrameGap = value > Number.EPSILON ? 1000 / value : 0
  }

  /**
   * @deprecated Use {@link LiteGraphGlobal.ROUND_RADIUS} instead.
   */
  get round_radius() {
    return LiteGraph.ROUND_RADIUS
  }

  /**
   * @deprecated Use {@link LiteGraphGlobal.ROUND_RADIUS} instead.
   */
  set round_radius(value: number) {
    LiteGraph.ROUND_RADIUS = value
  }

  // Cached LOD threshold values for performance
  private _lowQualityZoomThreshold: number = 0
  private _isLowQuality: boolean = false

  /**
   * Updates the low quality zoom threshold based on current settings.
   * Called when min_font_size_for_lod or DPR changes.
   */
  private updateLowQualityThreshold(): void {
    if (this._min_font_size_for_lod === 0) {
      // LOD disabled
      this._lowQualityZoomThreshold = 0
      this._isLowQuality = false
      return
    }

    const baseFontSize = LiteGraph.NODE_TEXT_SIZE // 14px
    const dprAdjustment = Math.sqrt(window.devicePixelRatio || 1) //Using sqrt here because higher DPR monitors do not linearily scale the readability of the font, instead they increase the font by some heurisitc, and to approximate we use sqrt to say basically a DPR of 2 increases the readability by 40%, 3 by 70%

    // Calculate the zoom level where text becomes unreadable
    this._lowQualityZoomThreshold =
      this._min_font_size_for_lod / (baseFontSize * dprAdjustment)

    // Update current state based on current zoom
    this._isLowQuality = this.ds.scale < this._lowQualityZoomThreshold
  }

  /**
   * Render low quality when zoomed out based on minimum readable font size.
   */
  get low_quality(): boolean {
    return this._isLowQuality
  }

  options: {
    skip_events?: boolean
    viewport?: Rect
    skip_render?: boolean
    autoresize?: boolean
  }

  background_image: string
  readonly ds: DragAndScale
  readonly pointer: CanvasPointer
  zoom_modify_alpha: boolean
  zoom_speed: number
  node_title_color: string
  default_link_color: string
  default_connection_color: {
    input_off: string
    input_on: string
    output_off: string
    output_on: string
  }

  default_connection_color_byType: Dictionary<CanvasColour>
  default_connection_color_byTypeOff: Dictionary<CanvasColour>

  /** Gets link colours. Extremely basic impl. until the legacy object dictionaries are removed. */
  colourGetter: DefaultConnectionColors = {
    getConnectedColor: (type: string) =>
      this.default_connection_color_byType[type] ||
      this.default_connection_color.output_on,
    getDisconnectedColor: (type: string) =>
      this.default_connection_color_byTypeOff[type] ||
      this.default_connection_color_byType[type] ||
      this.default_connection_color.output_off
  }

  highquality_render: boolean
  use_gradients: boolean
  editor_alpha: number
  pause_rendering: boolean
  clear_background: boolean
  clear_background_color: string
  render_only_selected: boolean
  show_info: boolean
  allow_dragcanvas: boolean
  allow_dragnodes: boolean
  allow_interaction: boolean
  multi_select: boolean
  allow_searchbox: boolean
  allow_reconnect_links: boolean
  align_to_grid: boolean
  drag_mode: boolean
  dragging_rectangle: Rect | null
  filter?: string | null
  set_canvas_dirty_on_mouse_event: boolean
  always_render_background: boolean
  render_shadows: boolean
  render_canvas_border: boolean
  render_connections_shadows: boolean
  render_connections_border: boolean
  render_curved_connections: boolean
  render_connection_arrows: boolean
  render_collapsed_slots: boolean
  render_execution_order: boolean
  render_link_tooltip: boolean

  /** Shape of the markers shown at the midpoint of links.  Default: Circle */
  linkMarkerShape: LinkMarkerShape = LinkMarkerShape.Circle
  links_render_mode: number
  /** Minimum font size in pixels before switching to low quality rendering.
   * This initializes first and if we can't get the value from the settings we default to 8px
   */
  private _min_font_size_for_lod: number = 8

  get min_font_size_for_lod(): number {
    return this._min_font_size_for_lod
  }

  set min_font_size_for_lod(value: number) {
    if (this._min_font_size_for_lod !== value) {
      this._min_font_size_for_lod = value
      this.updateLowQualityThreshold()
    }
  }
  /** mouse in canvas coordinates, where 0,0 is the top-left corner of the blue rectangle */
  readonly mouse: Point
  /** mouse in graph coordinates, where 0,0 is the top-left corner of the blue rectangle */
  readonly graph_mouse: Point
  /** @deprecated LEGACY: REMOVE THIS, USE {@link graph_mouse} INSTEAD */
  canvas_mouse: Point
  /** to personalize the search box */
  onSearchBox?: (
    helper: HTMLDivElement,
    str: string,
    canvas: LGraphCanvas
  ) => string[] | undefined
  onSearchBoxSelection?: (
    name: string,
    event: MouseEvent,
    canvas: LGraphCanvas
  ) => void
  onMouse?: (e: CanvasPointerEvent) => boolean
  /** to render background objects (behind nodes and connections) in the canvas affected by transform */
  onDrawBackground?: (
    ctx: CanvasRenderingContext2D,
    visible_area: Rectangle
  ) => void
  /** to render foreground objects (above nodes and connections) in the canvas affected by transform */
  onDrawForeground?: (
    ctx: CanvasRenderingContext2D,
    visible_area: Rectangle
  ) => void
  connections_width: number
  /** The current node being drawn by {@link drawNode}.  This should NOT be used to determine the currently selected node.  See {@link selectedItems} */
  current_node: LGraphNode | null
  /** used for widgets */
  node_widget?: [LGraphNode, IBaseWidget] | null
  /** The link to draw a tooltip for. */
  over_link_center?: LinkSegment
  last_mouse_position: Point
  /** The visible area of this canvas.  Tightly coupled with {@link ds}. */
  visible_area: Rectangle
  /** Contains all links and reroutes that were rendered.  Repopulated every render cycle. */
  renderedPaths: Set<LinkSegment> = new Set()
  /** @deprecated Replaced by {@link renderedPaths}, but length is set to 0 by some extensions. */
  visible_links: LLink[] = []
  /** @deprecated This array is populated and cleared to support legacy extensions. The contents are ignored by Litegraph. */
  connecting_links: ConnectingLink[] | null
  linkConnector = new LinkConnector((links) => (this.connecting_links = links))
  /** The viewport of this canvas.  Tightly coupled with {@link ds}. */
  readonly viewport?: Rect
  autoresize: boolean
  static active_canvas: LGraphCanvas
  frame = 0
  last_draw_time = 0
  render_time = 0
  fps = 0
  /** @deprecated See {@link LGraphCanvas.selectedItems} */
  selected_nodes: Dictionary<LGraphNode> = {}
  /** All selected nodes, groups, and reroutes */
  selectedItems: Set<Positionable> = new Set()
  /** The group currently being resized. */
  resizingGroup: LGraphGroup | null = null
  /** @deprecated See {@link LGraphCanvas.selectedItems} */
  selected_group: LGraphGroup | null = null
  /** The nodes that are currently visible on the canvas. */
  visible_nodes: LGraphNode[] = []
  /**
   * The IDs of the nodes that are currently visible on the canvas. More
   * performant than {@link visible_nodes} for visibility checks.
   */
  private _visible_node_ids: Set<NodeId> = new Set()
  node_over?: LGraphNode
  node_capturing_input?: LGraphNode | null
  highlighted_links: Dictionary<boolean> = {}

  private _visibleReroutes: Set<Reroute> = new Set()

  dirty_canvas: boolean = true
  dirty_bgcanvas: boolean = true
  /** A map of nodes that require selective-redraw */
  dirty_nodes = new Map<NodeId, LGraphNode>()
  dirty_area?: Rect | null
  /** @deprecated Unused */
  node_in_panel?: LGraphNode | null
  last_mouse: Readonly<Point> = [0, 0]
  last_mouseclick: number = 0
  graph: LGraph | Subgraph | null
  get _graph(): LGraph | Subgraph {
    if (!this.graph) throw new NullGraphError()
    return this.graph
  }

  canvas: HTMLCanvasElement & ICustomEventTarget<LGraphCanvasEventMap>
  bgcanvas: HTMLCanvasElement
  overlayCanvas: HTMLCanvasElement | null = null
  overlayCtx: CanvasRenderingContext2D | null = null
  ctx: CanvasRenderingContext2D
  _events_binded?: boolean
  _mousedown_callback?(e: PointerEvent): void
  _mousewheel_callback?(e: WheelEvent): void
  _mousemove_callback?(e: PointerEvent): void
  _mouseup_callback?(e: PointerEvent): void
  _mouseout_callback?(e: PointerEvent): void
  _mousecancel_callback?(e: PointerEvent): void
  _key_callback?(e: KeyboardEvent): void
  bgctx?: CanvasRenderingContext2D | null
  is_rendering?: boolean
  /** @deprecated Panels */
  block_click?: boolean
  /** @deprecated Panels */
  last_click_position?: Point | null
  resizing_node?: LGraphNode | null
  /** @deprecated See {@link LGraphCanvas.resizingGroup} */
  selected_group_resizing?: boolean
  /** @deprecated See {@link pointer}.{@link CanvasPointer.dragStarted dragStarted} */
  last_mouse_dragging?: boolean
  onMouseDown?: (arg0: CanvasPointerEvent) => void
  _highlight_pos?: Point
  _highlight_input?: INodeInputSlot
  // TODO: Check if panels are used
  /** @deprecated Panels */
  node_panel?: Panel
  /** @deprecated Panels */
  options_panel?: Panel
  _bg_img?: HTMLImageElement
  _pattern?: CanvasPattern
  _pattern_img?: HTMLImageElement
  bg_tint?: string | CanvasGradient | CanvasPattern
  // TODO: This looks like another panel thing
  prompt_box?: PromptDialog | null
  search_box?: HTMLDivElement
  /** @deprecated Panels */
  SELECTED_NODE?: LGraphNode
  /** @deprecated Panels */
  NODEPANEL_IS_OPEN?: boolean

  /** Once per frame check of snap to grid value.  @todo Update on change. */
  private _snapToGrid?: number
  /** Set on keydown, keyup. @todo */
  private _shiftDown: boolean = false

  /** Link rendering adapter for litegraph-to-canvas integration */
  linkRenderer: LitegraphLinkAdapter | null = null

  /** If true, enable drag zoom. Ctrl+Shift+Drag Up/Down: zoom canvas. */
  dragZoomEnabled: boolean = false
  /** The start position of the drag zoom and original read-only state. */
  private _dragZoomStart: {
    pos: Point
    scale: number
    readOnly: boolean
  } | null = null

  /** If true, enable live selection during drag. Nodes are selected/deselected in real-time. */
  liveSelection: boolean = false

  getMenuOptions?(): IContextMenuValue<string>[]
  getExtraMenuOptions?(
    canvas: LGraphCanvas,
    options: (IContextMenuValue<string> | null)[]
  ): (IContextMenuValue<string> | null)[]
  static active_node: LGraphNode
  /** called before modifying the graph */
  onBeforeChange?(graph: LGraph): void
  /** called after modifying the graph */
  onAfterChange?(graph: LGraph): void
  onClear?: () => void
  /** called after moving a node @deprecated Does not handle multi-node move, and can return the wrong node. */
  onNodeMoved?: (node_dragged: LGraphNode | undefined) => void
  /** @deprecated Called with the deprecated {@link selected_nodes} when the selection changes. Replacement not yet impl. */
  onSelectionChange?: (selected: Dictionary<Positionable>) => void
  /** called when rendering a tooltip */
  onDrawLinkTooltip?: (
    ctx: CanvasRenderingContext2D,
    link: LLink | null,
    canvas?: LGraphCanvas
  ) => boolean

  /** to render foreground objects not affected by transform (for GUIs) */
  onDrawOverlay?: (ctx: CanvasRenderingContext2D) => void
  onRenderBackground?: (
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D
  ) => boolean

  onNodeDblClicked?: (n: LGraphNode) => void
  onShowNodePanel?: (n: LGraphNode) => void
  onNodeSelected?: (node: LGraphNode) => void
  onNodeDeselected?: (node: LGraphNode) => void
  onRender?: (canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) => void

  /**
   * Creates a new instance of LGraphCanvas.
   * @param canvas The canvas HTML element (or its id) to use, or null / undefined to leave blank.
   * @param graph The graph that owns this canvas.
   * @param options
   */
  constructor(
    canvas: HTMLCanvasElement,
    graph: LGraph,
    options?: LGraphCanvas['options']
  ) {
    options ||= {}
    this.options = options

    // if(graph === undefined)
    // throw ("No graph assigned");
    this.background_image = LGraphCanvas.DEFAULT_BACKGROUND_IMAGE

    this.ds = new DragAndScale(canvas)
    this.pointer = new CanvasPointer(canvas)

    // Set up zoom change handler for efficient LOD updates
    this.ds.onChanged = (scale: number, _offset: Point) => {
      // Only check LOD threshold if it's enabled
      if (this._lowQualityZoomThreshold > 0) {
        this._isLowQuality = scale < this._lowQualityZoomThreshold
      }
    }

    // Initialize link renderer if graph is available
    if (graph) {
      this.linkRenderer = new LitegraphLinkAdapter(false)
    }

    this.linkConnector.events.addEventListener('link-created', () =>
      this._dirty()
    )

    // @deprecated Workaround: Keep until connecting_links is removed.
    this.linkConnector.events.addEventListener('reset', () => {
      this.connecting_links = null
      this.dirty_bgcanvas = true
    })

    // Dropped a link on the canvas
    this.linkConnector.events.addEventListener(
      'dropped-on-canvas',
      (customEvent) => {
        if (!this.connecting_links) return

        const e = customEvent.detail
        this.emitEvent({
          subType: 'empty-release',
          originalEvent: e,
          linkReleaseContext: { links: this.connecting_links }
        })

        const firstLink = this.linkConnector.renderLinks[0]

        // No longer in use
        // add menu when releasing link in empty space
        if (LiteGraph.release_link_on_empty_shows_menu) {
          const linkReleaseContext =
            this.linkConnector.state.connectingTo === 'input'
              ? {
                  node_from: firstLink.node as LGraphNode,
                  slot_from: firstLink.fromSlot as INodeOutputSlot,
                  type_filter_in: firstLink.fromSlot.type
                }
              : {
                  node_to: firstLink.node as LGraphNode,
                  slot_to: firstLink.fromSlot as INodeInputSlot,
                  type_filter_out: firstLink.fromSlot.type
                }

          const afterRerouteId = firstLink.fromReroute?.id

          if ('shiftKey' in e && e.shiftKey) {
            if (this.allow_searchbox) {
              this.showSearchBox(
                e as MouseEvent,
                linkReleaseContext as IShowSearchOptions
              )
            }
          } else if (this.linkConnector.state.connectingTo === 'input') {
            this.showConnectionMenu({
              nodeFrom: firstLink.node as LGraphNode,
              slotFrom: firstLink.fromSlot as INodeOutputSlot,
              e,
              afterRerouteId
            })
          } else {
            this.showConnectionMenu({
              nodeTo: firstLink.node as LGraphNode,
              slotTo: firstLink.fromSlot as INodeInputSlot,
              e,
              afterRerouteId
            })
          }
        }
      }
    )

    // otherwise it generates ugly patterns when scaling down too much
    this.zoom_modify_alpha = true
    // in range (1.01, 2.5). Less than 1 will invert the zoom direction
    this.zoom_speed = 1.1

    this.node_title_color = LiteGraph.NODE_TITLE_COLOR
    this.default_link_color = LiteGraph.LINK_COLOR
    this.default_connection_color = {
      input_off: '#778',
      input_on: '#7F7',
      output_off: '#778',
      output_on: '#7F7'
    }
    this.default_connection_color_byType = {
      /* number: "#7F7",
            string: "#77F",
            boolean: "#F77", */
    }
    this.default_connection_color_byTypeOff = {
      /* number: "#474",
            string: "#447",
            boolean: "#744", */
    }

    this.highquality_render = true
    // set to true to render titlebar with gradients
    this.use_gradients = false
    // used for transition
    this.editor_alpha = 1
    this.pause_rendering = false
    this.clear_background = true
    this.clear_background_color = '#222'

    this.render_only_selected = true
    this.show_info = true
    this.allow_dragcanvas = true
    this.allow_dragnodes = true
    // allow to control widgets, buttons, collapse, etc
    this.allow_interaction = true
    // allow selecting multi nodes without pressing extra keys
    this.multi_select = false
    this.allow_searchbox = true
    // allows to change a connection with having to redo it again
    this.allow_reconnect_links = true
    // snap to grid
    this.align_to_grid = false

    this.drag_mode = false
    this.dragging_rectangle = null

    // allows to filter to only accept some type of nodes in a graph
    this.filter = null

    // forces to redraw the canvas on mouse events (except move)
    this.set_canvas_dirty_on_mouse_event = true
    this.always_render_background = false
    this.render_shadows = true
    this.render_canvas_border = true
    // too much cpu
    this.render_connections_shadows = false
    this.render_connections_border = true
    this.render_curved_connections = false
    this.render_connection_arrows = false
    this.render_collapsed_slots = true
    this.render_execution_order = false
    this.render_link_tooltip = true

    this.links_render_mode = LinkRenderType.SPLINE_LINK

    this.mouse = [0, 0]
    this.graph_mouse = [0, 0]
    this.canvas_mouse = this.graph_mouse

    this.connections_width = 3

    this.current_node = null
    this.node_widget = null
    this.last_mouse_position = [0, 0]
    this.visible_area = this.ds.visible_area
    // Explicitly null-checked
    this.connecting_links = null

    // to constraint render area to a portion of the canvas
    this.viewport = options.viewport

    // link canvas and graph
    this.graph = graph
    graph?.attachCanvas(this)

    // TypeScript strict workaround: cannot use method to initialize properties.
    this.canvas = undefined!
    this.bgcanvas = undefined!
    this.ctx = undefined!

    this.setCanvas(canvas, options.skip_events)
    this.clear()

    LGraphCanvas._measureText = (
      text: string,
      fontStyle = this.inner_text_font
    ) => {
      const { ctx } = this
      const { font } = ctx
      try {
        ctx.font = fontStyle
        return ctx.measureText(text).width
      } finally {
        ctx.font = font
      }
    }

    if (!options.skip_render) {
      this.startRendering()
    }

    this.autoresize = options.autoresize ?? false

    this.updateLowQualityThreshold()
  }

  static onGroupAdd(
    _info: unknown,
    _entry: unknown,
    mouse_event: MouseEvent
  ): void {
    const canvas = LGraphCanvas.active_canvas

    const group = new LiteGraph.LGraphGroup()
    group.pos = canvas.convertEventToCanvasOffset(mouse_event)
    if (!canvas.graph) throw new NullGraphError()
    canvas.graph.add(group)
  }

  /**
   * @deprecated Functionality moved to {@link getBoundaryNodes}.  The new function returns null on failure, instead of an object with all null properties.
   * Determines the furthest nodes in each direction
   * @param nodes the nodes to from which boundary nodes will be extracted
   * @returns
   */
  static getBoundaryNodes(
    nodes: LGraphNode[] | Dictionary<LGraphNode>
  ): NullableProperties<IBoundaryNodes> {
    const _nodes = Array.isArray(nodes) ? nodes : Object.values(nodes)
    return (
      getBoundaryNodes(_nodes) ?? {
        top: null,
        right: null,
        bottom: null,
        left: null
      }
    )
  }

  /**
   * @deprecated Functionality moved to {@link alignNodes}.  The new function does not set dirty canvas.
   * @param nodes a list of nodes
   * @param direction Direction to align the nodes
   * @param align_to Node to align to (if null, align to the furthest node in the given direction)
   */
  static alignNodes(
    nodes: Dictionary<LGraphNode>,
    direction: Direction,
    align_to?: LGraphNode
  ): void {
    const newPositions = alignNodes(Object.values(nodes), direction, align_to)
    LGraphCanvas.active_canvas.repositionNodesVueMode(newPositions)
    LGraphCanvas.active_canvas.setDirty(true, true)
  }

  static onNodeAlign(
    _value: IContextMenuValue,
    _options: IContextMenuOptions,
    event: MouseEvent,
    prev_menu: ContextMenu<string>,
    node: LGraphNode
  ): void {
    new LiteGraph.ContextMenu(['Top', 'Bottom', 'Left', 'Right'], {
      event,
      callback: inner_clicked,
      parentMenu: prev_menu
    })

    function inner_clicked(value: string) {
      const newPositions = alignNodes(
        Object.values(LGraphCanvas.active_canvas.selected_nodes),
        value.toLowerCase() as Direction,
        node
      )
      LGraphCanvas.active_canvas.repositionNodesVueMode(newPositions)
      LGraphCanvas.active_canvas.setDirty(true, true)
    }
  }

  static onGroupAlign(
    _value: IContextMenuValue,
    _options: IContextMenuOptions,
    event: MouseEvent,
    prev_menu: ContextMenu<string>
  ): void {
    new LiteGraph.ContextMenu(['Top', 'Bottom', 'Left', 'Right'], {
      event,
      callback: inner_clicked,
      parentMenu: prev_menu
    })

    function inner_clicked(value: string) {
      const newPositions = alignNodes(
        Object.values(LGraphCanvas.active_canvas.selected_nodes),
        value.toLowerCase() as Direction
      )
      LGraphCanvas.active_canvas.repositionNodesVueMode(newPositions)
      LGraphCanvas.active_canvas.setDirty(true, true)
    }
  }

  static createDistributeMenu(
    _value: IContextMenuValue,
    _options: IContextMenuOptions,
    event: MouseEvent,
    prev_menu: ContextMenu<string>
  ): void {
    new LiteGraph.ContextMenu(['Vertically', 'Horizontally'], {
      event,
      callback: inner_clicked,
      parentMenu: prev_menu
    })

    function inner_clicked(value: string) {
      const canvas = LGraphCanvas.active_canvas
      const newPositions = distributeNodes(
        Object.values(canvas.selected_nodes),
        value === 'Horizontally'
      )
      canvas.repositionNodesVueMode(newPositions)
      canvas.setDirty(true, true)
    }
  }

  static onMenuAdd(
    _value: unknown,
    _options: unknown,
    e: MouseEvent,
    prev_menu?: ContextMenu<string>,
    callback?: (node: LGraphNode | null) => void
  ): boolean | undefined {
    const canvas = LGraphCanvas.active_canvas
    const { graph } = canvas
    if (!graph) return

    inner_onMenuAdded('', prev_menu)
    return false

    type AddNodeMenu = Omit<IContextMenuValue<string>, 'callback'> & {
      callback: (
        value: { value: string },
        event: Event,
        mouseEvent: MouseEvent,
        contextMenu: ContextMenu<string>
      ) => void
    }

    function inner_onMenuAdded(
      base_category: string,
      prev_menu?: ContextMenu<string>
    ): void {
      if (!graph) return

      const categories = LiteGraph.getNodeTypesCategories(
        canvas.filter || graph.filter
      ).filter((category) => category.startsWith(base_category))
      const entries: AddNodeMenu[] = []

      for (const category of categories) {
        if (!category) continue

        const base_category_regex = new RegExp(`^(${base_category})`)
        const category_name = category
          .replace(base_category_regex, '')
          .split('/', 1)[0]
        const category_path =
          base_category === ''
            ? `${category_name}/`
            : `${base_category}${category_name}/`

        let name = category_name
        // in case it has a namespace like "shader::math/rand" it hides the namespace
        if (name.includes('::')) name = name.split('::', 2)[1]

        const index = entries.findIndex(
          (entry) => entry.value === category_path
        )
        if (index === -1) {
          entries.push({
            value: category_path,
            content: name,
            has_submenu: true,
            callback: function (value, _event, _mouseEvent, contextMenu) {
              inner_onMenuAdded(value.value, contextMenu)
            }
          })
        }
      }

      const nodes = LiteGraph.getNodeTypesInCategory(
        base_category.slice(0, -1),
        canvas.filter || graph.filter
      )

      for (const node of nodes) {
        if (node.skip_list) continue

        const entry: AddNodeMenu = {
          value: node.type,
          content: node.title,
          has_submenu: false,
          callback: function (value, _event, _mouseEvent, contextMenu) {
            if (!canvas.graph) throw new NullGraphError()

            const first_event = contextMenu.getFirstEvent()
            canvas.graph.beforeChange()
            const node = LiteGraph.createNode(value.value)
            if (node) {
              if (!first_event)
                throw new TypeError(
                  'Context menu event was null. This should not occur in normal usage.'
                )
              node.pos = canvas.convertEventToCanvasOffset(first_event)
              canvas.graph.add(node)
            } else {
              console.warn('Failed to create node of type:', value.value)
            }

            callback?.(node)
            canvas.graph.afterChange()
          }
        }

        entries.push(entry)
      }

      new LiteGraph.ContextMenu(entries, { event: e, parentMenu: prev_menu })
    }
  }

  static onMenuCollapseAll() {}
  static onMenuNodeEdit() {}

  /** @param _options Parameter is never used */
  static showMenuNodeOptionalOutputs(
    _v: unknown,
    /** Unused - immediately overwritten */
    _options: INodeOutputSlot[],
    e: MouseEvent,
    prev_menu: ContextMenu<INodeSlotContextItem>,
    node: LGraphNode
  ): boolean | undefined {
    if (!node) return

    const canvas = LGraphCanvas.active_canvas

    let entries: (IContextMenuValue<INodeSlotContextItem> | null)[] = []

    if (
      LiteGraph.do_add_triggers_slots &&
      node.findOutputSlot('onExecuted') == -1
    ) {
      entries.push({
        content: 'On Executed',
        value: ['onExecuted', LiteGraph.EVENT, { nameLocked: true }],
        className: 'event'
      })
    }
    // add callback for modifying the menu elements onMenuNodeOutputs
    const retEntries = node.onMenuNodeOutputs?.(entries)
    if (retEntries) entries = retEntries

    if (!entries.length) return

    new LiteGraph.ContextMenu<INodeSlotContextItem>(entries, {
      event: e,
      callback: inner_clicked,
      parentMenu: prev_menu,
      node
    })

    function inner_clicked(
      this: ContextMenuDivElement<INodeSlotContextItem>,
      v?: string | IContextMenuValue<INodeSlotContextItem>,
      _options?: unknown,
      e?: MouseEvent,
      prev?: ContextMenu<INodeSlotContextItem>
    ) {
      if (!node) return
      if (!v || typeof v === 'string') return

      // TODO: This is a static method, so the below "that" appears broken.
      if (v.callback) void v.callback.call(this, node, v, e, prev)

      if (!v.value) return

      const value = v.value[1]

      if (value && (typeof value === 'object' || Array.isArray(value))) {
        // submenu why?
        const entries = []
        for (const i in value) {
          entries.push({ content: i, value: value[i] })
        }
        new LiteGraph.ContextMenu(entries, {
          event: e,
          callback: inner_clicked,
          parentMenu: prev_menu,
          node
        })
        return false
      }

      const { graph } = node
      if (!graph) throw new NullGraphError()

      graph.beforeChange()
      node.addOutput(v.value[0], v.value[1], v.value[2])

      // a callback to the node when adding a slot
      node.onNodeOutputAdd?.(v.value)
      canvas.setDirty(true, true)
      graph.afterChange()
    }

    return false
  }

  /** @param value Parameter is never used */
  static onShowMenuNodeProperties(
    value: NodeProperty | undefined,
    _options: unknown,
    e: MouseEvent,
    prev_menu: ContextMenu<string>,
    node: LGraphNode
  ): boolean | undefined {
    if (!node || !node.properties) return

    const canvas = LGraphCanvas.active_canvas

    const entries: IContextMenuValue<string>[] = []
    for (const i in node.properties) {
      value = node.properties[i] !== undefined ? node.properties[i] : ' '
      if (typeof value == 'object') value = JSON.stringify(value)
      const info = node.getPropertyInfo(i)
      if (info.type == 'enum' || info.type == 'combo')
        value = LGraphCanvas.getPropertyPrintableValue(value, info.values)

      // value could contain invalid html characters, clean that
      value = LGraphCanvas.decodeHTML(toString(value))
      entries.push({
        content:
          `<span class='property_name'>${info.label || i}</span>` +
          `<span class='property_value'>${value}</span>`,
        value: i
      })
    }
    if (!entries.length) {
      return
    }

    new LiteGraph.ContextMenu<string>(entries, {
      event: e,
      callback: inner_clicked,
      parentMenu: prev_menu,
      node
    })

    function inner_clicked(
      this: ContextMenuDivElement,
      v?: string | IContextMenuValue<string>
    ) {
      if (!node || typeof v === 'string' || !v?.value) return

      const rect = this.getBoundingClientRect()
      canvas.showEditPropertyValue(node, v.value, {
        position: [rect.left, rect.top]
      })
    }

    return false
  }

  /** @deprecated */
  static decodeHTML(str: string): string {
    const e = document.createElement('div')
    e.textContent = str
    return e.innerHTML
  }

  static onMenuResizeNode(
    _value: IContextMenuValue,
    _options: IContextMenuOptions,
    _e: MouseEvent,
    _menu: ContextMenu,
    node: LGraphNode
  ): void {
    if (!node) return

    const fApplyMultiNode = function (node: LGraphNode) {
      node.setSize(node.computeSize())
    }

    const canvas = LGraphCanvas.active_canvas
    if (
      !canvas.selected_nodes ||
      Object.keys(canvas.selected_nodes).length <= 1
    ) {
      fApplyMultiNode(node)
    } else {
      for (const i in canvas.selected_nodes) {
        fApplyMultiNode(canvas.selected_nodes[i])
      }
    }

    canvas.setDirty(true, true)
  }

  // TODO refactor :: this is used fot title but not for properties!
  static onShowPropertyEditor(
    item: { property: keyof LGraphNode; type: string },
    _options: IContextMenuOptions<string>,
    e: MouseEvent,
    _menu: ContextMenu<string>,
    node: LGraphNode
  ): void {
    const property = item.property || 'title'
    const value = node[property]

    const title = document.createElement('span')
    title.className = 'name'
    title.textContent = property

    const input = document.createElement('input')
    Object.assign(input, { type: 'text', className: 'value', autofocus: true })

    const button = document.createElement('button')
    button.textContent = 'OK'

    // TODO refactor :: use createDialog ?
    const dialog = Object.assign(document.createElement('div'), {
      is_modified: false,
      className: 'graphdialog',
      close: () => dialog.remove()
    })
    dialog.append(title, input, button)

    input.value = String(value)
    input.addEventListener('blur', function () {
      this.focus()
    })
    input.addEventListener('keydown', (e: KeyboardEvent) => {
      dialog.is_modified = true
      if (e.key == 'Escape') {
        // ESC
        dialog.close()
      } else if (e.key == 'Enter') {
        // save
        inner()
      } else if (
        !e.target ||
        !('localName' in e.target) ||
        e.target.localName != 'textarea'
      ) {
        return
      }
      e.preventDefault()
      e.stopPropagation()
    })

    const canvas = LGraphCanvas.active_canvas
    const canvasEl = canvas.canvas

    const rect = canvasEl.getBoundingClientRect()
    const offsetx = rect ? -20 - rect.left : -20
    const offsety = rect ? -20 - rect.top : -20

    if (e) {
      dialog.style.left = `${e.clientX + offsetx}px`
      dialog.style.top = `${e.clientY + offsety}px`
    } else {
      dialog.style.left = `${canvasEl.width * 0.5 + offsetx}px`
      dialog.style.top = `${canvasEl.height * 0.5 + offsety}px`
    }

    button.addEventListener('click', inner)

    if (canvasEl.parentNode == null)
      throw new TypeError('canvasEl.parentNode was null')
    canvasEl.parentNode.append(dialog)

    input.focus()

    let dialogCloseTimer: ReturnType<typeof setTimeout> | undefined
    dialog.addEventListener('mouseleave', function () {
      if (LiteGraph.dialog_close_on_mouse_leave) {
        if (!dialog.is_modified && LiteGraph.dialog_close_on_mouse_leave) {
          dialogCloseTimer = setTimeout(
            dialog.close,
            LiteGraph.dialog_close_on_mouse_leave_delay
          )
        }
      }
    })
    dialog.addEventListener('mouseenter', function () {
      if (LiteGraph.dialog_close_on_mouse_leave) {
        if (dialogCloseTimer) clearTimeout(dialogCloseTimer)
      }
    })

    function inner() {
      if (input) setValue(input.value)
    }

    function setValue(value: NodeProperty) {
      if (item.type == 'Number') {
        value = Number(value)
      } else if (item.type == 'Boolean') {
        value = Boolean(value)
      }
      // @ts-expect-error Requires refactor.
      node[property] = value
      dialog.remove()
      canvas.setDirty(true, true)
    }
  }

  static getPropertyPrintableValue(
    value: unknown,
    values: unknown[] | object | undefined
  ): string | undefined {
    if (!values) return String(value)

    if (Array.isArray(values)) {
      return String(value)
    }

    if (typeof values === 'object') {
      let desc_value = ''
      for (const k in values) {
        // @ts-expect-error deprecated #578
        if (values[k] != value) continue

        desc_value = k
        break
      }
      return `${String(value)} (${desc_value})`
    }
  }

  static onMenuNodeCollapse(
    _value: IContextMenuValue,
    _options: IContextMenuOptions,
    _e: MouseEvent,
    _menu: ContextMenu,
    node: LGraphNode
  ): void {
    if (!node.graph) throw new NullGraphError()

    node.graph.beforeChange()

    const fApplyMultiNode = function (node: LGraphNode) {
      node.collapse()
    }

    const graphcanvas = LGraphCanvas.active_canvas
    if (
      !graphcanvas.selected_nodes ||
      Object.keys(graphcanvas.selected_nodes).length <= 1
    ) {
      fApplyMultiNode(node)
    } else {
      for (const i in graphcanvas.selected_nodes) {
        fApplyMultiNode(graphcanvas.selected_nodes[i])
      }
    }

    node.graph.afterChange()
  }

  static onMenuToggleAdvanced(
    _value: IContextMenuValue,
    _options: IContextMenuOptions,
    _e: MouseEvent,
    _menu: ContextMenu,
    node: LGraphNode
  ): void {
    if (!node.graph) throw new NullGraphError()

    node.graph.beforeChange()
    const fApplyMultiNode = function (node: LGraphNode) {
      node.toggleAdvanced()
    }

    const graphcanvas = LGraphCanvas.active_canvas
    if (
      !graphcanvas.selected_nodes ||
      Object.keys(graphcanvas.selected_nodes).length <= 1
    ) {
      fApplyMultiNode(node)
    } else {
      for (const i in graphcanvas.selected_nodes) {
        fApplyMultiNode(graphcanvas.selected_nodes[i])
      }
    }
    node.graph.afterChange()
  }

  static onMenuNodeMode(
    _value: IContextMenuValue,
    _options: IContextMenuOptions,
    e: MouseEvent,
    menu: ContextMenu,
    node: LGraphNode
  ): boolean {
    new LiteGraph.ContextMenu(LiteGraph.NODE_MODES, {
      event: e,
      callback: inner_clicked,
      parentMenu: menu,
      node
    })

    function inner_clicked(v: string) {
      if (!node) return

      const kV = Object.values(LiteGraph.NODE_MODES).indexOf(v)
      const fApplyMultiNode = function (node: LGraphNode) {
        if (kV !== -1 && LiteGraph.NODE_MODES[kV]) {
          node.changeMode(kV)
        } else {
          console.warn(`unexpected mode: ${v}`)
          node.changeMode(LGraphEventMode.ALWAYS)
        }
      }

      const graphcanvas = LGraphCanvas.active_canvas
      if (
        !graphcanvas.selected_nodes ||
        Object.keys(graphcanvas.selected_nodes).length <= 1
      ) {
        fApplyMultiNode(node)
      } else {
        for (const i in graphcanvas.selected_nodes) {
          fApplyMultiNode(graphcanvas.selected_nodes[i])
        }
      }
    }

    return false
  }

  /** @param value Parameter is never used */
  static onMenuNodeColors(
    value: IContextMenuValue<string | null>,
    _options: IContextMenuOptions,
    e: MouseEvent,
    menu: ContextMenu<string | null>,
    node: LGraphNode
  ): boolean {
    if (!node) throw 'no node for color'

    const values: IContextMenuValue<
      string | null,
      unknown,
      { value: string | null }
    >[] = []
    values.push({
      value: null,
      content:
        "<span style='display: block; padding-left: 4px;'>No color</span>"
    })

    for (const i in LGraphCanvas.node_colors) {
      const color = LGraphCanvas.node_colors[i]
      value = {
        value: i,
        content:
          `<span style='display: block; color: #999; padding-left: 4px;` +
          ` border-left: 8px solid ${color.color}; background-color:${color.bgcolor}'>${i}</span>`
      }
      values.push(value)
    }
    new LiteGraph.ContextMenu<string | null>(values, {
      event: e,
      callback: inner_clicked,
      parentMenu: menu,
      node
    })

    function inner_clicked(v: IContextMenuValue<string>) {
      if (!node) return

      const fApplyColor = function (item: IColorable) {
        const colorOption = v.value ? LGraphCanvas.node_colors[v.value] : null
        item.setColorOption(colorOption)
      }

      const canvas = LGraphCanvas.active_canvas
      if (
        !canvas.selected_nodes ||
        Object.keys(canvas.selected_nodes).length <= 1
      ) {
        fApplyColor(node)
      } else {
        for (const i in canvas.selected_nodes) {
          fApplyColor(canvas.selected_nodes[i])
        }
      }
      canvas.setDirty(true, true)
    }

    return false
  }

  static onMenuNodeShapes(
    _value: IContextMenuValue<(typeof LiteGraph.VALID_SHAPES)[number]>,
    _options: IContextMenuOptions<(typeof LiteGraph.VALID_SHAPES)[number]>,
    e: MouseEvent,
    menu?: ContextMenu<(typeof LiteGraph.VALID_SHAPES)[number]>,
    node?: LGraphNode
  ): boolean {
    if (!node) throw 'no node passed'

    new LiteGraph.ContextMenu<(typeof LiteGraph.VALID_SHAPES)[number]>(
      LiteGraph.VALID_SHAPES,
      {
        event: e,
        callback: inner_clicked,
        parentMenu: menu,
        node
      }
    )

    function inner_clicked(v: (typeof LiteGraph.VALID_SHAPES)[number]) {
      if (!node) return
      if (!node.graph) throw new NullGraphError()

      node.graph.beforeChange()

      const fApplyMultiNode = function (node: LGraphNode) {
        node.shape = v
      }

      const canvas = LGraphCanvas.active_canvas
      if (
        !canvas.selected_nodes ||
        Object.keys(canvas.selected_nodes).length <= 1
      ) {
        fApplyMultiNode(node)
      } else {
        for (const i in canvas.selected_nodes) {
          fApplyMultiNode(canvas.selected_nodes[i])
        }
      }

      node.graph.afterChange()
      canvas.setDirty(true)
    }

    return false
  }

  static onMenuNodeRemove(): void {
    LGraphCanvas.active_canvas.deleteSelected()
  }

  static onMenuNodeClone(
    _value: IContextMenuValue,
    _options: IContextMenuOptions,
    _e: MouseEvent,
    _menu: ContextMenu,
    node: LGraphNode
  ): void {
    const canvas = LGraphCanvas.active_canvas
    const nodes = canvas.selectedItems.size ? [...canvas.selectedItems] : [node]
    if (nodes.length) LGraphCanvas.cloneNodes(nodes)
  }

  static cloneNodes(nodes: Positionable[]) {
    const canvas = LGraphCanvas.active_canvas

    // Find top-left-most boundary
    let offsetX = Infinity
    let offsetY = Infinity
    for (const item of nodes) {
      if (item.pos == null)
        throw new TypeError(
          'Invalid node encountered on clone.  `pos` was null.'
        )
      offsetX = Math.min(offsetX, item.pos[0])
      offsetY = Math.min(offsetY, item.pos[1])
    }

    return canvas._deserializeItems(canvas._serializeItems(nodes), {
      position: [offsetX + 5, offsetY + 5]
    })
  }

  /**
   * clears all the data inside
   *
   */
  clear(): void {
    this.frame = 0
    this.last_draw_time = 0
    this.render_time = 0
    this.fps = 0

    // this.scale = 1;
    // this.offset = [0,0];
    this.dragging_rectangle = null

    this.selected_nodes = {}
    this.selected_group = null
    this.selectedItems.clear()
    this.state.selectionChanged = true
    this.onSelectionChange?.(this.selected_nodes)

    this.visible_nodes = []
    this.node_over = undefined
    this.node_capturing_input = null
    this.connecting_links = null
    this.highlighted_links = {}

    this.dragging_canvas = false

    this._dirty()
    this.dirty_area = null

    this.node_in_panel = null
    this.node_widget = null

    this.last_mouse = [0, 0]
    this.last_mouseclick = 0
    this.pointer.reset()
    this.visible_area.set([0, 0, 0, 0])

    this.onClear?.()
  }

  /**
   * Assigns a new graph to this canvas.
   */
  setGraph(newGraph: LGraph | Subgraph): void {
    const { graph } = this
    if (newGraph === graph) return

    this.clear()
    newGraph.attachCanvas(this)

    // Re-initialize link renderer with new graph
    this.linkRenderer = new LitegraphLinkAdapter(false)

    this.dispatch('litegraph:set-graph', { newGraph, oldGraph: graph })
    this._dirty()
  }

  openSubgraph(subgraph: Subgraph, fromNode: SubgraphNode): void {
    const { graph } = this
    if (!graph) throw new NullGraphError()

    const options = {
      bubbles: true,
      detail: { subgraph, closingGraph: graph, fromNode },
      cancelable: true
    }
    const mayContinue = this.canvas.dispatchEvent(
      new CustomEvent('subgraph-opening', options)
    )
    if (!mayContinue) return

    this.clear()
    this.subgraph = subgraph
    this.setGraph(subgraph)

    this.canvas.dispatchEvent(new CustomEvent('subgraph-opened', options))
  }

  /**
   * @returns the visually active graph (in case there are more in the stack)
   */
  getCurrentGraph(): LGraph | null {
    return this.graph
  }

  /**
   * Finds the canvas if required, throwing on failure.
   * @param canvas Canvas element, or its element ID
   * @returns The canvas element
   * @throws If {@link canvas} is an element ID that does not belong to a valid HTML canvas element
   */
  private _validateCanvas(
    canvas: string | HTMLCanvasElement
  ): HTMLCanvasElement & { data?: LGraphCanvas } {
    if (typeof canvas === 'string') {
      const el = document.getElementById(canvas)
      if (!(el instanceof HTMLCanvasElement))
        throw 'Error validating LiteGraph canvas: Canvas element not found'
      return el
    }
    return canvas
  }

  /**
   * Sets the current HTML canvas element.
   * Calls bindEvents to add input event listeners, and (re)creates the background canvas.
   * @param canvas The canvas element to assign, or its HTML element ID.  If null or undefined, the current reference is cleared.
   * @param skip_events If true, events on the previous canvas will not be removed.  Has no effect on the first invocation.
   */
  setCanvas(canvas: string | HTMLCanvasElement, skip_events?: boolean) {
    const element = this._validateCanvas(canvas)
    if (element === this.canvas) return
    // maybe detach events from old_canvas
    if (!element && this.canvas && !skip_events) this.unbindEvents()

    this.canvas = element
    this.ds.element = element
    this.pointer.element = element

    if (!element) return

    // TODO: classList.add
    element.className += ' lgraphcanvas'
    Object.defineProperty(element, 'data', {
      value: this,
      writable: true,
      configurable: true,
      enumerable: false
    })

    // Background canvas: To render objects behind nodes (background, links, groups)
    this.bgcanvas = document.createElement('canvas')
    this.bgcanvas.width = this.canvas.width
    this.bgcanvas.height = this.canvas.height

    const ctx = element.getContext?.('2d')
    if (ctx == null) {
      if (element.localName != 'canvas') {
        throw `Element supplied for LGraphCanvas must be a <canvas> element, you passed a ${element.localName}`
      }
      throw "This browser doesn't support Canvas"
    }
    this.ctx = ctx

    if (!skip_events) this.bindEvents()
  }

  /** Captures an event and prevents default - returns false. */
  _doNothing(e: Event): boolean {
    // console.log("pointerevents: _doNothing "+e.type);
    e.preventDefault()
    return false
  }

  /** Captures an event and prevents default - returns true. */
  _doReturnTrue(e: Event): boolean {
    e.preventDefault()
    return true
  }

  /**
   * binds mouse, keyboard, touch and drag events to the canvas
   */
  bindEvents(): void {
    if (this._events_binded) {
      console.warn('LGraphCanvas: events already bound')
      return
    }

    const { canvas } = this
    // hack used when moving canvas between windows
    const { document } = this.getCanvasWindow()

    this._mousedown_callback = this.processMouseDown.bind(this)
    this._mousewheel_callback = this.processMouseWheel.bind(this)
    this._mousemove_callback = this.processMouseMove.bind(this)
    this._mouseup_callback = this.processMouseUp.bind(this)
    this._mouseout_callback = this.processMouseOut.bind(this)
    this._mousecancel_callback = this.processMouseCancel.bind(this)

    canvas.addEventListener('pointerdown', this._mousedown_callback, true)
    canvas.addEventListener('wheel', this._mousewheel_callback, false)

    canvas.addEventListener('pointerup', this._mouseup_callback, true)
    canvas.addEventListener('pointermove', this._mousemove_callback)
    canvas.addEventListener('pointerout', this._mouseout_callback)
    canvas.addEventListener('pointercancel', this._mousecancel_callback, true)

    canvas.addEventListener('contextmenu', this._doNothing)

    // Keyboard
    this._key_callback = this.processKey.bind(this)

    canvas.addEventListener('keydown', this._key_callback, true)
    // keyup event must be bound on the document
    document.addEventListener('keyup', this._key_callback, true)

    canvas.addEventListener('dragover', this._doNothing, false)
    canvas.addEventListener('dragend', this._doNothing, false)
    canvas.addEventListener('dragenter', this._doReturnTrue, false)

    this._events_binded = true
  }

  /**
   * unbinds mouse events from the canvas
   */
  unbindEvents(): void {
    if (!this._events_binded) {
      console.warn('LGraphCanvas: no events bound')
      return
    }

    // console.log("pointerevents: unbindEvents");
    const { document } = this.getCanvasWindow()
    const { canvas } = this

    // Assertions: removing nullish is fine.
    canvas.removeEventListener('pointercancel', this._mousecancel_callback!)
    canvas.removeEventListener('pointerout', this._mouseout_callback!)
    canvas.removeEventListener('pointermove', this._mousemove_callback!)
    canvas.removeEventListener('pointerup', this._mouseup_callback!)
    canvas.removeEventListener('pointerdown', this._mousedown_callback!)
    canvas.removeEventListener('wheel', this._mousewheel_callback!)
    canvas.removeEventListener('keydown', this._key_callback!)
    document.removeEventListener('keyup', this._key_callback!)
    canvas.removeEventListener('contextmenu', this._doNothing)
    canvas.removeEventListener('dragenter', this._doReturnTrue)

    this._mousedown_callback = undefined
    this._mousewheel_callback = undefined
    this._key_callback = undefined

    this._events_binded = false
  }

  /**
   * Ensures the canvas will be redrawn on the next frame by setting the dirty flag(s).
   * Without parameters, this function does nothing.
   * @todo Impl. `setDirty()` or similar as shorthand to redraw everything.
   * @param fgcanvas If true, marks the foreground canvas as dirty (nodes and anything drawn on top of them).  Default: false
   * @param bgcanvas If true, mark the background canvas as dirty (background, groups, links).  Default: false
   */
  setDirty(fgcanvas: boolean, bgcanvas?: boolean): void {
    if (fgcanvas) this.dirty_canvas = true
    if (bgcanvas) this.dirty_bgcanvas = true
  }

  /** Marks the entire canvas as dirty. */
  private _dirty(): void {
    this.dirty_canvas = true
    this.dirty_bgcanvas = true
  }

  private _linkConnectorDrop(): void {
    const { graph, linkConnector, pointer } = this
    if (!graph) throw new NullGraphError()

    pointer.onDragEnd = (upEvent) => linkConnector.dropLinks(graph, upEvent)
    pointer.finally = () => this.linkConnector.reset(true)
  }

  /**
   * Used to attach the canvas in a popup
   * @returns returns the window where the canvas is attached (the DOM root node)
   */
  getCanvasWindow(): Window {
    if (!this.canvas) return window

    const doc = this.canvas.ownerDocument
    // @ts-expect-error Check if required
    return doc.defaultView || doc.parentWindow
  }

  /**
   * starts rendering the content of the canvas when needed
   *
   */
  startRendering(): void {
    // already rendering
    if (this.is_rendering) return

    this.is_rendering = true
    renderFrame.call(this)

    /** Render loop */
    function renderFrame(this: LGraphCanvas) {
      if (!this.pause_rendering) {
        this.draw()
      }

      const window = this.getCanvasWindow()
      if (this.is_rendering) {
        if (this._maximumFrameGap > 0) {
          // Manual FPS limit
          const gap =
            this._maximumFrameGap - (LiteGraph.getTime() - this.last_draw_time)
          setTimeout(renderFrame.bind(this), Math.max(1, gap))
        } else {
          // FPS limited by refresh rate
          window.requestAnimationFrame(renderFrame.bind(this))
        }
      }
    }
  }

  /**
   * stops rendering the content of the canvas (to save resources)
   *
   */
  stopRendering(): void {
    this.is_rendering = false
    /*
    if(this.rendering_timer_id)
    {
        clearInterval(this.rendering_timer_id);
        this.rendering_timer_id = null;
    }
    */
  }

  /* LiteGraphCanvas input */
  // used to block future mouse events (because of im gui)
  blockClick(): void {
    this.block_click = true
    this.last_mouseclick = 0
  }

  /**
   * Gets the widget at the current cursor position.
   * @param node Optional node to check for widgets under cursor
   * @returns The widget located at the current cursor position, if any is found.
   * @deprecated Use {@link LGraphNode.getWidgetOnPos} instead.
   * ```ts
   * const [x, y] = canvas.graph_mouse
   * const widget = canvas.node_over?.getWidgetOnPos(x, y, true)
   * ```
   */
  getWidgetAtCursor(node?: LGraphNode): IBaseWidget | undefined {
    node ??= this.node_over
    return node?.getWidgetOnPos(this.graph_mouse[0], this.graph_mouse[1], true)
  }

  /**
   * Clears highlight and mouse-over information from nodes that should not have it.
   *
   * Intended to be called when the pointer moves away from a node.
   * @param node The node that the mouse is now over
   * @param e MouseEvent that is triggering this
   */
  updateMouseOverNodes(node: LGraphNode | null, e: CanvasPointerEvent): void {
    if (!this.graph) throw new NullGraphError()

    const { pointer } = this
    const nodes = this.graph._nodes
    for (const otherNode of nodes) {
      if (otherNode.mouseOver && node != otherNode) {
        // mouse leave
        if (!pointer.eDown) pointer.resizeDirection = undefined
        otherNode.mouseOver = undefined
        this._highlight_input = undefined
        this._highlight_pos = undefined
        this.linkConnector.overWidget = undefined

        // Hover transitions
        // TODO: Implement single lerp ease factor for current progress on hover in/out.
        // In drawNode, multiply by ease factor and differential value (e.g. bg alpha +0.5).
        otherNode.lostFocusAt = LiteGraph.getTime()

        this.node_over?.onMouseLeave?.(e)
        this.node_over = undefined
        this.dirty_canvas = true
      }
    }
  }

  processMouseDown(e: MouseEvent): void {
    if (this.state.ghostNodeId != null) {
      if (e.button === 0) this.finalizeGhostPlacement(false)
      if (e.button === 2) this.finalizeGhostPlacement(true)
      e.stopPropagation()
      e.preventDefault()
      return
    }

    if (
      this.dragZoomEnabled &&
      e.ctrlKey &&
      e.shiftKey &&
      !e.altKey &&
      e.buttons
    ) {
      this._dragZoomStart = {
        pos: [e.x, e.y],
        scale: this.ds.scale,
        readOnly: this.read_only
      }
      this.read_only = true
      return
    }

    const { graph, pointer } = this
    this.adjustMouseEvent(e)
    if (e.isPrimary) pointer.down(e)

    if (this.set_canvas_dirty_on_mouse_event) this.dirty_canvas = true

    if (!graph) return

    const ref_window = this.getCanvasWindow()
    LGraphCanvas.active_canvas = this

    const x = e.clientX
    const y = e.clientY
    this.ds.viewport = this.viewport
    const is_inside = !this.viewport || isInRect(x, y, this.viewport)

    if (!is_inside) return

    let node =
      graph.getNodeOnPos(e.canvasX, e.canvasY, this.visible_nodes) ?? undefined

    // In Vue nodes mode, slots extend beyond node bounds due to CSS transforms.
    // If no node was found, check if the click is on a slot and use its owning node.
    if (!node && LiteGraph.vueNodesMode) {
      const slotLayout = layoutStore.querySlotAtPoint({
        x: e.canvasX,
        y: e.canvasY
      })
      if (slotLayout) {
        node = graph.getNodeById(slotLayout.nodeId) ?? undefined
      }
    }

    this.mouse[0] = x
    this.mouse[1] = y
    this.graph_mouse[0] = e.canvasX
    this.graph_mouse[1] = e.canvasY
    this.last_click_position = [this.mouse[0], this.mouse[1]]

    pointer.isDouble = pointer.isDown && e.isPrimary
    pointer.isDown = true

    this.canvas.focus()

    LiteGraph.closeAllContextMenus(ref_window)

    if (this.onMouse?.(e) == true) return

    // left button mouse / single finger
    if (e.button === 0 && !pointer.isDouble) {
      this._processPrimaryButton(e, node)
    } else if (e.button === 1) {
      this._processMiddleButton(e, node)
    } else if (
      (e.button === 2 || pointer.isDouble) &&
      this.allow_interaction &&
      !this.read_only
    ) {
      // Right / aux button
      const { linkConnector, subgraph } = this

      // Sticky select - won't remove single nodes
      if (subgraph?.inputNode.containsPoint(this.graph_mouse)) {
        // Subgraph input node
        this.processSelect(subgraph.inputNode, e, true)
        subgraph.inputNode.onPointerDown(e, pointer, linkConnector)
      } else if (subgraph?.outputNode.containsPoint(this.graph_mouse)) {
        // Subgraph output node
        this.processSelect(subgraph.outputNode, e, true)
        subgraph.outputNode.onPointerDown(e, pointer, linkConnector)
      } else {
        if (node) {
          this.processSelect(node, e, true)
        } else if (this.links_render_mode !== LinkRenderType.HIDDEN_LINK) {
          // Reroutes
          // Try layout store first, fallback to old method
          const rerouteLayout = layoutStore.queryRerouteAtPoint({
            x: e.canvasX,
            y: e.canvasY
          })

          let reroute: Reroute | undefined
          if (rerouteLayout) {
            reroute = graph.getReroute(rerouteLayout.id)
          } else {
            reroute = graph.getRerouteOnPos(
              e.canvasX,
              e.canvasY,
              this._visibleReroutes
            )
          }
          if (reroute) {
            if (e.altKey) {
              pointer.onClick = (upEvent) => {
                if (upEvent.altKey) {
                  // Ensure deselected
                  if (reroute.selected) {
                    this.deselect(reroute)
                    this.onSelectionChange?.(this.selected_nodes)
                  }
                  reroute.remove()
                }
              }
            } else {
              this.processSelect(reroute, e, true)
            }
          }
        }

        // Show context menu for the node or group under the pointer
        pointer.onClick ??= () => this.processContextMenu(node, e)
      }
    }

    this.last_mouse = [x, y]
    this.last_mouseclick = LiteGraph.getTime()
    this.last_mouse_dragging = true

    graph.change()

    // this is to ensure to defocus(blur) if a text input element is on focus
    if (
      !ref_window.document.activeElement ||
      (ref_window.document.activeElement.nodeName.toLowerCase() != 'input' &&
        ref_window.document.activeElement.nodeName.toLowerCase() != 'textarea')
    ) {
      e.preventDefault()
    }
    e.stopPropagation()

    this.onMouseDown?.(e)
  }

  /**
   * Returns the first matching positionable item at the given co-ordinates.
   *
   * Order of preference:
   * - Subgraph IO Nodes
   * - Reroutes
   * - Group titlebars
   * @param x The x coordinate in canvas space
   * @param y The y coordinate in canvas space
   * @returns The positionable item or undefined
   */
  private _getPositionableOnPos(
    x: number,
    y: number
  ): Positionable | undefined {
    const ioNode = this.subgraph?.getIoNodeOnPos(x, y)
    if (ioNode) return ioNode

    for (const reroute of this._visibleReroutes) {
      if (reroute.containsPoint([x, y])) return reroute
    }

    return this.graph?.getGroupTitlebarOnPos(x, y)
  }

  private _processPrimaryButton(
    e: CanvasPointerEvent,
    node: LGraphNode | undefined
  ) {
    const { pointer, graph, linkConnector, subgraph } = this
    if (!graph) throw new NullGraphError()

    const x = e.canvasX
    const y = e.canvasY

    // Modifiers
    const ctrlOrMeta = e.ctrlKey || e.metaKey

    // Multi-select drag rectangle
    if (
      ctrlOrMeta &&
      !e.altKey &&
      LiteGraph.leftMouseClickBehavior === 'panning'
    ) {
      this._setupNodeSelectionDrag(e, pointer, node)

      return
    }

    if (this.read_only) {
      pointer.finally = () => (this.dragging_canvas = false)
      this.dragging_canvas = true
      return
    }

    // clone node ALT dragging
    if (
      !LiteGraph.vueNodesMode &&
      LiteGraph.alt_drag_do_clone_nodes &&
      e.altKey &&
      !e.ctrlKey &&
      node &&
      this.allow_interaction
    ) {
      const items = this._deserializeItems(this._serializeItems([node]), {
        position: node.pos
      })
      const cloned = items?.created[0] as LGraphNode | undefined
      if (!cloned) return

      cloned.setPos(cloned.pos[0] + 5, cloned.pos[1] + 5)

      if (this.allow_dragnodes) {
        pointer.onDragStart = (pointer) => {
          this._startDraggingItems(cloned, pointer)
        }
        pointer.onDragEnd = (e) => this._processDraggedItems(e)
      }
      return
    }

    // Node clicked
    if (node && (this.allow_interaction || node.flags.allow_interaction)) {
      this._processNodeClick(e, ctrlOrMeta, node)
    } else {
      // Subgraph IO nodes
      if (subgraph) {
        const { inputNode, outputNode } = subgraph

        if (processSubgraphIONode(this, inputNode)) return
        if (processSubgraphIONode(this, outputNode)) return

        function processSubgraphIONode(
          canvas: LGraphCanvas,
          ioNode: SubgraphInputNode | SubgraphOutputNode
        ) {
          if (!ioNode.containsPoint([x, y])) return false

          ioNode.onPointerDown(e, pointer, linkConnector)
          pointer.onClick ??= () => canvas.processSelect(ioNode, e)
          pointer.onDragStart ??= () =>
            canvas._startDraggingItems(ioNode, pointer, true)
          pointer.onDragEnd ??= (eUp) => canvas._processDraggedItems(eUp)
          return true
        }
      }

      // Reroutes
      if (this.links_render_mode !== LinkRenderType.HIDDEN_LINK) {
        // Try layout store first for hit detection
        const rerouteLayout = layoutStore.queryRerouteAtPoint({ x, y })
        let foundReroute: Reroute | undefined

        if (rerouteLayout) {
          foundReroute = graph.getReroute(rerouteLayout.id)
        }

        // Fallback to checking visible reroutes directly
        for (const reroute of this._visibleReroutes) {
          const overReroute =
            foundReroute === reroute || reroute.containsPoint([x, y])
          if (!reroute.isSlotHovered && !overReroute) continue

          if (overReroute) {
            pointer.onClick = () => this.processSelect(reroute, e)
            if (!e.shiftKey) {
              pointer.onDragStart = (pointer) =>
                this._startDraggingItems(reroute, pointer, true)
              pointer.onDragEnd = (e) => this._processDraggedItems(e)
            }
          }

          if (reroute.isOutputHovered || (overReroute && e.shiftKey)) {
            linkConnector.dragFromReroute(graph, reroute)
            this._linkConnectorDrop()
          }

          if (reroute.isInputHovered) {
            linkConnector.dragFromRerouteToOutput(graph, reroute)
            this._linkConnectorDrop()
          }

          reroute.hideSlots()
          this.dirty_bgcanvas = true
          return
        }
      }

      // Links - paths of links & reroutes
      // Set the width of the line for isPointInStroke checks
      const { lineWidth } = this.ctx
      this.ctx.lineWidth = this.connections_width + 7
      const dpi = Math.max(window?.devicePixelRatio ?? 1, 1)

      // Try layout store for segment hit testing first (more precise)
      const hitSegment = layoutStore.queryLinkSegmentAtPoint({ x, y }, this.ctx)

      for (const linkSegment of this.renderedPaths) {
        const centre = linkSegment._pos
        if (!centre) continue

        // Check if this link segment was hit
        let isLinkHit =
          hitSegment &&
          linkSegment.id ===
            (linkSegment instanceof Reroute
              ? hitSegment.rerouteId
              : hitSegment.linkId)

        if (!isLinkHit && linkSegment.path) {
          // Fallback to direct path hit testing if not found in layout store
          isLinkHit = this.ctx.isPointInStroke(
            linkSegment.path,
            x * dpi,
            y * dpi
          )
        }

        // If we shift click on a link then start a link from that input
        if ((e.shiftKey || e.altKey) && isLinkHit) {
          this.ctx.lineWidth = lineWidth

          if (e.shiftKey && !e.altKey) {
            linkConnector.dragFromLinkSegment(graph, linkSegment)
            this._linkConnectorDrop()

            return
          } else if (e.altKey && !e.shiftKey) {
            const newReroute = graph.createReroute([x, y], linkSegment)
            pointer.onDragStart = (pointer) =>
              this._startDraggingItems(newReroute, pointer)
            pointer.onDragEnd = (e) => this._processDraggedItems(e)
            return
          }
        } else if (
          this.linkMarkerShape !== LinkMarkerShape.None &&
          isInRectangle(x, y, centre[0] - 4, centre[1] - 4, 8, 8)
        ) {
          this.ctx.lineWidth = lineWidth

          pointer.onClick = () => this.showLinkMenu(linkSegment, e)
          pointer.onDragStart = () => (this.dragging_canvas = true)
          pointer.finally = () => (this.dragging_canvas = false)

          // clear tooltip
          this.over_link_center = undefined
          return
        }
      }

      // Restore line width
      this.ctx.lineWidth = lineWidth

      // Groups
      const group = graph.getGroupOnPos(x, y)
      this.selected_group = group ?? null
      if (group) {
        if (group.isInResize(x, y)) {
          // Resize group
          const b = group.boundingRect
          const offsetX = x - (b[0] + b[2])
          const offsetY = y - (b[1] + b[3])

          pointer.onDragStart = () => (this.resizingGroup = group)
          pointer.onDrag = (eMove) => {
            if (this.read_only) return

            // Resize only by the exact pointer movement
            const pos: Point = [
              eMove.canvasX - group.pos[0] - offsetX,
              eMove.canvasY - group.pos[1] - offsetY
            ]
            // Unless snapping.
            if (this._snapToGrid) snapPoint(pos, this._snapToGrid)

            const resized = group.resize(pos[0], pos[1])
            if (resized) this.dirty_bgcanvas = true
          }
          pointer.finally = () => (this.resizingGroup = null)
        } else {
          const f = group.font_size || LiteGraph.DEFAULT_GROUP_FONT_SIZE
          const headerHeight = f * 1.4
          if (
            isInRectangle(
              x,
              y,
              group.pos[0],
              group.pos[1],
              group.size[0],
              headerHeight
            )
          ) {
            // In title bar
            pointer.onClick = () => this.processSelect(group, e)
            pointer.onDragStart = (pointer) => {
              group.recomputeInsideNodes()
              this._startDraggingItems(group, pointer, true)
            }
            pointer.onDragEnd = (e) => this._processDraggedItems(e)
          }
        }

        pointer.onDoubleClick = () => {
          this.emitEvent({
            subType: 'group-double-click',
            originalEvent: e,
            group
          })
        }
      } else {
        pointer.onDoubleClick = () => {
          // Double click within group should not trigger the searchbox.
          if (this.allow_searchbox) {
            this.showSearchBox(e)
            e.preventDefault()
          }
          this.emitEvent({
            subType: 'empty-double-click',
            originalEvent: e
          })
        }
      }
    }

    if (
      !pointer.onDragStart &&
      !pointer.onClick &&
      !pointer.onDrag &&
      this.allow_dragcanvas
    ) {
      // allow dragging canvas based on leftMouseClickBehavior or read-only mode
      if (LiteGraph.leftMouseClickBehavior === 'panning' || this.read_only) {
        pointer.onClick = () => this.processSelect(null, e)
        pointer.finally = () => (this.dragging_canvas = false)
        this.dragging_canvas = true
      } else {
        this._setupNodeSelectionDrag(e, pointer)
      }
    }
  }

  private _setupNodeSelectionDrag(
    e: CanvasPointerEvent,
    pointer: CanvasPointer,
    node?: LGraphNode | undefined
  ): void {
    const dragRect: Rect = [0, 0, 0, 0]

    dragRect[0] = e.canvasX
    dragRect[1] = e.canvasY
    dragRect[2] = 1
    dragRect[3] = 1

    pointer.onClick = (eUp) => {
      // Click, not drag
      const clickedItem =
        node ?? this._getPositionableOnPos(eUp.canvasX, eUp.canvasY)
      this.processSelect(clickedItem, eUp)
    }
    pointer.onDragStart = () => (this.dragging_rectangle = dragRect)

    if (this.liveSelection) {
      const initialSelection = new Set(this.selectedItems)

      pointer.onDrag = (eMove) =>
        this.handleLiveSelect(eMove, dragRect, initialSelection)

      pointer.onDragEnd = () => this.finalizeLiveSelect()
    } else {
      // Classic mode: select only when drag ends
      pointer.onDragEnd = (upEvent) =>
        this._handleMultiSelect(upEvent, dragRect)
    }

    pointer.finally = () => (this.dragging_rectangle = null)
  }

  /**
   * Processes a pointerdown event inside the bounds of a node.  Part of {@link processMouseDown}.
   * @param e The pointerdown event
   * @param ctrlOrMeta Ctrl or meta key is pressed
   * @param node The node to process a click event for
   */
  private _processNodeClick(
    e: CanvasPointerEvent,
    ctrlOrMeta: boolean,
    node: LGraphNode
  ): void {
    // In Vue nodes mode, Vue components own all node-level interactions
    // Skip LiteGraph handling to prevent dual event processing
    if (LiteGraph.vueNodesMode) {
      return
    }

    const { pointer, graph, linkConnector } = this
    if (!graph) throw new NullGraphError()

    const x = e.canvasX
    const y = e.canvasY

    pointer.onClick = () => this.processSelect(node, e)

    // Immediately bring to front
    if (!node.flags.pinned) {
      this.bringToFront(node)
    }

    // Collapse toggle
    const inCollapse = node.isPointInCollapse(x, y)
    if (inCollapse) {
      pointer.onClick = () => {
        node.collapse()
        this.setDirty(true, true)
      }
    } else if (!node.flags.collapsed) {
      const { inputs, outputs } = node

      function hasRelevantOutputLinks(
        output: INodeOutputSlot,
        network: LinkNetwork
      ): boolean {
        const outputLinks = [
          ...(output.links ?? []),
          ...[...(output._floatingLinks ?? new Set())]
        ]
        return outputLinks.some(
          (linkId) =>
            typeof linkId === 'number' && network.getLink(linkId) !== undefined
        )
      }

      // Outputs
      if (outputs) {
        for (const [i, output] of outputs.entries()) {
          const link_pos = node.getOutputPos(i)
          if (isInRectangle(x, y, link_pos[0] - 15, link_pos[1] - 10, 30, 20)) {
            // Drag multiple output links
            if (e.shiftKey && hasRelevantOutputLinks(output, graph)) {
              linkConnector.moveOutputLink(graph, output)
              this._linkConnectorDrop()
              return
            }

            // New output link
            linkConnector.dragNewFromOutput(graph, node, output)
            this._linkConnectorDrop()

            if (LiteGraph.shift_click_do_break_link_from) {
              if (e.shiftKey) {
                node.disconnectOutput(i)
              }
            } else if (LiteGraph.ctrl_alt_click_do_break_link) {
              if (ctrlOrMeta && e.altKey && !e.shiftKey) {
                node.disconnectOutput(i)
              }
            }

            // TODO: Move callbacks to the start of this closure (onInputClick is already correct).
            pointer.onDoubleClick = () => node.onOutputDblClick?.(i, e)
            pointer.onClick = () => node.onOutputClick?.(i, e)

            return
          }
        }
      }

      // Inputs
      if (inputs) {
        for (const [i, input] of inputs.entries()) {
          const link_pos = node.getInputPos(i)
          const isInSlot =
            input instanceof NodeInputSlot
              ? isInRect(x, y, input.boundingRect)
              : isInRectangle(x, y, link_pos[0] - 15, link_pos[1] - 10, 30, 20)

          if (isInSlot) {
            pointer.onDoubleClick = () => node.onInputDblClick?.(i, e)
            pointer.onClick = () => node.onInputClick?.(i, e)

            const shouldBreakLink =
              LiteGraph.ctrl_alt_click_do_break_link &&
              ctrlOrMeta &&
              e.altKey &&
              !e.shiftKey
            if (input.link !== null || input._floatingLinks?.size) {
              // Existing link
              if (shouldBreakLink || LiteGraph.click_do_break_link_to) {
                node.disconnectInput(i, true)
              } else if (e.shiftKey || this.allow_reconnect_links) {
                linkConnector.moveInputLink(graph, input)
              }
            }

            // Dragging a new link from input to output
            if (!linkConnector.isConnecting) {
              linkConnector.dragNewFromInput(graph, node, input)
            }

            this._linkConnectorDrop()
            this.dirty_bgcanvas = true

            return
          }
        }
      }
    }

    // Click was inside the node, but not on input/output, or resize area
    const pos: Point = [x - node.pos[0], y - node.pos[1]]

    // Widget
    const widget = node.getWidgetOnPos(x, y)
    if (widget) {
      this.processWidgetClick(e, node, widget)
      this.node_widget = [node, widget]
    } else {
      // Node background
      pointer.onDoubleClick = () => {
        // Double-click
        // Check if it's a double click on the title bar
        // Note: pos[1] is the y-coordinate of the node's body
        // If clicking on node header (title), pos[1] is negative
        if (pos[1] < 0 && !inCollapse) {
          node.onNodeTitleDblClick?.(e, pos, this)
        } else if (node instanceof SubgraphNode) {
          this.openSubgraph(node.subgraph, node)
        }

        node.onDblClick?.(e, pos, this)
        this.emitEvent({
          subType: 'node-double-click',
          originalEvent: e,
          node
        })
        this.processNodeDblClicked(node)
      }

      // Check for title button clicks before calling onMouseDown
      if (node.title_buttons?.length && !node.flags.collapsed) {
        // pos contains the offset from the node's position, so we need to use node-relative coordinates
        const nodeRelativeX = pos[0]
        const nodeRelativeY = pos[1]

        for (let i = 0; i < node.title_buttons.length; i++) {
          const button = node.title_buttons[i]
          if (
            button.visible &&
            button.isPointInside(nodeRelativeX, nodeRelativeY)
          ) {
            node.onTitleButtonClick(button, this)
            // Set a no-op click handler to prevent fallback canvas dragging
            pointer.onClick = () => {}
            return
          }
        }
      }
      for (const badge of node.badges.map(toValue).filter((b) => b.onClick)) {
        if (isInRect(pos[0], pos[1], badge.boundingRect)) {
          pointer.onClick = badge.onClick
          return
        }
      }

      // Mousedown callback - can block drag
      if (node.onMouseDown?.(e, pos, this)) {
        // Node handled the event (e.g., title button clicked)
        // Set a no-op click handler to prevent fallback canvas dragging
        pointer.onClick = () => {}
        return
      }

      if (!this.allow_dragnodes) return

      // Check for resize AFTER checking all other interaction areas
      if (!node.flags.collapsed) {
        const resizeDirection = node.findResizeDirection(x, y)
        if (resizeDirection) {
          pointer.resizeDirection = resizeDirection
          const startBounds = new Rectangle(
            node.pos[0],
            node.pos[1],
            node.size[0],
            node.size[1]
          )

          pointer.onDragStart = () => {
            graph.beforeChange()
            this.resizing_node = node
          }

          pointer.onDrag = (eMove) => {
            if (this.read_only) return

            const deltaX = eMove.canvasX - x
            const deltaY = eMove.canvasY - y

            const newBounds = new Rectangle(
              startBounds.x,
              startBounds.y,
              startBounds.width,
              startBounds.height
            )

            // Handle resize based on the direction
            switch (resizeDirection) {
              case 'NE': // North-East (top-right)
                newBounds.y = startBounds.y + deltaY
                newBounds.width = startBounds.width + deltaX
                newBounds.height = startBounds.height - deltaY
                break
              case 'SE': // South-East (bottom-right)
                newBounds.width = startBounds.width + deltaX
                newBounds.height = startBounds.height + deltaY
                break
              case 'SW': // South-West (bottom-left)
                newBounds.x = startBounds.x + deltaX
                newBounds.width = startBounds.width - deltaX
                newBounds.height = startBounds.height + deltaY
                break
              case 'NW': // North-West (top-left)
                newBounds.x = startBounds.x + deltaX
                newBounds.y = startBounds.y + deltaY
                newBounds.width = startBounds.width - deltaX
                newBounds.height = startBounds.height - deltaY
                break
            }

            // Apply snapping to position changes
            if (this._snapToGrid) {
              if (
                resizeDirection.includes('N') ||
                resizeDirection.includes('W')
              ) {
                const originalX = newBounds.x
                const originalY = newBounds.y

                snapPoint(newBounds.pos, this._snapToGrid)

                // Adjust size to compensate for snapped position
                if (resizeDirection.includes('N')) {
                  newBounds.height += originalY - newBounds.y
                }
                if (resizeDirection.includes('W')) {
                  newBounds.width += originalX - newBounds.x
                }
              }

              snapPoint(newBounds.size, this._snapToGrid)
            }

            // Apply snapping to size changes

            // Enforce minimum size
            const min = node.computeSize()
            if (newBounds.width < min[0]) {
              // If resizing from left, adjust position to maintain right edge
              if (resizeDirection.includes('W')) {
                newBounds.x = startBounds.x + startBounds.width - min[0]
              }
              newBounds.width = min[0]
            }
            if (newBounds.height < min[1]) {
              // If resizing from top, adjust position to maintain bottom edge
              if (resizeDirection.includes('N')) {
                newBounds.y = startBounds.y + startBounds.height - min[1]
              }
              newBounds.height = min[1]
            }

            node.pos = newBounds.pos
            node.setSize(newBounds.size)

            this._dirty()
          }

          pointer.onDragEnd = () => {
            this._dirty()
            graph.afterChange(node)
          }
          pointer.finally = () => {
            this.resizing_node = null
            pointer.resizeDirection = undefined
          }

          // Set appropriate cursor for resize direction
          this.canvas.style.cursor = cursors[resizeDirection]
          return
        }
      }

      // Drag node
      pointer.onDragStart = (pointer) =>
        this._startDraggingItems(node, pointer, true)
      pointer.onDragEnd = (e) => this._processDraggedItems(e)
    }

    this.dirty_canvas = true
  }

  processWidgetClick(
    e: CanvasPointerEvent,
    node: LGraphNode,
    widget: IBaseWidget,
    pointer = this.pointer
  ) {
    // Custom widget - CanvasPointer
    if (typeof widget.onPointerDown === 'function') {
      const handled = widget.onPointerDown(pointer, node, this)
      if (handled) return
    }

    const oldValue = widget.value

    const pos = this.graph_mouse
    const x = pos[0] - node.pos[0]
    const y = pos[1] - node.pos[1]

    const widgetInstance = toConcreteWidget(widget, node, false)
    if (widgetInstance) {
      pointer.onClick = () =>
        widgetInstance.onClick({
          e,
          node,
          canvas: this
        })
      pointer.onDrag = (eMove) =>
        widgetInstance.onDrag?.({
          e: eMove,
          node,
          canvas: this
        })
    } else if (widget.mouse) {
      const result = widget.mouse(e, [x, y], node)
      if (result != null) this.dirty_canvas = result
    }

    // value changed
    if (oldValue != widget.value) {
      node.onWidgetChanged?.(widget.name, widget.value, oldValue, widget)
      if (!node.graph) throw new NullGraphError()
      node.graph._version++
    }

    // Clean up state var
    pointer.finally = () => {
      // Legacy custom widget callback
      if (widget.mouse) {
        const { eUp } = pointer
        if (!eUp) return
        const { canvasX, canvasY } = eUp
        widget.mouse(eUp, [canvasX - node.pos[0], canvasY - node.pos[1]], node)
      }

      this.node_widget = null
    }
  }

  /**
   * Pointer middle button click processing.  Part of {@link processMouseDown}.
   * @param e The pointerdown event
   * @param node The node to process a click event for
   */
  private _processMiddleButton(
    e: CanvasPointerEvent,
    node: LGraphNode | undefined
  ) {
    const { pointer } = this

    if (
      LiteGraph.middle_click_slot_add_default_node &&
      node &&
      this.allow_interaction &&
      !this.read_only &&
      !this.connecting_links &&
      !node.flags.collapsed
    ) {
      // not dragging mouse to connect two slots
      let mClikSlot: INodeSlot | false = false
      let mClikSlot_index: number | false = false
      let mClikSlot_isOut: boolean = false
      const { inputs, outputs } = node

      // search for outputs
      if (outputs) {
        for (const [i, output] of outputs.entries()) {
          const link_pos = node.getOutputPos(i)
          if (
            isInRectangle(
              e.canvasX,
              e.canvasY,
              link_pos[0] - 15,
              link_pos[1] - 10,
              30,
              20
            )
          ) {
            mClikSlot = output
            mClikSlot_index = i
            mClikSlot_isOut = true
            break
          }
        }
      }

      // search for inputs
      if (inputs) {
        for (const [i, input] of inputs.entries()) {
          const link_pos = node.getInputPos(i)
          if (
            isInRectangle(
              e.canvasX,
              e.canvasY,
              link_pos[0] - 15,
              link_pos[1] - 10,
              30,
              20
            )
          ) {
            mClikSlot = input
            mClikSlot_index = i
            mClikSlot_isOut = false
            break
          }
        }
      }
      // Middle clicked a slot
      if (mClikSlot && mClikSlot_index !== false) {
        const alphaPosY =
          0.5 -
          (mClikSlot_index + 1) /
            (mClikSlot_isOut ? outputs.length : inputs.length)
        const node_bounding = node.getBounding()
        // estimate a position: this is a bad semi-bad-working mess .. REFACTOR with
        // a correct autoplacement that knows about the others slots and nodes
        const posRef: Point = [
          !mClikSlot_isOut
            ? node_bounding[0]
            : node_bounding[0] + node_bounding[2],
          e.canvasY - 80
        ]

        pointer.onClick = () =>
          this.createDefaultNodeForSlot({
            nodeFrom: !mClikSlot_isOut ? null : node,
            slotFrom: !mClikSlot_isOut ? null : mClikSlot_index,
            nodeTo: !mClikSlot_isOut ? node : null,
            slotTo: !mClikSlot_isOut ? mClikSlot_index : null,
            position: posRef,
            nodeType: 'AUTO',
            posAdd: [!mClikSlot_isOut ? -30 : 30, -alphaPosY * 130],
            posSizeFix: [!mClikSlot_isOut ? -1 : 0, 0]
          })
      }
    }

    // Drag canvas using middle mouse button
    if (this.allow_dragcanvas) {
      pointer.onDragStart = () => (this.dragging_canvas = true)
      pointer.finally = () => (this.dragging_canvas = false)
    }
  }

  private _processDragZoom(e: PointerEvent): void {
    // stop canvas zoom action
    if (!e.buttons) {
      this._finishDragZoom()
      return
    }

    const start = this._dragZoomStart
    if (!start) throw new TypeError('Drag-zoom state object was null')
    if (!this.graph) throw new NullGraphError()

    // calculate delta
    const deltaY = e.y - start.pos[1]
    const startScale = start.scale

    const scale = startScale - deltaY / 100

    this.ds.changeScale(scale, start.pos)
    this.graph.change()
  }

  private _finishDragZoom(): void {
    const start = this._dragZoomStart
    if (!start) return
    this._dragZoomStart = null
    this.read_only = start.readOnly
  }

  /**
   * Called when a mouse move event has to be processed
   */
  processMouseMove(e: PointerEvent): void {
    if (
      this.dragZoomEnabled &&
      e.ctrlKey &&
      e.shiftKey &&
      this._dragZoomStart
    ) {
      this._processDragZoom(e)
      return
    }

    if (this.autoresize) this.resize()

    if (this.set_canvas_dirty_on_mouse_event) this.dirty_canvas = true

    const { graph, resizingGroup, linkConnector, pointer, subgraph } = this
    if (!graph) return

    LGraphCanvas.active_canvas = this
    this.adjustMouseEvent(e)
    const mouse: Readonly<Point> = [e.clientX, e.clientY]
    this.mouse[0] = mouse[0]
    this.mouse[1] = mouse[1]
    const delta = [mouse[0] - this.last_mouse[0], mouse[1] - this.last_mouse[1]]
    this.last_mouse = mouse
    const { canvasX: x, canvasY: y } = e
    this.graph_mouse[0] = x
    this.graph_mouse[1] = y

    if (e.isPrimary) pointer.move(e)

    /** See {@link state}.{@link LGraphCanvasState.hoveringOver hoveringOver} */
    let underPointer = CanvasItem.Nothing
    if (subgraph) {
      underPointer |= subgraph.inputNode.onPointerMove(e)
      underPointer |= subgraph.outputNode.onPointerMove(e)
    }

    if (this.block_click) {
      e.preventDefault()
      return
    }

    e.dragging = this.last_mouse_dragging

    if (this.node_widget) {
      // Legacy widget mouse callbacks for pointermove events
      const [node, widget] = this.node_widget

      if (widget?.mouse) {
        const relativeX = x - node.pos[0]
        const relativeY = y - node.pos[1]
        const result = widget.mouse(e, [relativeX, relativeY], node)
        if (result != null) this.dirty_canvas = result
      }
    }

    // get node over
    const node = LiteGraph.vueNodesMode
      ? null
      : graph.getNodeOnPos(x, y, this.visible_nodes)

    const dragRect = this.dragging_rectangle
    if (dragRect) {
      dragRect[2] = x - dragRect[0]
      dragRect[3] = y - dragRect[1]
      this.dirty_canvas = true
    } else if (resizingGroup) {
      // Resizing a group
      underPointer |= CanvasItem.Group
      pointer.resizeDirection = 'SE'
    } else if (this.dragging_canvas) {
      this.ds.offset[0] += delta[0] / this.ds.scale
      this.ds.offset[1] += delta[1] / this.ds.scale
      this._dirty()
    } else if (
      (this.allow_interaction || node?.flags.allow_interaction) &&
      !this.read_only
    ) {
      if (linkConnector.isConnecting) this.dirty_canvas = true

      // remove mouseover flag
      this.updateMouseOverNodes(node, e)

      // mouse over a node
      if (node) {
        underPointer |= CanvasItem.Node

        if (node.redraw_on_mouse) this.dirty_canvas = true

        // For input/output hovering
        // to store the output of isOverNodeInput
        const pos: Point = [0, 0]

        // Try to use layout store for hit testing first, fallback to old method
        let inputId: number = -1
        let outputId: number = -1

        const slotLayout = layoutStore.querySlotAtPoint({ x, y })
        if (slotLayout && slotLayout.nodeId === String(node.id)) {
          if (slotLayout.type === 'input') {
            inputId = slotLayout.index
            pos[0] = slotLayout.position.x
            pos[1] = slotLayout.position.y
          } else {
            outputId = slotLayout.index
            pos[0] = slotLayout.position.x
            pos[1] = slotLayout.position.y
          }
        } else {
          // Fallback to old method
          inputId = isOverNodeInput(node, x, y, pos)
          outputId = isOverNodeOutput(node, x, y, pos)
        }
        const overWidget = node.getWidgetOnPos(x, y, true) ?? undefined

        if (!node.mouseOver) {
          // mouse enter
          node.mouseOver = {}
          this.node_over = node
          this.dirty_canvas = true

          for (const reroute of this._visibleReroutes) {
            reroute.hideSlots()
            this.dirty_bgcanvas = true
          }
          node.onMouseEnter?.(e)
        }

        // in case the node wants to do something
        node.onMouseMove?.(e, [x - node.pos[0], y - node.pos[1]], this)

        // The input the mouse is over has changed
        const { mouseOver } = node
        if (
          mouseOver.inputId !== inputId ||
          mouseOver.outputId !== outputId ||
          mouseOver.overWidget !== overWidget
        ) {
          mouseOver.inputId = inputId
          mouseOver.outputId = outputId
          mouseOver.overWidget = overWidget

          // State reset
          linkConnector.overWidget = undefined

          // Check if link is over anything it could connect to - record position of valid target for snap / highlight
          if (linkConnector.isConnecting) {
            const firstLink = linkConnector.renderLinks.at(0)

            // Default: nothing highlighted
            let highlightPos: Point | undefined
            let highlightInput: INodeInputSlot | undefined

            if (!firstLink || !linkConnector.isNodeValidDrop(node)) {
              // No link, or none of the dragged links may be dropped here
            } else if (linkConnector.state.connectingTo === 'input') {
              if (overWidget) {
                // Check widgets first - inputId is only valid if over the input socket
                const slot = node.getSlotFromWidget(overWidget)

                if (slot && linkConnector.isInputValidDrop(node, slot)) {
                  highlightInput = slot
                  if (LiteGraph.vueNodesMode) {
                    const idx = node.inputs.indexOf(slot)
                    highlightPos =
                      idx !== -1
                        ? getSlotPosition(node, idx, true)
                        : node.getInputSlotPos(slot)
                  } else {
                    highlightPos = node.getInputSlotPos(slot)
                  }
                  linkConnector.overWidget = overWidget
                }
              }

              // Not over a valid widget - treat drop on invalid widget same as node background
              if (!linkConnector.overWidget) {
                if (inputId === -1 && outputId === -1) {
                  // Node background / title under the pointer
                  const result = node.findInputByType(firstLink.fromSlot.type)
                  if (result) {
                    highlightInput = result.slot
                    highlightPos = LiteGraph.vueNodesMode
                      ? getSlotPosition(node, result.index, true)
                      : node.getInputSlotPos(result.slot)
                  }
                } else if (
                  inputId != -1 &&
                  node.inputs[inputId] &&
                  LiteGraph.isValidConnection(
                    firstLink.fromSlot.type,
                    node.inputs[inputId].type
                  )
                ) {
                  highlightPos = pos
                  // XXX CHECK THIS
                  highlightInput = node.inputs[inputId]
                }

                if (highlightInput) {
                  const widget = node.getWidgetFromSlot(highlightInput)
                  if (widget) linkConnector.overWidget = widget
                }
              }
            } else if (linkConnector.state.connectingTo === 'output') {
              // Connecting from an input to an output
              if (inputId === -1 && outputId === -1) {
                const result = node.findOutputByType(firstLink.fromSlot.type)
                if (result) {
                  highlightPos = LiteGraph.vueNodesMode
                    ? getSlotPosition(node, result.index, false)
                    : node.getOutputPos(result.index)
                }
              } else {
                // check if I have a slot below de mouse
                if (
                  outputId != -1 &&
                  node.outputs[outputId] &&
                  LiteGraph.isValidConnection(
                    firstLink.fromSlot.type,
                    node.outputs[outputId].type
                  )
                ) {
                  highlightPos = pos
                }
              }
            }
            this._highlight_pos = highlightPos
            this._highlight_input = highlightInput
          }

          this.dirty_canvas = true
        }

        // Resize direction - only show resize cursor if not over inputs/outputs/widgets
        if (!pointer.eDown) {
          if (inputId === -1 && outputId === -1 && !overWidget) {
            pointer.resizeDirection = node.findResizeDirection(x, y)
          } else {
            // Clear resize direction when over inputs/outputs/widgets
            pointer.resizeDirection &&= undefined
          }
        }
      } else {
        // Reroutes
        underPointer = this._updateReroutes(underPointer)

        // Not over a node
        const segment = this._getLinkCentreOnPos(e)
        if (this.over_link_center !== segment) {
          underPointer |= CanvasItem.Link
          this.over_link_center = segment
          this.dirty_bgcanvas = true
        }

        if (this.canvas) {
          const group = graph.getGroupOnPos(x, y)
          if (
            group &&
            !e.ctrlKey &&
            !this.read_only &&
            group.isInResize(x, y)
          ) {
            pointer.resizeDirection = 'SE'
          } else {
            pointer.resizeDirection &&= undefined
          }
        }
      }

      // send event to node if capturing input (used with widgets that allow drag outside of the area of the node)
      if (this.node_capturing_input && this.node_capturing_input != node) {
        this.node_capturing_input.onMouseMove?.(
          e,
          [
            x - this.node_capturing_input.pos[0],
            y - this.node_capturing_input.pos[1]
          ],
          this
        )
      }

      // Items being dragged
      if (this.isDragging) {
        const selected = this.selectedItems
        const allItems = e.ctrlKey ? selected : getAllNestedItems(selected)

        const deltaX = delta[0] / this.ds.scale
        const deltaY = delta[1] / this.ds.scale

        if (LiteGraph.vueNodesMode) {
          this.moveChildNodesInGroupVueMode(allItems, deltaX, deltaY)
        } else {
          for (const item of allItems) {
            item.move(deltaX, deltaY, true)
          }
        }

        this._dirty()
      }
    }

    this.hoveringOver = underPointer

    e.preventDefault()
    return
  }

  /**
   * Updates the hover / snap state of all visible reroutes.
   * @returns The original value of {@link underPointer}, with any found reroute items added.
   */
  private _updateReroutes(underPointer: CanvasItem): CanvasItem {
    const { graph, pointer, linkConnector } = this
    if (!graph) throw new NullGraphError()

    // Update reroute hover state
    if (!pointer.isDown) {
      let anyChanges = false
      for (const reroute of this._visibleReroutes) {
        anyChanges ||= reroute.updateVisibility(this.graph_mouse)

        if (reroute.isSlotHovered) underPointer |= CanvasItem.RerouteSlot
      }
      if (anyChanges) this.dirty_bgcanvas = true
    } else if (linkConnector.isConnecting) {
      // Highlight the reroute that the mouse is over
      for (const reroute of this._visibleReroutes) {
        if (reroute.containsPoint(this.graph_mouse)) {
          if (linkConnector.isRerouteValidDrop(reroute)) {
            linkConnector.overReroute = reroute
            this._highlight_pos = reroute.pos
          }

          return (underPointer |= CanvasItem.RerouteSlot)
        }
      }
    }

    this._highlight_pos &&= undefined
    linkConnector.overReroute &&= undefined
    return underPointer
  }

  /**
   * Start dragging an item, optionally including all other selected items.
   *
   * ** This function sets the {@link CanvasPointer.finally}() callback. **
   * @param item The item that the drag event started on
   * @param pointer The pointer event that initiated the drag, e.g. pointerdown
   * @param sticky If `true`, the item is added to the selection - see {@link processSelect}
   */
  private _startDraggingItems(
    item: Positionable,
    pointer: CanvasPointer,
    sticky = false
  ): void {
    this.emitBeforeChange()
    this.graph?.beforeChange()
    // Ensure that dragging is properly cleaned up, on success or failure.
    pointer.finally = () => {
      this.isDragging = false
      this.graph?.afterChange()
      this.emitAfterChange()
    }

    this.processSelect(item, pointer.eDown, sticky)
    this.isDragging = true
  }

  /**
   * Handles shared clean up and placement after items have been dragged.
   * @param e The event that completed the drag, e.g. pointerup, pointermove
   */
  private _processDraggedItems(e: CanvasPointerEvent): void {
    const { graph } = this
    if (e.shiftKey || LiteGraph.alwaysSnapToGrid)
      graph?.snapToGrid(this.selectedItems)

    this.dirty_canvas = true
    this.dirty_bgcanvas = true

    // TODO: Replace legacy behaviour: callbacks were never extended for multiple items
    this.onNodeMoved?.(findFirstNode(this.selectedItems))
  }

  /**
   * Starts ghost placement mode for a node.
   * The node will be semi-transparent and follow the cursor until the user
   * clicks to place it, or presses Escape/right-clicks to cancel.
   * @param node The node to place
   * @param dragEvent Optional mouse event for positioning under cursor
   */
  startGhostPlacement(node: LGraphNode, dragEvent?: MouseEvent): void {
    this.emitBeforeChange()
    this.graph?.beforeChange()

    if (dragEvent) {
      this.adjustMouseEvent(dragEvent)
      const e = dragEvent as CanvasPointerEvent
      node.setPos(e.canvasX - node.size[0] / 2, e.canvasY + 10)
      // Update last_mouse to prevent jump on first drag move
      this.last_mouse = [e.clientX, e.clientY]
    } else {
      node.setPos(
        this.graph_mouse[0] - node.size[0] / 2,
        this.graph_mouse[1] + 10
      )
    }

    this.state.ghostNodeId = node.id

    this.deselectAll()
    this.select(node)
    this.isDragging = true
  }

  /**
   * Finalizes ghost placement mode.
   * @param cancelled If true, the node is removed; otherwise it's placed
   */
  finalizeGhostPlacement(cancelled: boolean): void {
    const nodeId = this.state.ghostNodeId
    if (nodeId == null) return

    this.state.ghostNodeId = null
    this.isDragging = false

    const node = this.graph?.getNodeById(nodeId)
    if (!node) return

    if (cancelled) {
      this.deselect(node)
      this.graph?.remove(node)
    } else {
      delete node.flags.ghost
      this.graph?.trigger('node:property:changed', {
        nodeId: node.id,
        property: 'flags.ghost',
        oldValue: true,
        newValue: false
      })

      this.state.selectionChanged = true
      this.onSelectionChange?.(this.selected_nodes)
    }

    this.dirty_canvas = true
    this.dirty_bgcanvas = true

    this.graph?.afterChange()
    this.emitAfterChange()
  }

  /**
   * Called when a mouse up event has to be processed
   */
  processMouseUp(e: PointerEvent): void {
    // early exit for extra pointer
    if (e.isPrimary === false) return

    const { graph, pointer } = this
    if (!graph) return

    this._finishDragZoom()

    LGraphCanvas.active_canvas = this

    this.adjustMouseEvent(e)

    const now = LiteGraph.getTime()
    e.click_time = now - this.last_mouseclick

    /** The mouseup event occurred near the mousedown event. */
    /** Normal-looking click event - mouseUp occurred near mouseDown, without dragging. */
    const isClick = pointer.up(e)
    if (isClick === true) {
      pointer.isDown = false
      pointer.isDouble = false
      // Required until all link behaviour is added to Pointer API
      this.connecting_links = null
      this.dragging_canvas = false

      graph.change()

      e.stopPropagation()
      e.preventDefault()
      return
    }

    this.last_mouse_dragging = false
    this.last_click_position = null

    // used to avoid sending twice a click in an immediate button
    this.block_click &&= false

    if (e.button === 0) {
      // left button
      this.selected_group = null

      this.isDragging = false

      const x = e.canvasX
      const y = e.canvasY

      if (!this.linkConnector.isConnecting) {
        this.dirty_canvas = true

        this.node_over?.onMouseUp?.(
          e,
          [x - this.node_over.pos[0], y - this.node_over.pos[1]],
          this
        )
        this.node_capturing_input?.onMouseUp?.(
          e,
          [
            x - this.node_capturing_input.pos[0],
            y - this.node_capturing_input.pos[1]
          ],
          this
        )
      }
    } else if (e.button === 1) {
      // middle button
      this.dirty_canvas = true
      this.dragging_canvas = false
    } else if (e.button === 2) {
      // right button
      this.dirty_canvas = true
    }

    pointer.isDown = false
    pointer.isDouble = false

    graph.change()

    e.stopPropagation()
    e.preventDefault()
    return
  }

  /**
   * Called when the mouse moves off the canvas.  Clears all node hover states.
   * @param e
   */
  processMouseOut(e: PointerEvent): void {
    // TODO: Check if document.contains(e.relatedTarget) - handle mouseover node textarea etc.
    this.adjustMouseEvent(e)
    this.updateMouseOverNodes(null, e)
  }

  processMouseCancel(): void {
    console.warn('Pointer cancel!')
    this.pointer.reset()
  }

  /**
   * Called when a mouse wheel event has to be processed
   */
  processMouseWheel(e: WheelEvent): void {
    if (!this.graph || !this.allow_dragcanvas) return

    this.adjustMouseEvent(e)

    const pos: Point = [e.clientX, e.clientY]
    if (this.viewport && !isPointInRect(pos, this.viewport)) return

    let { scale } = this.ds

    // Detect if this is a trackpad gesture or mouse wheel
    const isTrackpad = this.pointer.isTrackpadGesture(e)
    const isCtrlOrMacMeta =
      e.ctrlKey || (e.metaKey && navigator.platform.includes('Mac'))
    const isZoomModifier = isCtrlOrMacMeta && !e.altKey && !e.shiftKey

    if (isZoomModifier || LiteGraph.mouseWheelScroll === 'zoom') {
      // Zoom mode or modifier key pressed - use wheel for zoom
      if (isTrackpad) {
        // Trackpad gesture - use smooth scaling
        scale *= 1 + e.deltaY * (1 - this.zoom_speed) * 0.18
        this.ds.changeScale(scale, [e.clientX, e.clientY], false)
      } else {
        // Mouse wheel - use stepped scaling
        if (e.deltaY < 0) {
          scale *= this.zoom_speed
        } else if (e.deltaY > 0) {
          scale *= 1 / this.zoom_speed
        }
        this.ds.changeScale(scale, [e.clientX, e.clientY])
      }
    } else {
      // Trackpads and mice work on significantly different scales
      const factor = isTrackpad ? 0.18 : 0.008_333

      if (!isTrackpad && e.shiftKey && e.deltaX === 0) {
        this.ds.offset[0] -= e.deltaY * (1 + factor) * (1 / scale)
      } else {
        this.ds.offset[0] -= e.deltaX * (1 + factor) * (1 / scale)
        this.ds.offset[1] -= e.deltaY * (1 + factor) * (1 / scale)
      }
    }

    this.graph.change()

    e.preventDefault()
    return
  }

  private _noItemsSelected(): void {
    const event = new CustomEvent('litegraph:no-items-selected', {
      bubbles: true
    })
    this.canvas.dispatchEvent(event)
  }

  /**
   * process a key event
   */
  processKey(e: KeyboardEvent): void {
    this._shiftDown = e.shiftKey

    const { graph } = this
    if (!graph) return

    // Cancel ghost placement
    if (
      (e.key === 'Escape' || e.key === 'Delete' || e.key === 'Backspace') &&
      this.state.ghostNodeId != null
    ) {
      this.finalizeGhostPlacement(true)
      e.stopPropagation()
      e.preventDefault()
      return
    }

    let block_default = false
    // @ts-expect-error EventTarget.localName is not in standard types
    if (e.target.localName == 'input') return

    if (e.type == 'keydown') {
      // TODO: Switch
      if (e.key === ' ') {
        // space
        this.read_only = true
        if (this._previously_dragging_canvas === null) {
          this._previously_dragging_canvas = this.dragging_canvas
        }
        this.dragging_canvas = this.pointer.isDown
        block_default = true
      } else if (e.key === 'Escape') {
        // esc
        if (this.linkConnector.isConnecting) {
          this.linkConnector.reset()
          e.preventDefault()
          return
        }
        this.node_panel?.close()
        this.options_panel?.close()
        if (this.node_panel || this.options_panel) block_default = true
      } else if (e.keyCode === 65 && e.ctrlKey) {
        // select all Control A
        this.selectItems()
        block_default = true
      } else if (e.keyCode === 67 && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
        // copy
        if (this.selected_nodes) {
          this.copyToClipboard()
          block_default = true
        }
      } else if (e.keyCode === 86 && (e.metaKey || e.ctrlKey)) {
        // paste
        this.pasteFromClipboard({ connectInputs: e.shiftKey })
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        // delete or backspace
        // @ts-expect-error EventTarget.localName is not in standard types
        if (e.target.localName != 'input' && e.target.localName != 'textarea') {
          if (this.selectedItems.size === 0) {
            this._noItemsSelected()
            return
          }

          this.deleteSelected()
          block_default = true
        }
      }

      // TODO
      for (const node of Object.values(this.selected_nodes)) {
        node.onKeyDown?.(e)
      }
    } else if (e.type == 'keyup') {
      if (e.key === ' ') {
        // space
        this.read_only = false
        this.dragging_canvas =
          (this._previously_dragging_canvas ?? false) && this.pointer.isDown
        this._previously_dragging_canvas = null
      }

      for (const node of Object.values(this.selected_nodes)) {
        node.onKeyUp?.(e)
      }
    }

    // TODO: Do we need to remeasure and recalculate everything on every key down/up?
    graph.change()

    if (block_default) {
      e.preventDefault()
      e.stopImmediatePropagation()
    }
  }
  _serializeItems(items?: Iterable<Positionable>): ClipboardItems {
    const serialisable: Required<ClipboardItems> = {
      nodes: [],
      groups: [],
      reroutes: [],
      links: [],
      subgraphs: []
    }

    // NOTE: logic for traversing nested subgraphs depends on this being a set.
    const subgraphs = new Set<Subgraph>()

    // Create serialisable objects
    for (const item of items ?? this.selectedItems) {
      if (item instanceof LGraphNode) {
        // Nodes
        if (item.clonable === false) continue

        const cloned = item.clone()?.serialize()
        if (!cloned) continue

        cloned.id = item.id
        serialisable.nodes.push(cloned)

        // Links
        if (item.inputs) {
          for (const { link: linkId } of item.inputs) {
            if (linkId == null) continue

            const link = this.graph?._links.get(linkId)?.asSerialisable()
            if (link) serialisable.links.push(link)
          }
        }

        // Find all unique referenced subgraphs
        if (item instanceof SubgraphNode) {
          subgraphs.add(item.subgraph)
        }
      } else if (item instanceof LGraphGroup) {
        // Groups
        serialisable.groups.push(item.serialize())
      } else if (item instanceof Reroute) {
        // Reroutes
        serialisable.reroutes.push(item.asSerialisable())
      }
    }

    // Add unique subgraph entries
    // NOTE: subgraphs is appended to mid iteration.
    for (const subgraph of subgraphs) {
      for (const node of subgraph.nodes) {
        if (node instanceof SubgraphNode) {
          subgraphs.add(node.subgraph)
        }
      }
      const cloned = subgraph.clone(true).asSerialisable()
      serialisable.subgraphs.push(cloned)
    }
    return serialisable
  }

  /**
   * Copies canvas items to an internal, app-specific clipboard backed by local storage.
   * When called without parameters, it copies {@link selectedItems}.
   * @param items The items to copy.  If nullish, all selected items are copied.
   */
  copyToClipboard(items?: Iterable<Positionable>): string {
    const serializedData = JSON.stringify(this._serializeItems(items))
    localStorage.setItem('litegrapheditor_clipboard', serializedData)
    return serializedData
  }

  emitEvent(detail: LGraphCanvasEventMap['litegraph:canvas']): void {
    this.canvas.dispatchEvent(
      new CustomEvent('litegraph:canvas', {
        bubbles: true,
        detail
      })
    )
  }

  /** @todo Refactor to where it belongs - e.g. Deleting / creating nodes is not actually canvas event. */
  emitBeforeChange(): void {
    this.emitEvent({
      subType: 'before-change'
    })
  }

  /** @todo See {@link emitBeforeChange} */
  emitAfterChange(): void {
    this.emitEvent({
      subType: 'after-change'
    })
  }

  /**
   * Pastes the items from the canvas "clipbaord" - a local storage variable.
   */
  _pasteFromClipboard(
    options: IPasteFromClipboardOptions = {}
  ): ClipboardPasteResult | undefined {
    const data = localStorage.getItem('litegrapheditor_clipboard')
    if (!data) return
    return this._deserializeItems(JSON.parse(data), options)
  }

  _deserializeItems(
    parsed: ClipboardItems,
    options: IPasteFromClipboardOptions
  ): ClipboardPasteResult | undefined {
    const { connectInputs = false, position = this.graph_mouse } = options

    // if ctrl + shift + v is off, return when isConnectUnselected is true (shift is pressed) to maintain old behavior
    if (
      !LiteGraph.ctrl_shift_v_paste_connect_unselected_outputs &&
      connectInputs
    )
      return

    const { graph } = this
    if (!graph) throw new NullGraphError()
    graph.beforeChange()
    this.emitBeforeChange()

    // Parse & initialise
    parsed.nodes ??= []
    parsed.groups ??= []
    parsed.reroutes ??= []
    parsed.links ??= []
    parsed.subgraphs ??= []

    // Find top-left-most boundary
    let offsetX = Infinity
    let offsetY = Infinity
    for (const item of [...parsed.nodes, ...parsed.reroutes]) {
      if (item.pos == null)
        throw new TypeError(
          'Invalid node encountered on paste.  `pos` was null.'
        )

      if (item.pos[0] < offsetX) offsetX = item.pos[0]
      if (item.pos[1] < offsetY) offsetY = item.pos[1]
    }

    // TODO: Remove when implementing `asSerialisable`
    if (parsed.groups) {
      for (const group of parsed.groups) {
        if (group.bounding[0] < offsetX) offsetX = group.bounding[0]
        if (group.bounding[1] < offsetY) offsetY = group.bounding[1]
      }
    }

    const results: ClipboardPasteResult = {
      created: [],
      nodes: new Map<NodeId, LGraphNode>(),
      links: new Map<LinkId, LLink>(),
      reroutes: new Map<RerouteId, Reroute>(),
      subgraphs: new Map<UUID, Subgraph>()
    }
    const { created, nodes, links, reroutes } = results

    // const failedNodes: ISerialisedNode[] = []
    const subgraphIdMap: Record<string, string> = {}
    // SubgraphV2: Remove always-clone behaviour
    //Update subgraph ids
    for (const subgraphInfo of parsed.subgraphs)
      subgraphInfo.id = subgraphIdMap[subgraphInfo.id] = createUuidv4()
    const allNodeInfo: ISerialisedNode[] = [
      parsed.nodes ? [parsed.nodes] : [],
      parsed.subgraphs ? parsed.subgraphs.map((s) => s.nodes ?? []) : []
    ].flat(2)
    for (const nodeInfo of allNodeInfo)
      if (nodeInfo.type in subgraphIdMap)
        nodeInfo.type = subgraphIdMap[nodeInfo.type]
    remapClipboardSubgraphNodeIds(parsed, graph.rootGraph)

    // Subgraphs
    for (const info of parsed.subgraphs) {
      const subgraph = graph.createSubgraph(info)
      results.subgraphs.set(info.id, subgraph)
    }
    for (const info of parsed.subgraphs)
      results.subgraphs.get(info.id)?.configure(info)

    // Groups
    for (const info of parsed.groups) {
      info.id = -1

      const group = new LGraphGroup()
      group.configure(info)
      graph.add(group)
      created.push(group)
    }

    // Nodes
    for (const info of parsed.nodes) {
      const node = info.type == null ? null : LiteGraph.createNode(info.type)
      if (!node) {
        // failedNodes.push(info)
        continue
      }

      nodes.set(info.id, node)
      info.id = -1

      graph.add(node)
      node.configure(info)

      created.push(node)
    }

    // Reroutes
    for (const info of parsed.reroutes) {
      const { id, ...rerouteInfo } = info

      const reroute = graph.setReroute(rerouteInfo)
      created.push(reroute)
      reroutes.set(id, reroute)
    }

    // Remap reroute parentIds for pasted reroutes
    for (const reroute of reroutes.values()) {
      if (reroute.parentId == null) continue

      const mapped = reroutes.get(reroute.parentId)
      if (mapped) reroute.parentId = mapped.id
    }

    // Links
    for (const info of parsed.links) {
      // Find the copied node / reroute ID
      let outNode: LGraphNode | null | undefined = nodes.get(info.origin_id)
      let afterRerouteId: number | undefined
      if (info.parentId != null)
        afterRerouteId = reroutes.get(info.parentId)?.id

      // If it wasn't copied, use the original graph value
      if (
        connectInputs &&
        LiteGraph.ctrl_shift_v_paste_connect_unselected_outputs
      ) {
        outNode ??= graph.getNodeById(info.origin_id)
        afterRerouteId ??= info.parentId
      }

      const inNode = nodes.get(info.target_id)
      if (inNode) {
        const link = outNode?.connect(
          info.origin_slot,
          inNode,
          info.target_slot,
          afterRerouteId
        )
        if (link) links.set(info.id, link)
      }
    }

    // Remap linkIds
    for (const reroute of reroutes.values()) {
      const ids = [...reroute.linkIds].map((x) => links.get(x)?.id ?? x)
      reroute.update(reroute.parentId, undefined, ids, reroute.floating)

      // Remove any invalid items
      if (!reroute.validateLinks(graph.links, graph.floatingLinks)) {
        graph.removeReroute(reroute.id)
      }
    }

    // Adjust positions - use move/setPos to ensure layout store is updated
    const dx = position[0] - offsetX
    const dy = position[1] - offsetY
    for (const item of created) {
      if (item instanceof LGraphNode) {
        item.setPos(item.pos[0] + dx, item.pos[1] + dy)
      } else if (item instanceof Reroute) {
        item.move(dx, dy)
      }
    }

    // TODO: Report failures, i.e. `failedNodes`

    const newPositions = created
      .filter((item): item is LGraphNode => item instanceof LGraphNode)
      .map((node) => ({
        nodeId: String(node.id),
        bounds: {
          x: node.pos[0],
          y: node.pos[1],
          width: node.size?.[0] ?? 100,
          height: node.size?.[1] ?? 200
        }
      }))

    if (newPositions.length) layoutStore.setSource(LayoutSource.Canvas)
    layoutStore.batchUpdateNodeBounds(newPositions)

    this.selectItems(created)
    forEachNode(graph, (n) => n.onGraphConfigured?.())
    forEachNode(graph, (n) => n.onAfterGraphConfigured?.())

    graph.afterChange()
    this.emitAfterChange()

    return results
  }

  pasteFromClipboard(options: IPasteFromClipboardOptions = {}): void {
    this.emitBeforeChange()
    try {
      this._pasteFromClipboard(options)
    } finally {
      this.emitAfterChange()
    }
  }

  processNodeDblClicked(n: LGraphNode): void {
    this.onShowNodePanel?.(n)
    this.onNodeDblClicked?.(n)

    this.setDirty(true)
  }

  /**
   * Normalizes a drag rectangle to have positive width and height.
   * @param dragRect The drag rectangle to normalize (modified in place)
   * @returns The normalized rectangle
   */
  private _normalizeDragRect(dragRect: Rect): Rect {
    const w = Math.abs(dragRect[2])
    const h = Math.abs(dragRect[3])
    if (dragRect[2] < 0) dragRect[0] -= w
    if (dragRect[3] < 0) dragRect[1] -= h
    dragRect[2] = w
    dragRect[3] = h
    return dragRect
  }

  /**
   * Gets all positionable items that overlap with the given rectangle.
   * @param rect The rectangle to check against
   * @returns Set of positionable items that overlap with the rectangle
   */
  private _getItemsInRect(rect: Rect): Set<Positionable> {
    const { graph, subgraph } = this
    if (!graph) throw new NullGraphError()

    const items = new Set<Positionable>()

    if (subgraph) {
      const { inputNode, outputNode } = subgraph
      if (overlapBounding(rect, inputNode.boundingRect)) items.add(inputNode)
      if (overlapBounding(rect, outputNode.boundingRect)) items.add(outputNode)
    }

    for (const node of graph._nodes) {
      if (overlapBounding(rect, node.boundingRect)) items.add(node)
    }

    // Check groups (must be wholly inside)
    for (const group of graph.groups) {
      if (containsRect(rect, group._bounding)) {
        group.recomputeInsideNodes()
        items.add(group)
      }
    }

    // Check reroutes (center point must be inside)
    for (const reroute of graph.reroutes.values()) {
      if (isPointInRect(reroute.pos, rect)) items.add(reroute)
    }

    return items
  }

  /**
   * Handles live selection updates during drag. Called on each pointer move.
   * @param e The pointer move event
   * @param dragRect The current drag rectangle
   * @param initialSelection The selection state before the drag started
   */
  private handleLiveSelect(
    e: CanvasPointerEvent,
    dragRect: Rect,
    initialSelection: Set<Positionable>
  ): void {
    // Ensure rect is current even if pointer.onDrag fires before processMouseMove updates it
    dragRect[2] = e.canvasX - dragRect[0]
    dragRect[3] = e.canvasY - dragRect[1]

    // Create a normalized copy for overlap checking
    const normalizedRect: Rect = [
      dragRect[0],
      dragRect[1],
      dragRect[2],
      dragRect[3]
    ]
    this._normalizeDragRect(normalizedRect)

    const itemsInRect = this._getItemsInRect(normalizedRect)

    const desired = new Set<Positionable>()
    if (e.shiftKey && !e.altKey) {
      for (const item of initialSelection) desired.add(item)
      for (const item of itemsInRect) desired.add(item)
    } else if (e.altKey && !e.shiftKey) {
      for (const item of initialSelection)
        if (!itemsInRect.has(item)) desired.add(item)
    } else {
      for (const item of itemsInRect) desired.add(item)
    }

    let changed = false
    for (const item of [...this.selectedItems]) {
      if (!desired.has(item)) {
        this.deselect(item)
        changed = true
      }
    }
    for (const item of desired) {
      if (!this.selectedItems.has(item)) {
        this.select(item)
        changed = true
      }
    }

    if (changed) {
      this.onSelectionChange?.(this.selected_nodes)
      this.setDirty(true)
    }
  }

  /**
   * Finalizes the live selection when drag ends.
   */
  private finalizeLiveSelect(): void {
    // Selection is already updated by handleLiveSelect
    // Just trigger the final selection change callback
    this.onSelectionChange?.(this.selected_nodes)
  }

  /**
   * Handles multi-select when drag ends (classic mode).
   * @param e The pointer up event
   * @param dragRect The drag rectangle
   */
  private _handleMultiSelect(e: CanvasPointerEvent, dragRect: Rect): void {
    const normalizedRect: Rect = [
      dragRect[0],
      dragRect[1],
      dragRect[2],
      dragRect[3]
    ]
    this._normalizeDragRect(normalizedRect)

    const itemsInRect = this._getItemsInRect(normalizedRect)
    const { selectedItems } = this

    if (e.shiftKey) {
      // Add to selection
      for (const item of itemsInRect) this.select(item)
    } else if (e.altKey) {
      // Remove from selection
      for (const item of itemsInRect) this.deselect(item)
    } else {
      // Replace selection
      for (const item of selectedItems.values()) {
        if (!itemsInRect.has(item)) this.deselect(item)
      }
      for (const item of itemsInRect) this.select(item)
    }

    this.onSelectionChange?.(this.selected_nodes)
  }

  /**
   * Determines whether to select or deselect an item that has received a pointer event.  Will deselect other nodes if
   * @param item Canvas item to select/deselect
   * @param e The MouseEvent to handle
   * @param sticky Prevents deselecting individual nodes (as used by aux/right-click)
   * @remarks
   * Accessibility: anyone using {@link mutli_select} always deselects when clicking empty space.
   */
  processSelect<TPositionable extends Positionable = LGraphNode>(
    item: TPositionable | null | undefined,
    e: CanvasPointerEvent | undefined,
    sticky: boolean = false
  ): void {
    const addModifier = e?.shiftKey
    const subtractModifier = e != null && (e.metaKey || e.ctrlKey)
    const eitherModifier = addModifier || subtractModifier
    const modifySelection = eitherModifier || this.multi_select

    if (!item) {
      if (!eitherModifier || this.multi_select) this.deselectAll()
    } else if (!item.selected || !this.selectedItems.has(item)) {
      if (!modifySelection) this.deselectAll(item)
      this.select(item)
    } else if (modifySelection && !sticky) {
      this.deselect(item)
    } else if (!sticky) {
      this.deselectAll(item)
    } else {
      return
    }
    this.onSelectionChange?.(this.selected_nodes)
    this.setDirty(true)
  }

  /**
   * Selects a {@link Positionable} item.
   * @param item The canvas item to add to the selection.
   */
  select<TPositionable extends Positionable = LGraphNode>(
    item: TPositionable
  ): void {
    if (item.selected && this.selectedItems.has(item)) return

    item.selected = true
    this.selectedItems.add(item)
    this.state.selectionChanged = true

    if (item instanceof LGraphGroup) {
      item.recomputeInsideNodes()
      return
    }

    if (!(item instanceof LGraphNode)) return

    // Node-specific handling
    item.onSelected?.()
    this.selected_nodes[item.id] = item

    this.onNodeSelected?.(item)

    // Highlight links
    if (item.inputs) {
      for (const input of item.inputs) {
        if (input.link == null) continue
        this.highlighted_links[input.link] = true
      }
    }
    if (item.outputs) {
      for (const id of item.outputs.flatMap((x) => x.links)) {
        if (id == null) continue
        this.highlighted_links[id] = true
      }
    }
  }

  /**
   * Deselects a {@link Positionable} item.
   * @param item The canvas item to remove from the selection.
   */
  deselect<TPositionable extends Positionable = LGraphNode>(
    item: TPositionable
  ): void {
    if (!item.selected && !this.selectedItems.has(item)) return

    item.selected = false
    this.selectedItems.delete(item)
    this.state.selectionChanged = true
    if (!(item instanceof LGraphNode)) return

    // Node-specific handling
    item.onDeselected?.()
    delete this.selected_nodes[item.id]

    this.onNodeDeselected?.(item)

    // Should be moved to top of function, and throw if null
    const { graph } = this
    if (!graph) return

    // Clear link highlight
    if (item.inputs) {
      for (const input of item.inputs) {
        if (input.link == null) continue

        const node = LLink.getOriginNode(graph, input.link)
        if (node && this.selectedItems.has(node)) continue

        delete this.highlighted_links[input.link]
      }
    }
    if (item.outputs) {
      for (const id of item.outputs.flatMap((x) => x.links)) {
        if (id == null) continue

        const node = LLink.getTargetNode(graph, id)
        if (node && this.selectedItems.has(node)) continue

        delete this.highlighted_links[id]
      }
    }
  }

  /** @deprecated See {@link LGraphCanvas.processSelect} */
  processNodeSelected(item: LGraphNode, e: CanvasPointerEvent): void {
    this.processSelect(
      item,
      e,
      e && (e.shiftKey || e.metaKey || e.ctrlKey || this.multi_select)
    )
  }

  /** @deprecated See {@link LGraphCanvas.select} */
  selectNode(node: LGraphNode, add_to_current_selection?: boolean): void {
    if (node == null) {
      this.deselectAll()
    } else {
      this.selectNodes([node], add_to_current_selection)
    }
  }

  get empty(): boolean {
    if (!this.graph) throw new NullGraphError()
    return this.graph.empty
  }

  get positionableItems() {
    if (!this.graph) throw new NullGraphError()
    return this.graph.positionableItems()
  }

  /**
   * Selects several items.
   * @param items Items to select - if falsy, all items on the canvas will be selected
   * @param add_to_current_selection If set, the items will be added to the current selection instead of replacing it
   */
  selectItems(
    items?: Positionable[],
    add_to_current_selection?: boolean
  ): void {
    const itemsToSelect = items ?? this.positionableItems
    if (!add_to_current_selection) this.deselectAll()
    for (const item of itemsToSelect) this.select(item)
    this.onSelectionChange?.(this.selected_nodes)
    this.setDirty(true)
  }

  /**
   * selects several nodes (or adds them to the current selection)
   * @deprecated See {@link LGraphCanvas.selectItems}
   */
  selectNodes(nodes?: LGraphNode[], add_to_current_selection?: boolean): void {
    this.selectItems(nodes, add_to_current_selection)
  }

  /** @deprecated See {@link LGraphCanvas.deselect} */
  deselectNode(node: LGraphNode): void {
    this.deselect(node)
  }

  /**
   * Deselects all items on the canvas.
   * @param keepSelected If set, this item will not be removed from the selection.
   */
  deselectAll(keepSelected?: Positionable): void {
    if (!this.graph) return

    const selected = this.selectedItems
    if (!selected.size) return

    const initialSelectionSize = selected.size
    let wasSelected: Positionable | undefined
    for (const sel of selected) {
      if (sel === keepSelected) {
        wasSelected = sel
        continue
      }
      sel.onDeselected?.()
      sel.selected = false
    }
    selected.clear()
    if (wasSelected) selected.add(wasSelected)

    this.setDirty(true)

    // Legacy code
    const oldNode =
      keepSelected?.id == null ? null : this.selected_nodes[keepSelected.id]
    this.selected_nodes = {}
    this.current_node = null
    this.highlighted_links = {}

    if (keepSelected instanceof LGraphNode) {
      // Handle old object lookup
      if (oldNode) this.selected_nodes[oldNode.id] = oldNode

      // Highlight links
      if (keepSelected.inputs) {
        for (const input of keepSelected.inputs) {
          if (input.link == null) continue
          this.highlighted_links[input.link] = true
        }
      }
      if (keepSelected.outputs) {
        for (const id of keepSelected.outputs.flatMap((x) => x.links)) {
          if (id == null) continue
          this.highlighted_links[id] = true
        }
      }
    }

    // Only set selectionChanged if selection actually changed
    const finalSelectionSize = selected.size
    if (initialSelectionSize !== finalSelectionSize) {
      this.state.selectionChanged = true
      this.onSelectionChange?.(this.selected_nodes)
    }
  }

  /** @deprecated See {@link LGraphCanvas.deselectAll} */
  deselectAllNodes(): void {
    this.deselectAll()
  }

  /**
   * Deletes all selected items from the graph.
   * @todo Refactor deletion task to LGraph.  Selection is a canvas property, delete is a graph action.
   */
  deleteSelected(): void {
    const { graph } = this
    if (!graph) throw new NullGraphError()

    this.emitBeforeChange()
    graph.beforeChange()

    for (const item of this.selectedItems) {
      if (item instanceof LGraphNode) {
        const node = item
        if (node.block_delete) continue
        node.connectInputToOutput()
        graph.remove(node)
        this.onNodeDeselected?.(node)
      } else if (item instanceof LGraphGroup) {
        graph.remove(item)
      } else if (item instanceof Reroute) {
        graph.removeReroute(item.id)
      }
    }

    this.selected_nodes = {}
    this.selectedItems.clear()
    this.current_node = null
    this.highlighted_links = {}

    this.state.selectionChanged = true
    this.onSelectionChange?.(this.selected_nodes)
    this.setDirty(true)
    graph.afterChange()
    this.emitAfterChange()
  }

  /**
   * deletes all nodes in the current selection from the graph
   * @deprecated See {@link LGraphCanvas.deleteSelected}
   */
  deleteSelectedNodes(): void {
    this.deleteSelected()
  }

  /**
   * centers the camera on a given node
   */
  centerOnNode(node: LGraphNode): void {
    const dpi = window?.devicePixelRatio || 1
    this.ds.offset[0] =
      -node.pos[0] -
      node.size[0] * 0.5 +
      (this.canvas.width * 0.5) / (this.ds.scale * dpi)
    this.ds.offset[1] =
      -node.pos[1] -
      node.size[1] * 0.5 +
      (this.canvas.height * 0.5) / (this.ds.scale * dpi)
    this.setDirty(true, true)
  }

  /**
   * adds some useful properties to a mouse event, like the position in graph coordinates
   */
  adjustMouseEvent<T extends MouseEvent>(
    e: T & Partial<CanvasPointerExtensions>
  ): asserts e is T & CanvasPointerEvent {
    let clientX_rel = e.clientX
    let clientY_rel = e.clientY

    if (this.canvas) {
      const b = this.canvas.getBoundingClientRect()
      clientX_rel -= b.left
      clientY_rel -= b.top
    }

    e.safeOffsetX = clientX_rel
    e.safeOffsetY = clientY_rel

    // TODO: Find a less brittle way to do this

    // Only set deltaX and deltaY if not already set.
    // If deltaX and deltaY are already present, they are read-only.
    // Setting them would result browser error => zoom in/out feature broken.
    if (e.deltaX === undefined)
      e.deltaX = clientX_rel - this.last_mouse_position[0]
    if (e.deltaY === undefined)
      e.deltaY = clientY_rel - this.last_mouse_position[1]

    this.last_mouse_position[0] = clientX_rel
    this.last_mouse_position[1] = clientY_rel

    e.canvasX = clientX_rel / this.ds.scale - this.ds.offset[0]
    e.canvasY = clientY_rel / this.ds.scale - this.ds.offset[1]
  }

  /**
   * changes the zoom level of the graph (default is 1), you can pass also a place used to pivot the zoom
   */
  setZoom(value: number, zooming_center: Point) {
    this.ds.changeScale(value, zooming_center)
    this._dirty()
  }

  /**
   * converts a coordinate from graph coordinates to canvas2D coordinates
   */
  convertOffsetToCanvas(pos: Point, _out?: Point): Point {
    return this.ds.convertOffsetToCanvas(pos)
  }

  /**
   * converts a coordinate from Canvas2D coordinates to graph space
   */
  convertCanvasToOffset(pos: Point, out?: Point): Point {
    return this.ds.convertCanvasToOffset(pos, out)
  }

  // converts event coordinates from canvas2D to graph coordinates
  convertEventToCanvasOffset(e: MouseEvent): Point {
    const rect = this.canvas.getBoundingClientRect()
    // TODO: -> this.ds.convertCanvasToOffset
    return this.convertCanvasToOffset([
      e.clientX - rect.left,
      e.clientY - rect.top
    ])
  }

  /**
   * brings a node to front (above all other nodes)
   */
  bringToFront(node: LGraphNode): void {
    const { graph } = this
    if (!graph) throw new NullGraphError()

    const i = graph._nodes.indexOf(node)
    if (i == -1) return

    graph._nodes.splice(i, 1)
    graph._nodes.push(node)
  }

  /**
   * sends a node to the back (below all other nodes)
   */
  sendToBack(node: LGraphNode): void {
    const { graph } = this
    if (!graph) throw new NullGraphError()

    const i = graph._nodes.indexOf(node)
    if (i == -1) return

    graph._nodes.splice(i, 1)
    graph._nodes.unshift(node)
  }

  /**
   * Determines which nodes are visible and populates {@link out} with the results.
   * @param nodes The list of nodes to check - if falsy, all nodes in the graph will be checked
   * @param out Array to write visible nodes into - if falsy, a new array is created instead
   * @returns Array passed ({@link out}), or a new array containing all visible nodes
   */
  computeVisibleNodes(nodes?: LGraphNode[], out?: LGraphNode[]): LGraphNode[] {
    const visible_nodes = out || []
    visible_nodes.length = 0
    if (!this.graph) throw new NullGraphError()

    const _nodes = nodes || this.graph._nodes
    for (const node of _nodes) {
      node.updateArea(this.ctx)
      // Not in visible area
      if (!overlapBounding(this.visible_area, node.renderArea)) continue

      visible_nodes.push(node)
    }
    return visible_nodes
  }

  /**
   * Checks if a node is visible on the canvas.
   * @param node The node to check
   * @returns `true` if the node is visible, otherwise `false`
   */
  isNodeVisible(node: LGraphNode): boolean {
    return this._visible_node_ids.has(node.id)
  }

  /**
   * renders the whole canvas content, by rendering in two separated canvas, one containing the background grid and the connections, and one containing the nodes)
   */
  draw(force_canvas?: boolean, force_bgcanvas?: boolean): void {
    if (!this.canvas || this.canvas.width == 0 || this.canvas.height == 0)
      return

    // fps counting
    const now = LiteGraph.getTime()
    this.render_time = (now - this.last_draw_time) * 0.001
    this.last_draw_time = now

    if (this.graph) this.ds.computeVisibleArea(this.viewport)

    // Compute node size before drawing links.
    if (this.dirty_canvas || force_canvas) {
      this.computeVisibleNodes(undefined, this.visible_nodes)
      // Update visible node IDs
      this._visible_node_ids = new Set(
        this.visible_nodes.map((node) => node.id)
      )

      // Arrange subgraph IO nodes
      const { subgraph } = this
      if (subgraph) {
        subgraph.inputNode.arrange()
        subgraph.outputNode.arrange()
      }
    }

    if (
      this.dirty_bgcanvas ||
      force_bgcanvas ||
      this.always_render_background ||
      (this.graph?._last_trigger_time &&
        now - this.graph._last_trigger_time < 1000)
    ) {
      this.drawBackCanvas()
    }

    if (this.dirty_canvas || force_canvas) this.drawFrontCanvas()

    this.fps = this.render_time ? 1.0 / this.render_time : 0
    this.frame++
  }

  /**
   * draws the front canvas (the one containing all the nodes)
   */
  drawFrontCanvas(): void {
    this.dirty_canvas = false

    const { ctx, canvas, graph } = this

    // @ts-expect-error start2D method not in standard CanvasRenderingContext2D
    if (ctx.start2D && !this.viewport) {
      // @ts-expect-error start2D method not in standard CanvasRenderingContext2D
      ctx.start2D()
      ctx.restore()
      ctx.setTransform(1, 0, 0, 1, 0, 0)
    }

    // clip dirty area if there is one, otherwise work in full canvas
    const area = this.viewport || this.dirty_area
    if (area) {
      ctx.save()
      ctx.beginPath()
      ctx.rect(area[0], area[1], area[2], area[3])
      ctx.clip()
    }

    // TODO: Set snapping value when changed instead of once per frame
    this._snapToGrid =
      this._shiftDown || LiteGraph.alwaysSnapToGrid
        ? this.graph?.getSnapToGridSize()
        : undefined

    // clear
    // canvas.width = canvas.width;
    if (this.clear_background) {
      if (area) ctx.clearRect(area[0], area[1], area[2], area[3])
      else ctx.clearRect(0, 0, canvas.width, canvas.height)
    }

    // draw bg canvas
    if (this.bgcanvas == this.canvas) {
      this.drawBackCanvas()
    } else {
      const scale = window.devicePixelRatio
      ctx.drawImage(
        this.bgcanvas,
        0,
        0,
        this.bgcanvas.width / scale,
        this.bgcanvas.height / scale
      )
    }

    // rendering
    this.onRender?.(canvas, ctx)

    // info widget
    if (this.show_info) {
      const pos = this.fpsInfoLocation ?? area
      this.renderInfo(ctx, pos?.[0] ?? 0, pos?.[1] ?? 0)
    }

    if (graph) {
      // apply transformations
      ctx.save()
      this.ds.toCanvasContext(ctx)

      // draw nodes
      const { visible_nodes } = this
      const drawSnapGuides =
        this._snapToGrid &&
        (this.isDragging || layoutStore.isDraggingVueNodes.value)

      for (const node of visible_nodes) {
        ctx.save()

        // Draw snap shadow
        if (drawSnapGuides && this.selectedItems.has(node))
          this.drawSnapGuide(ctx, node)

        // Localise co-ordinates to node position
        ctx.translate(node.pos[0], node.pos[1])

        // Draw
        this.drawNode(node, ctx)

        ctx.restore()
      }

      // Draw subgraph IO nodes
      this.subgraph?.draw(
        ctx,
        this.colourGetter,
        this.linkConnector.renderLinks[0]?.fromSlot,
        this.editor_alpha
      )

      // on top (debug)
      if (this.render_execution_order) {
        this.drawExecutionOrder(ctx)
      }

      // connections ontop?
      if (graph.config.links_ontop) {
        this.drawConnections(ctx)
      }

      if (!LiteGraph.vueNodesMode || !this.overlayCtx) {
        this._drawConnectingLinks(ctx)
      } else {
        this._drawOverlayLinks()
      }

      this._drawLinkTooltip(ctx)

      this.onDrawForeground?.(ctx, this.visible_area)

      ctx.restore()
    }

    this.onDrawOverlay?.(ctx)

    if (area) ctx.restore()
  }

  private _getLinkCentreOnPos(e: CanvasPointerEvent): LinkSegment | undefined {
    if (this.linkMarkerShape === LinkMarkerShape.None) {
      return undefined
    }

    for (const linkSegment of this.renderedPaths) {
      const centre = linkSegment._pos
      if (!centre) continue

      if (
        isInRectangle(e.canvasX, e.canvasY, centre[0] - 4, centre[1] - 4, 8, 8)
      ) {
        return linkSegment
      }
    }
  }

  private _drawConnectingLinks(ctx: CanvasRenderingContext2D): void {
    const { linkConnector } = this
    if (!linkConnector.isConnecting) return

    const { renderLinks } = linkConnector
    const highlightPos = this._getHighlightPosition()
    ctx.lineWidth = this.connections_width

    for (const renderLink of renderLinks) {
      const {
        fromSlot,
        fromPos: pos,
        fromDirection,
        dragDirection
      } = renderLink
      const connShape = fromSlot.shape
      const connType = fromSlot.type

      const colour = resolveConnectingLinkColor(connType)

      if (this.linkRenderer) {
        this.linkRenderer.renderDraggingLink(
          ctx,
          pos,
          highlightPos,
          colour,
          fromDirection,
          dragDirection,
          {
            ...this.buildLinkRenderContext(),
            linkMarkerShape: LinkMarkerShape.None
          }
        )
      }
      if (renderLink instanceof MovingInputLink) this.setDirty(false, true)

      ctx.fillStyle = colour
      ctx.beginPath()
      if (connType === LiteGraph.EVENT || connShape === RenderShape.BOX) {
        ctx.rect(pos[0] - 6 + 0.5, pos[1] - 5 + 0.5, 14, 10)
        ctx.rect(highlightPos[0] - 6 + 0.5, highlightPos[1] - 5 + 0.5, 14, 10)
      } else if (connShape === RenderShape.ARROW) {
        ctx.moveTo(pos[0] + 8, pos[1] + 0.5)
        ctx.lineTo(pos[0] - 4, pos[1] + 6 + 0.5)
        ctx.lineTo(pos[0] - 4, pos[1] - 6 + 0.5)
        ctx.closePath()
      } else {
        ctx.arc(pos[0], pos[1], 4, 0, Math.PI * 2)
        ctx.arc(highlightPos[0], highlightPos[1], 4, 0, Math.PI * 2)
      }
      ctx.fill()
    }

    this._renderSnapHighlight(ctx, highlightPos)
  }

  private _drawLinkTooltip(ctx: CanvasRenderingContext2D): void {
    if (!this.isDragging && this.over_link_center && this.render_link_tooltip) {
      this.drawLinkTooltip(ctx, this.over_link_center)
    } else {
      this.onDrawLinkTooltip?.(ctx, null)
    }
  }

  private _drawOverlayLinks(): void {
    const octx = this.overlayCtx
    const overlayCanvas = this.overlayCanvas
    if (!octx || !overlayCanvas) return

    octx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height)

    if (!this.linkConnector.isConnecting) return

    octx.save()

    const scale = overlayCanvas.width / (overlayCanvas.clientWidth || 1)
    octx.setTransform(scale, 0, 0, scale, 0, 0)

    this.ds.toCanvasContext(octx)

    this._drawConnectingLinks(octx)

    octx.restore()
  }

  /** Get the target snap / highlight point in graph space */
  private _getHighlightPosition(): Readonly<Point> {
    return LiteGraph.snaps_for_comfy
      ? (this.linkConnector.state.snapLinksPos ??
          this._highlight_pos ??
          this.graph_mouse)
      : this.graph_mouse
  }

  /**
   * Renders indicators showing where a link will connect if released.
   * Partial border over target node and a highlight over the slot itself.
   * @param ctx Canvas 2D context
   */
  private _renderSnapHighlight(
    ctx: CanvasRenderingContext2D,
    highlightPos: Readonly<Point>
  ): void {
    const linkConnectorSnap = !!this.linkConnector.state.snapLinksPos
    if (!this._highlight_pos && !linkConnectorSnap) return

    ctx.fillStyle = '#ffcc00'
    ctx.beginPath()
    const shape = this._highlight_input?.shape

    if (shape === RenderShape.ARROW) {
      ctx.moveTo(highlightPos[0] + 8, highlightPos[1] + 0.5)
      ctx.lineTo(highlightPos[0] - 4, highlightPos[1] + 6 + 0.5)
      ctx.lineTo(highlightPos[0] - 4, highlightPos[1] - 6 + 0.5)
      ctx.closePath()
    } else {
      ctx.arc(highlightPos[0], highlightPos[1], 6, 0, Math.PI * 2)
    }
    ctx.fill()

    const { linkConnector } = this
    const { overReroute, overWidget } = linkConnector
    if (
      !LiteGraph.snap_highlights_node ||
      !linkConnector.isConnecting ||
      linkConnectorSnap
    )
      return

    // Reroute highlight
    overReroute?.drawHighlight(ctx, '#ffcc00aa')

    // Ensure we're mousing over a node and connecting a link
    const node = this.node_over
    if (!node) return

    const { strokeStyle, lineWidth } = ctx

    const area = node.boundingRect
    const gap = 3
    const radius = LiteGraph.ROUND_RADIUS + gap

    const x = area[0] - gap
    const y = area[1] - gap
    const width = area[2] + gap * 2
    const height = area[3] + gap * 2

    ctx.beginPath()
    ctx.roundRect(x, y, width, height, radius)

    // TODO: Currently works on LTR slots only.  Add support for other directions.
    const start = linkConnector.state.connectingTo === 'output' ? 0 : 1
    const inverter = start ? -1 : 1

    // Radial highlight centred on highlight pos
    const hx = highlightPos[0]
    const hy = highlightPos[1]
    const gRadius =
      width < height ? width : width * Math.max(height / width, 0.5)

    const gradient = ctx.createRadialGradient(hx, hy, 0, hx, hy, gRadius)
    gradient.addColorStop(1, '#00000000')
    gradient.addColorStop(0, '#ffcc00aa')

    // Linear gradient over half the node.
    const linearGradient = ctx.createLinearGradient(x, y, x + width, y)
    linearGradient.addColorStop(0.5, '#00000000')
    linearGradient.addColorStop(start + 0.67 * inverter, '#ddeeff33')
    linearGradient.addColorStop(start + inverter, '#ffcc0055')

    /**
     * Workaround for a canvas render issue.
     * In Chromium 129 (2024-10-15), rounded corners can be rendered with the wrong part of a gradient colour.
     * Occurs only at certain thicknesses / arc sizes.
     */
    ctx.setLineDash([radius, radius * 0.001])

    ctx.lineWidth = 1
    ctx.strokeStyle = linearGradient
    ctx.stroke()

    if (overWidget) {
      const { computedHeight } = overWidget

      ctx.beginPath()
      const {
        pos: [nodeX, nodeY]
      } = node
      const height = LiteGraph.NODE_WIDGET_HEIGHT
      if (
        overWidget.type.startsWith('custom') &&
        computedHeight != null &&
        computedHeight > height * 2
      ) {
        // Most likely DOM widget text box
        ctx.rect(
          nodeX + 9,
          nodeY + overWidget.y + 9,
          (overWidget.width ?? area[2]) - 18,
          computedHeight - 18
        )
      } else {
        // Regular widget, probably
        ctx.roundRect(
          nodeX + BaseWidget.margin,
          nodeY + overWidget.y,
          overWidget.width ?? area[2],
          height,
          height * 0.5
        )
      }
      ctx.stroke()
    }

    ctx.strokeStyle = gradient
    ctx.stroke()

    ctx.setLineDash([])
    ctx.lineWidth = lineWidth
    ctx.strokeStyle = strokeStyle
  }

  /**
   * draws some useful stats in the corner of the canvas
   */
  renderInfo(ctx: CanvasRenderingContext2D, x: number, y: number): void {
    x = x || 10
    y = y || this.canvas.offsetHeight - 80

    ctx.save()
    ctx.translate(x, y)

    ctx.font = `10px ${LiteGraph.DEFAULT_FONT}`
    ctx.fillStyle = '#888'
    ctx.textAlign = 'left'
    if (this.graph) {
      ctx.fillText(`T: ${this.graph.globaltime.toFixed(2)}s`, 5, 13 * 1)
      ctx.fillText(`I: ${this.graph.iteration}`, 5, 13 * 2)
      ctx.fillText(
        `N: ${this.graph._nodes.length} [${this.visible_nodes.length}]`,
        5,
        13 * 3
      )
      ctx.fillText(`V: ${this.graph._version}`, 5, 13 * 4)
      ctx.fillText(`FPS:${this.fps.toFixed(2)}`, 5, 13 * 5)
    } else {
      ctx.fillText('No graph selected', 5, 13 * 1)
    }
    ctx.restore()
  }

  /**
   * draws the back canvas (the one containing the background and the connections)
   */
  drawBackCanvas(): void {
    const canvas = this.bgcanvas
    if (
      canvas.width != this.canvas.width ||
      canvas.height != this.canvas.height
    ) {
      canvas.width = this.canvas.width
      canvas.height = this.canvas.height
    }

    if (!this.bgctx) {
      this.bgctx = this.bgcanvas.getContext('2d')
    }
    const ctx = this.bgctx
    if (!ctx) throw new TypeError('Background canvas context was null.')

    const viewport = this.viewport || [
      0,
      0,
      ctx.canvas.width,
      ctx.canvas.height
    ]

    // clear
    if (this.clear_background) {
      ctx.clearRect(viewport[0], viewport[1], viewport[2], viewport[3])
    }

    const bg_already_painted = this.onRenderBackground
      ? this.onRenderBackground(canvas, ctx)
      : false

    // reset in case of error
    if (!this.viewport) {
      const scale = window.devicePixelRatio
      ctx.restore()
      ctx.setTransform(scale, 0, 0, scale, 0, 0)
    }

    if (this.graph) {
      // apply transformations
      ctx.save()
      this.ds.toCanvasContext(ctx)

      // render BG
      if (
        this.ds.scale < 1.5 &&
        !bg_already_painted &&
        this.clear_background_color
      ) {
        ctx.fillStyle = this.clear_background_color
        ctx.fillRect(
          this.visible_area[0],
          this.visible_area[1],
          this.visible_area[2],
          this.visible_area[3]
        )
      }

      if (this.background_image && this.ds.scale > 0.5 && !bg_already_painted) {
        if (this.zoom_modify_alpha) {
          ctx.globalAlpha = (1.0 - 0.5 / this.ds.scale) * this.editor_alpha
        } else {
          ctx.globalAlpha = this.editor_alpha
        }
        ctx.imageSmoothingEnabled = false
        if (!this._bg_img || this._bg_img.name != this.background_image) {
          this._bg_img = new Image()
          this._bg_img.name = this.background_image
          this._bg_img.src = this.background_image
          this._bg_img.addEventListener('load', () => {
            this.draw(true, true)
          })
        }

        let pattern = this._pattern
        if (pattern == null && this._bg_img.width > 0) {
          pattern = ctx.createPattern(this._bg_img, 'repeat') ?? undefined
          this._pattern_img = this._bg_img
          this._pattern = pattern
        }

        // NOTE: This ridiculous kludge provides a significant performance increase when rendering many large (> canvas width) paths in HTML canvas.
        // I could find no documentation or explanation.  Requires that the BG image is set.
        if (pattern) {
          ctx.fillStyle = pattern
          ctx.fillRect(
            this.visible_area[0],
            this.visible_area[1],
            this.visible_area[2],
            this.visible_area[3]
          )
          ctx.fillStyle = 'transparent'
        }

        ctx.globalAlpha = 1.0
        ctx.imageSmoothingEnabled = true
      }
      if (this.bg_tint) {
        ctx.fillStyle = this.bg_tint
        ctx.fillRect(
          this.visible_area[0],
          this.visible_area[1],
          this.visible_area[2],
          this.visible_area[3]
        )
        ctx.fillStyle = 'transparent'
      }

      // groups
      if (this.graph._groups.length) {
        this.drawGroups(canvas, ctx)
      }

      this.onDrawBackground?.(ctx, this.visible_area)

      // DEBUG: show clipping area
      // ctx.fillStyle = "red";
      // ctx.fillRect( this.visible_area[0] + 10, this.visible_area[1] + 10, this.visible_area[2] - 20, this.visible_area[3] - 20);
      // bg
      if (this.render_canvas_border) {
        ctx.strokeStyle = '#235'
        ctx.strokeRect(0, 0, canvas.width, canvas.height)
      }

      if (this.render_connections_shadows) {
        ctx.shadowColor = '#000'
        ctx.shadowOffsetX = 0
        ctx.shadowOffsetY = 0
        ctx.shadowBlur = 6
      } else {
        ctx.shadowColor = 'rgba(0,0,0,0)'
      }

      // draw connections
      this.drawConnections(ctx)

      ctx.shadowColor = 'rgba(0,0,0,0)'

      // restore state
      ctx.restore()
    }

    this.dirty_bgcanvas = false
    // Forces repaint of the front canvas.
    this.dirty_canvas = true
  }

  /**
   * draws the given node inside the canvas
   */
  drawNode(node: LGraphNode, ctx: CanvasRenderingContext2D): void {
    this.current_node = node

    // When Vue nodes mode is enabled, LiteGraph should not draw node chrome or widgets.
    // We still need to keep slot metrics and layout in sync for hit-testing and links.
    // Interaction system changes coming later, chances are vue nodes mode will be mostly broken on land
    if (LiteGraph.vueNodesMode) {
      // Prepare concrete slots and compute layout measures without rendering visuals.
      node._setConcreteSlots()
      if (!node.collapsed) {
        node.arrange()
      }
      // Skip all node body/widget/title rendering. Vue overlay handles visuals.
      return
    }

    const color = node.renderingColor
    const bgcolor = node.renderingBgColor

    ctx.globalAlpha = this.getNodeModeAlpha(node)

    if (this.render_shadows && !this.low_quality) {
      ctx.shadowColor = LiteGraph.DEFAULT_SHADOW_COLOR
      ctx.shadowOffsetX = 2 * this.ds.scale
      ctx.shadowOffsetY = 2 * this.ds.scale
      ctx.shadowBlur = 3 * this.ds.scale
    } else {
      ctx.shadowColor = 'transparent'
    }

    // custom draw collapsed method (draw after shadows because they are affected)
    if (node.flags.collapsed && node.onDrawCollapsed?.(ctx, this) == true)
      return

    // clip if required (mask)
    const shape = node._shape || RenderShape.BOX
    const size = temp_vec2
    size[0] = node.renderingSize[0]
    size[1] = node.renderingSize[1]

    if (node.collapsed) {
      ctx.font = this.inner_text_font
    }

    if (node.clip_area) {
      // Start clipping
      ctx.save()
      ctx.beginPath()
      if (shape == RenderShape.BOX) {
        ctx.rect(0, 0, size[0], size[1])
      } else if (shape == RenderShape.ROUND) {
        ctx.roundRect(0, 0, size[0], size[1], [10])
      } else if (shape == RenderShape.CIRCLE) {
        ctx.arc(size[0] * 0.5, size[1] * 0.5, size[0] * 0.5, 0, Math.PI * 2)
      }
      ctx.clip()
    }

    // draw shape
    this.drawNodeShape(node, ctx, size, color, bgcolor, !!node.selected)

    // Render title buttons (if not collapsed)
    if (node.title_buttons && !node.flags.collapsed) {
      const title_height = LiteGraph.NODE_TITLE_HEIGHT
      let current_x = size[0] // Start flush with right edge

      for (let i = 0; i < node.title_buttons.length; i++) {
        const button = node.title_buttons[i]
        if (!button.visible) {
          continue
        }

        const button_width = button.getWidth(ctx)
        current_x -= button_width

        // Center button vertically in title bar
        const button_y = -title_height + (title_height - button.height) / 2

        button.draw(ctx, current_x, button_y)
        current_x -= 2
      }
    }

    if (!this.low_quality) {
      node.drawBadges(ctx)
    }

    ctx.shadowColor = 'transparent'

    // TODO: Legacy behaviour: onDrawForeground received ctx in this state
    ctx.strokeStyle = LiteGraph.NODE_BOX_OUTLINE_COLOR

    // Draw Foreground
    node.onDrawForeground?.(ctx, this, this.canvas)

    // connection slots
    ctx.font = this.inner_text_font

    // render inputs and outputs
    node._setConcreteSlots()
    if (!node.collapsed) {
      node.arrange()
      node.drawSlots(ctx, {
        fromSlot: this.linkConnector.renderLinks[0]?.fromSlot as
          | INodeOutputSlot
          | INodeInputSlot,
        colorContext: this.colourGetter,
        editorAlpha: this.editor_alpha,
        lowQuality: this.low_quality
      })

      ctx.textAlign = 'left'
      ctx.globalAlpha = 1

      this.drawNodeWidgets(node, null, ctx)
    } else if (this.render_collapsed_slots) {
      node.drawCollapsedSlots(ctx)
    }

    if (node.clip_area) {
      ctx.restore()
    }

    ctx.globalAlpha = 1.0
  }

  /**
   * Draws the link mouseover effect and tooltip.
   * @param ctx Canvas 2D context to draw on
   * @param link The link to render the mouseover effect for
   * @remarks
   * Called against {@link LGraphCanvas.over_link_center}.
   * @todo Split tooltip from hover, so it can be drawn / eased separately
   */
  drawLinkTooltip(ctx: CanvasRenderingContext2D, link: LinkSegment): void {
    const pos = link._pos
    ctx.fillStyle = 'black'
    ctx.beginPath()
    if (this.linkMarkerShape === LinkMarkerShape.Arrow) {
      const transform = ctx.getTransform()
      ctx.translate(pos[0], pos[1])
      // Assertion: Number.isFinite guarantees this is a number.
      if (Number.isFinite(link._centreAngle))
        ctx.rotate(link._centreAngle as number)
      ctx.moveTo(-2, -3)
      ctx.lineTo(+4, 0)
      ctx.lineTo(-2, +3)
      ctx.setTransform(transform)
    } else if (
      this.linkMarkerShape == null ||
      this.linkMarkerShape === LinkMarkerShape.Circle
    ) {
      ctx.arc(pos[0], pos[1], 3, 0, Math.PI * 2)
    }
    ctx.fill()

    // @ts-expect-error TODO: Better value typing
    const { data } = link
    if (data == null) return

    // @ts-expect-error TODO: Better value typing
    if (this.onDrawLinkTooltip?.(ctx, link, this) == true) return

    let text: string | null = null

    if (typeof data === 'number') text = data.toFixed(2)
    else if (typeof data === 'string') text = `"${data}"`
    else if (typeof data === 'boolean') text = String(data)
    else if (data.toToolTip) text = data.toToolTip()
    else text = `[${data.constructor.name}]`

    if (text == null) return

    // Hard-coded tooltip limit
    text = text.substring(0, 30)

    ctx.font = '14px Courier New'
    const info = ctx.measureText(text)
    const w = info.width + 20
    const h = 24
    ctx.shadowColor = 'black'
    ctx.shadowOffsetX = 2
    ctx.shadowOffsetY = 2
    ctx.shadowBlur = 3
    ctx.fillStyle = '#454'
    ctx.beginPath()
    ctx.roundRect(pos[0] - w * 0.5, pos[1] - 15 - h, w, h, [3])
    ctx.moveTo(pos[0] - 10, pos[1] - 15)
    ctx.lineTo(pos[0] + 10, pos[1] - 15)
    ctx.lineTo(pos[0], pos[1] - 5)
    ctx.fill()
    ctx.shadowColor = 'transparent'
    ctx.textAlign = 'center'
    ctx.fillStyle = '#CEC'
    ctx.fillText(text, pos[0], pos[1] - 15 - h * 0.3)
  }

  /**
   * Draws the shape of the given node on the canvas
   * @param node The node to draw
   * @param ctx 2D canvas rendering context used to draw
   * @param size Size of the background to draw, in graph units.  Differs from node size if collapsed, etc.
   * @param fgcolor Foreground colour - used for text
   * @param bgcolor Background colour of the node
   * @param _selected Whether to render the node as selected.  Likely to be removed in future, as current usage is simply the selected property of the node.
   */
  drawNodeShape(
    node: LGraphNode,
    ctx: CanvasRenderingContext2D,
    size: Size,
    fgcolor: CanvasColour,
    bgcolor: CanvasColour,
    _selected: boolean
  ): void {
    // Rendering options
    ctx.strokeStyle = fgcolor
    ctx.fillStyle = bgcolor

    const title_height = LiteGraph.NODE_TITLE_HEIGHT
    const { low_quality } = this

    const { collapsed } = node.flags
    const shape = node.renderingShape
    const { title_mode } = node

    const render_title =
      title_mode == TitleMode.TRANSPARENT_TITLE ||
      title_mode == TitleMode.NO_TITLE
        ? false
        : true

    // Normalised node dimensions
    const area = tmp_area
    area.set(node.boundingRect)
    area[0] -= node.pos[0]
    area[1] -= node.pos[1]

    const old_alpha = ctx.globalAlpha

    // Draw node background (shape)
    ctx.beginPath()
    if (shape == RenderShape.BOX || low_quality) {
      ctx.rect(area[0], area[1], area[2], area[3])
    } else if (shape == RenderShape.ROUND || shape == RenderShape.CARD) {
      ctx.roundRect(
        area[0],
        area[1],
        area[2],
        area[3],
        shape == RenderShape.CARD
          ? [LiteGraph.ROUND_RADIUS, LiteGraph.ROUND_RADIUS, 0, 0]
          : [LiteGraph.ROUND_RADIUS]
      )
    } else if (shape == RenderShape.CIRCLE) {
      ctx.arc(size[0] * 0.5, size[1] * 0.5, size[0] * 0.5, 0, Math.PI * 2)
    }
    ctx.fill()

    // Separator - title bar <-> body
    if (!collapsed && render_title) {
      ctx.shadowColor = 'transparent'
      ctx.fillStyle = 'rgba(0,0,0,0.2)'
      ctx.fillRect(0, -1, area[2], 2)
    }
    ctx.shadowColor = 'transparent'

    node.onDrawBackground?.(ctx)

    // Title bar background (remember, it is rendered ABOVE the node)
    if (render_title || title_mode == TitleMode.TRANSPARENT_TITLE) {
      node.drawTitleBarBackground(ctx, {
        scale: this.ds.scale,
        low_quality
      })

      // title box
      node.drawTitleBox(ctx, {
        scale: this.ds.scale,
        low_quality,
        box_size: 10
      })

      ctx.globalAlpha = old_alpha

      // title text
      node.drawTitleText(ctx, {
        scale: this.ds.scale,
        default_title_color: this.node_title_color,
        low_quality
      })

      // custom title render
      node.onDrawTitle?.(ctx)
    }

    // Draw stroke styles
    for (const getStyle of Object.values(node.strokeStyles)) {
      const strokeStyle = getStyle.call(node)
      if (strokeStyle) {
        strokeShape(ctx, area, {
          shape,
          title_height,
          title_mode,
          collapsed,
          ...strokeStyle
        })
      }
    }

    node.drawProgressBar(ctx)

    // these counter helps in conditioning drawing based on if the node has been executed or an action occurred
    if (node.execute_triggered != null && node.execute_triggered > 0)
      node.execute_triggered--
    if (node.action_triggered != null && node.action_triggered > 0)
      node.action_triggered--
  }

  /**
   * Draws a snap guide for a {@link Positionable} item.
   *
   * Initial design was a simple white rectangle representing the location the
   * item would land if dropped.
   * @param ctx The 2D canvas context to draw on
   * @param item The item to draw a snap guide for
   * @param shape The shape of the snap guide to draw
   * @todo Update to align snapping with boundingRect
   * @todo Shapes
   */
  drawSnapGuide(
    ctx: CanvasRenderingContext2D,
    item: Positionable,
    shape = RenderShape.ROUND
  ) {
    const snapGuide = temp
    snapGuide.set(item.boundingRect)

    // Not all items have pos equal to top-left of bounds
    const { pos } = item
    const offsetX = pos[0] - snapGuide[0]
    const offsetY = pos[1] - snapGuide[1]

    // Normalise boundingRect to pos to snap
    snapGuide[0] += offsetX
    snapGuide[1] += offsetY
    if (this._snapToGrid) snapPoint(snapGuide, this._snapToGrid)
    snapGuide[0] -= offsetX
    snapGuide[1] -= offsetY

    const { globalAlpha } = ctx
    ctx.globalAlpha = 1
    ctx.beginPath()
    const [x, y, w, h] = snapGuide
    if (shape === RenderShape.CIRCLE) {
      const midX = x + w * 0.5
      const midY = y + h * 0.5
      const radius = Math.min(w * 0.5, h * 0.5)
      ctx.arc(midX, midY, radius, 0, Math.PI * 2)
    } else {
      ctx.rect(x, y, w, h)
    }

    ctx.lineWidth = 0.5
    ctx.strokeStyle = '#FFFFFF66'
    ctx.fillStyle = '#FFFFFF22'
    ctx.fill()
    ctx.stroke()
    ctx.globalAlpha = globalAlpha
  }

  drawConnections(ctx: CanvasRenderingContext2D): void {
    this.renderedPaths.clear()
    if (this.links_render_mode === LinkRenderType.HIDDEN_LINK) return

    // Skip link rendering while waiting for slot positions to sync after reconfigure
    if (LiteGraph.vueNodesMode && layoutStore.pendingSlotSync) {
      this._visibleReroutes.clear()
      return
    }

    const { graph, subgraph } = this
    if (!graph) throw new NullGraphError()

    const visibleReroutes: Reroute[] = []

    const now = LiteGraph.getTime()
    const { visible_area } = this
    margin_area[0] = visible_area[0] - 20
    margin_area[1] = visible_area[1] - 20
    margin_area[2] = visible_area[2] + 40
    margin_area[3] = visible_area[3] + 40

    // draw connections
    ctx.lineWidth = this.connections_width

    ctx.fillStyle = '#AAA'
    ctx.strokeStyle = '#AAA'
    ctx.globalAlpha = this.editor_alpha
    // for every node
    const nodes = graph._nodes
    for (const node of nodes) {
      // for every input (we render just inputs because it is easier as every slot can only have one input)
      const { inputs } = node
      if (!inputs?.length) continue

      for (const [i, input] of inputs.entries()) {
        if (!input || input.link == null) continue

        const link_id = input.link
        const link = graph._links.get(link_id)
        if (!link) continue

        const endPos: Point = LiteGraph.vueNodesMode // TODO: still use LG get pos if vue nodes is off until stable
          ? getSlotPosition(node, i, true)
          : node.getInputPos(i)

        // find link info
        const start_node = graph.getNodeById(link.origin_id)
        if (start_node == null) continue

        const outputId = link.origin_slot
        const startPos: Point =
          outputId === -1
            ? [start_node.pos[0] + 10, start_node.pos[1] + 10]
            : LiteGraph.vueNodesMode // TODO: still use LG get pos if vue nodes is off until stable
              ? getSlotPosition(start_node, outputId, false)
              : start_node.getOutputPos(outputId)

        const output = start_node.outputs[outputId]
        if (!output) continue

        this._renderAllLinkSegments(
          ctx,
          link,
          startPos,
          endPos,
          visibleReroutes,
          now,
          output.dir,
          input.dir
        )
      }
    }

    if (subgraph) {
      for (const output of subgraph.inputNode.slots) {
        if (!output.linkIds.length) continue

        // find link info
        for (const linkId of output.linkIds) {
          const resolved = LLink.resolve(linkId, graph)
          if (!resolved) continue

          const { link, inputNode, input } = resolved
          if (!inputNode || !input) continue

          const endPos = LiteGraph.vueNodesMode
            ? getSlotPosition(inputNode, link.target_slot, true)
            : inputNode.getInputPos(link.target_slot)

          this._renderAllLinkSegments(
            ctx,
            link,
            output.pos,
            endPos,
            visibleReroutes,
            now,
            input.dir,
            input.dir
          )
        }
      }

      for (const input of subgraph.outputNode.slots) {
        if (!input.linkIds.length) continue

        // find link info
        const resolved = LLink.resolve(input.linkIds[0], graph)
        if (!resolved) continue

        const { link, outputNode, output } = resolved
        if (!outputNode || !output) continue

        const startPos = LiteGraph.vueNodesMode
          ? getSlotPosition(outputNode, link.origin_slot, false)
          : outputNode.getOutputPos(link.origin_slot)

        this._renderAllLinkSegments(
          ctx,
          link,
          startPos,
          input.pos,
          visibleReroutes,
          now,
          output.dir,
          input.dir
        )
      }
    }

    if (graph.floatingLinks.size > 0) {
      this._renderFloatingLinks(ctx, graph, visibleReroutes, now)
    }

    const rerouteSet = this._visibleReroutes
    rerouteSet.clear()

    // Render reroutes, ordered by number of non-floating links
    visibleReroutes.sort((a, b) => a.linkIds.size - b.linkIds.size)
    for (const reroute of visibleReroutes) {
      rerouteSet.add(reroute)

      if (
        this._snapToGrid &&
        this.isDragging &&
        this.selectedItems.has(reroute)
      ) {
        this.drawSnapGuide(ctx, reroute, RenderShape.CIRCLE)
      }
      reroute.draw(ctx, this._pattern)

      // Never draw slots when the pointer is down
      if (!this.pointer.isDown) reroute.drawSlots(ctx)
    }

    const highlightPos = this._getHighlightPosition()
    this.linkConnector.renderLinks
      .filter((rl) => rl instanceof MovingInputLink)
      .forEach((rl) => rl.drawConnectionCircle(ctx, highlightPos))

    ctx.globalAlpha = 1
  }

  private getNodeModeAlpha(node: LGraphNode) {
    if (node.flags.ghost) return 0.3
    return node.mode === LGraphEventMode.BYPASS
      ? 0.2
      : node.mode === LGraphEventMode.NEVER
        ? 0.4
        : this.editor_alpha
  }

  private _renderFloatingLinks(
    ctx: CanvasRenderingContext2D,
    graph: LGraph,
    visibleReroutes: Reroute[],
    now: number
  ) {
    // Render floating links with 3/4 current alpha
    const { globalAlpha } = ctx
    ctx.globalAlpha = globalAlpha * 0.33

    // Floating reroutes
    for (const link of graph.floatingLinks.values()) {
      const reroutes = LLink.getReroutes(graph, link)
      const firstReroute = reroutes[0]
      const reroute = reroutes.at(-1)
      if (!firstReroute || !reroute?.floating) continue

      // Input not connected
      if (reroute.floating.slotType === 'input') {
        const node = graph.getNodeById(link.target_id)
        if (!node) continue

        const startPos = firstReroute.pos
        const endPos: Point = LiteGraph.vueNodesMode
          ? getSlotPosition(node, link.target_slot, true)
          : node.getInputPos(link.target_slot)
        const endDirection = node.inputs[link.target_slot]?.dir

        firstReroute._dragging = true
        this._renderAllLinkSegments(
          ctx,
          link,
          startPos,
          endPos,
          visibleReroutes,
          now,
          LinkDirection.CENTER,
          endDirection,
          true
        )
      } else {
        const node = graph.getNodeById(link.origin_id)
        if (!node) continue

        const startPos: Point = LiteGraph.vueNodesMode
          ? getSlotPosition(node, link.origin_slot, false)
          : node.getOutputPos(link.origin_slot)
        const endPos = reroute.pos
        const startDirection = node.outputs[link.origin_slot]?.dir

        link._dragging = true
        this._renderAllLinkSegments(
          ctx,
          link,
          startPos,
          endPos,
          visibleReroutes,
          now,
          startDirection,
          LinkDirection.CENTER,
          true
        )
      }
    }
    ctx.globalAlpha = globalAlpha
  }

  private _renderAllLinkSegments(
    ctx: CanvasRenderingContext2D,
    link: LLink,
    startPos: Point,
    endPos: Point,
    visibleReroutes: Reroute[],
    now: number,
    startDirection?: LinkDirection,
    endDirection?: LinkDirection,
    disabled: boolean = false
  ) {
    const { graph, renderedPaths } = this
    if (!graph) return

    // Get all points this link passes through
    const reroutes = LLink.getReroutes(graph, link)
    const points: [Point, ...Point[], Point] = [
      startPos,
      ...reroutes.map((x) => x.pos),
      endPos
    ]

    // Bounding box of all points (bezier overshoot on long links will be cut)
    const pointsX = points.map((x) => x[0])
    const pointsY = points.map((x) => x[1])
    link_bounding[0] = Math.min(...pointsX)
    link_bounding[1] = Math.min(...pointsY)
    link_bounding[2] = Math.max(...pointsX) - link_bounding[0]
    link_bounding[3] = Math.max(...pointsY) - link_bounding[1]

    // skip links outside of the visible area of the canvas
    if (!overlapBounding(link_bounding, margin_area)) return

    const start_dir = startDirection || LinkDirection.RIGHT
    const end_dir = endDirection || LinkDirection.LEFT

    // Has reroutes
    if (reroutes.length) {
      let startControl: Point | undefined

      const l = reroutes.length
      for (let j = 0; j < l; j++) {
        const reroute = reroutes[j]

        // Only render once
        if (!renderedPaths.has(reroute)) {
          renderedPaths.add(reroute)
          visibleReroutes.push(reroute)
          reroute._colour =
            link.color ||
            LGraphCanvas.link_type_colors[link.type] ||
            this.default_link_color

          const prevReroute = graph.getReroute(reroute.parentId)
          const rerouteStartPos = prevReroute?.pos ?? startPos
          reroute.calculateAngle(this.last_draw_time, graph, rerouteStartPos)

          // Skip the first segment if it is being dragged
          if (!reroute._dragging) {
            this.renderLink(
              ctx,
              rerouteStartPos,
              reroute.pos,
              link,
              false,
              0,
              null,
              startControl === undefined ? start_dir : LinkDirection.CENTER,
              LinkDirection.CENTER,
              {
                startControl,
                endControl: reroute.controlPoint,
                reroute,
                disabled
              }
            )
          }
        }

        if (!startControl && reroutes.at(-1)?.floating?.slotType === 'input') {
          // Floating link connected to an input
          startControl = [0, 0]
        } else {
          // Calculate start control for the next iter control point
          const nextPos = reroutes[j + 1]?.pos ?? endPos
          const dist = Math.min(
            Reroute.maxSplineOffset,
            distance(reroute.pos, nextPos) * 0.25
          )
          startControl = [dist * reroute.cos, dist * reroute.sin]
        }
      }

      // Skip the last segment if it is being dragged
      if (link._dragging) return

      // Use runtime fallback; TypeScript cannot evaluate this correctly.
      const segmentStartPos = points.at(-2) ?? startPos

      // Render final link segment
      this.renderLink(
        ctx,
        segmentStartPos,
        endPos,
        link,
        false,
        0,
        null,
        LinkDirection.CENTER,
        end_dir,
        { startControl, disabled }
      )
      // Skip normal render when link is being dragged
    } else if (!link._dragging) {
      this.renderLink(
        ctx,
        startPos,
        endPos,
        link,
        false,
        0,
        null,
        start_dir,
        end_dir
      )
    }
    renderedPaths.add(link)

    // event triggered rendered on top
    if (link?._last_time && now - link._last_time < 1000) {
      const f = 2.0 - (now - link._last_time) * 0.002
      const tmp = ctx.globalAlpha
      ctx.globalAlpha = tmp * f
      this.renderLink(
        ctx,
        startPos,
        endPos,
        link,
        true,
        f,
        'white',
        start_dir,
        end_dir
      )
      ctx.globalAlpha = tmp
    }
  }

  /**
   * Build LinkRenderContext from canvas properties
   * Helper method for using LitegraphLinkAdapter
   */
  private buildLinkRenderContext(): LinkRenderContext {
    return {
      // Canvas settings
      renderMode: this.links_render_mode,
      connectionWidth: this.connections_width,
      renderBorder: this.render_connections_border,
      lowQuality: this.low_quality,
      highQualityRender: this.highquality_render,
      scale: this.ds.scale,
      linkMarkerShape: this.linkMarkerShape,
      renderConnectionArrows: this.render_connection_arrows,

      // State
      highlightedLinks: new Set(Object.keys(this.highlighted_links)),

      // Colors
      defaultLinkColor: this.default_link_color,
      linkTypeColors: LGraphCanvas.link_type_colors,

      // Pattern for disabled links
      disabledPattern: this._pattern
    }
  }

  /**
   * draws a link between two points
   * @param ctx Canvas 2D rendering context
   * @param a start pos
   * @param b end pos
   * @param link the link object with all the link info
   * @param skip_border ignore the shadow of the link
   * @param flow show flow animation (for events)
   * @param color the color for the link
   * @param start_dir the direction enum
   * @param end_dir the direction enum
   */
  renderLink(
    ctx: CanvasRenderingContext2D,
    a: Readonly<Point>,
    b: Readonly<Point>,
    link: LLink | null,
    skip_border: boolean,
    flow: number | null,
    color: CanvasColour | null,
    start_dir: LinkDirection,
    end_dir: LinkDirection,
    {
      startControl,
      endControl,
      reroute,
      num_sublines = 1,
      disabled = false
    }: {
      /** When defined, render data will be saved to this reroute instead of the {@link link}. */
      reroute?: Reroute
      /** Offset of the bezier curve control point from {@link a point a} (output side) */
      startControl?: Readonly<Point>
      /** Offset of the bezier curve control point from {@link b point b} (input side) */
      endControl?: Readonly<Point>
      /** Number of sublines (useful to represent vec3 or rgb) @todo If implemented, refactor calculations out of the loop */
      num_sublines?: number
      /** Whether this is a floating link segment */
      disabled?: boolean
    } = {}
  ): void {
    if (this.linkRenderer) {
      const context = this.buildLinkRenderContext()
      this.linkRenderer.renderLinkDirect(
        ctx,
        a,
        b,
        link,
        skip_border,
        flow,
        color,
        start_dir,
        end_dir,
        context,
        {
          reroute,
          startControl,
          endControl,
          num_sublines,
          disabled
        }
      )
    }
  }

  drawExecutionOrder(ctx: CanvasRenderingContext2D): void {
    ctx.shadowColor = 'transparent'
    ctx.globalAlpha = 0.25

    ctx.textAlign = 'center'
    ctx.strokeStyle = 'white'
    ctx.globalAlpha = 0.75

    const { visible_nodes } = this
    for (const node of visible_nodes) {
      ctx.fillStyle = 'black'
      ctx.fillRect(
        node.pos[0] - LiteGraph.NODE_TITLE_HEIGHT,
        node.pos[1] - LiteGraph.NODE_TITLE_HEIGHT,
        LiteGraph.NODE_TITLE_HEIGHT,
        LiteGraph.NODE_TITLE_HEIGHT
      )
      if (node.order == 0) {
        ctx.strokeRect(
          node.pos[0] - LiteGraph.NODE_TITLE_HEIGHT + 0.5,
          node.pos[1] - LiteGraph.NODE_TITLE_HEIGHT + 0.5,
          LiteGraph.NODE_TITLE_HEIGHT,
          LiteGraph.NODE_TITLE_HEIGHT
        )
      }
      ctx.fillStyle = '#FFF'
      ctx.fillText(
        toString(node.order),
        node.pos[0] + LiteGraph.NODE_TITLE_HEIGHT * -0.5,
        node.pos[1] - 6
      )
    }
    ctx.globalAlpha = 1
  }

  /**
   * draws the widgets stored inside a node
   * @deprecated Use {@link LGraphNode.drawWidgets} instead.
   * @remarks Currently there are extensions hijacking this function, so we cannot remove it.
   */
  drawNodeWidgets(
    node: LGraphNode,
    _posY: null,
    ctx: CanvasRenderingContext2D
  ): void {
    node.drawWidgets(ctx, {
      lowQuality: this.low_quality,
      editorAlpha: this.getNodeModeAlpha(node)
    })
  }

  /**
   * draws every group area in the background
   */
  drawGroups(_canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D): void {
    if (!this.graph) return

    const groups = this.graph._groups

    ctx.save()
    ctx.globalAlpha = 0.5 * this.editor_alpha
    const drawSnapGuides =
      this._snapToGrid &&
      (this.isDragging || layoutStore.isDraggingVueNodes.value)

    for (const group of groups) {
      // out of the visible area
      if (!overlapBounding(this.visible_area, group._bounding)) {
        continue
      }

      // Draw snap shadow
      if (drawSnapGuides && this.selectedItems.has(group))
        this.drawSnapGuide(ctx, group)

      group.draw(this, ctx)
    }

    ctx.restore()
  }

  /**
   * resizes the canvas to a given size, if no size is passed, then it tries to fill the parentNode
   * @todo Remove or rewrite
   */
  resize(width?: number, height?: number): void {
    if (!width && !height) {
      const parent = this.canvas.parentElement
      if (!parent)
        throw new TypeError(
          'Attempted to resize canvas, but parent element was null.'
        )
      width = parent.offsetWidth
      height = parent.offsetHeight
    }

    if (this.canvas.width == width && this.canvas.height == height) return

    this.canvas.width = width ?? 0
    this.canvas.height = height ?? 0
    this.bgcanvas.width = this.canvas.width
    this.bgcanvas.height = this.canvas.height
    this.setDirty(true, true)
  }

  onNodeSelectionChange(): void {}

  /**
   * Determines the furthest nodes in each direction for the currently selected nodes
   */
  boundaryNodesForSelection(): NullableProperties<IBoundaryNodes> {
    return LGraphCanvas.getBoundaryNodes(this.selected_nodes)
  }

  showLinkMenu(segment: LinkSegment, e: CanvasPointerEvent): boolean {
    const { graph } = this
    if (!graph) throw new NullGraphError()

    const title =
      'data' in segment && segment.data != null
        ? segment.data.constructor.name
        : undefined

    const { origin_id, origin_slot } = segment
    if (origin_id == null || origin_slot == null) {
      new LiteGraph.ContextMenu<string>(['Link has no origin'], {
        event: e,
        title
      })
      return false
    }

    const node_left = graph.getNodeById(origin_id)
    const fromType = node_left?.outputs?.[origin_slot]?.type

    const options = ['Add Node', 'Add Reroute', null, 'Delete', null]

    const menu = new LiteGraph.ContextMenu<string>(options, {
      event: e,
      title,
      callback: inner_clicked.bind(this)
    })

    return false

    function inner_clicked(
      this: LGraphCanvas,
      v: string,
      _options: unknown,
      e: MouseEvent
    ) {
      if (!graph) throw new NullGraphError()

      switch (v) {
        case 'Add Node':
          LGraphCanvas.onMenuAdd(null, null, e, menu, (node) => {
            if (
              !node?.inputs?.length ||
              !node?.outputs?.length ||
              origin_slot == null
            )
              return

            // leave the connection type checking inside connectByType
            const options = { afterRerouteId: segment.parentId }
            if (
              node_left?.connectByType(
                origin_slot,
                node,
                fromType ?? '*',
                options
              )
            ) {
              node.setPos(node.pos[0] - node.size[0] * 0.5, node.pos[1])
            }
          })
          break

        case 'Add Reroute': {
          try {
            this.emitBeforeChange()
            this.adjustMouseEvent(e)
            graph.createReroute(segment._pos, segment)
            this.setDirty(false, true)
          } catch (error) {
            console.error(error)
          } finally {
            this.emitAfterChange()
          }
          break
        }

        case 'Delete': {
          // segment can be a Reroute object, in which case segment.id is the reroute id
          const linkId =
            segment instanceof Reroute
              ? segment.linkIds.values().next().value
              : segment.id
          if (linkId !== undefined) {
            graph.removeLink(linkId)
            // Clean up layout store
            layoutStore.deleteLinkLayout(linkId)
          }
          break
        }
        default:
      }
    }
  }

  createDefaultNodeForSlot(optPass: ICreateDefaultNodeOptions): boolean {
    type DefaultOptions = ICreateDefaultNodeOptions & {
      posAdd: Point
      posSizeFix: Point
    }

    const opts = Object.assign<DefaultOptions, ICreateDefaultNodeOptions>(
      {
        nodeFrom: null,
        slotFrom: null,
        nodeTo: null,
        slotTo: null,
        position: [0, 0],
        nodeType: undefined,
        posAdd: [0, 0],
        posSizeFix: [0, 0]
      },
      optPass
    )
    const { afterRerouteId } = opts

    const isFrom = opts.nodeFrom && opts.slotFrom !== null
    const isTo = !isFrom && opts.nodeTo && opts.slotTo !== null

    if (!isFrom && !isTo) {
      console.warn(
        `No data passed to createDefaultNodeForSlot`,
        opts.nodeFrom,
        opts.slotFrom,
        opts.nodeTo,
        opts.slotTo
      )
      return false
    }
    if (!opts.nodeType) {
      console.warn('No type to createDefaultNodeForSlot')
      return false
    }

    const nodeX = isFrom ? opts.nodeFrom : opts.nodeTo
    if (!nodeX)
      throw new TypeError('nodeX was null when creating default node for slot.')

    let slotX = isFrom ? opts.slotFrom : opts.slotTo

    let iSlotConn: number | false = false
    if (nodeX instanceof SubgraphIONodeBase) {
      if (typeof slotX !== 'object' || !slotX) {
        console.warn('Cant get slot information', slotX)
        return false
      }
      const { name } = slotX
      iSlotConn = nodeX.slots.findIndex((s) => s.name === name)
      slotX = nodeX.slots[iSlotConn]
      if (!slotX) {
        console.warn('Cant get slot information', slotX)
        return false
      }
    } else {
      switch (typeof slotX) {
        case 'string':
          iSlotConn = isFrom
            ? nodeX.findOutputSlot(slotX, false)
            : nodeX.findInputSlot(slotX, false)
          slotX = isFrom ? nodeX.outputs[slotX] : nodeX.inputs[slotX]
          break
        case 'object':
          if (slotX === null) {
            console.warn('Cant get slot information', slotX)
            return false
          }

          // ok slotX
          iSlotConn = isFrom
            ? nodeX.findOutputSlot(slotX.name)
            : nodeX.findInputSlot(slotX.name)
          break
        case 'number':
          iSlotConn = slotX
          slotX = isFrom ? nodeX.outputs[slotX] : nodeX.inputs[slotX]
          break
        case 'undefined':
        default:
          console.warn('Cant get slot information', slotX)
          return false
      }
    }

    // check for defaults nodes for this slottype
    const fromSlotType = slotX.type == LiteGraph.EVENT ? '_event_' : slotX.type
    const slotTypesDefault = isFrom
      ? LiteGraph.slot_types_default_out
      : LiteGraph.slot_types_default_in
    if (slotTypesDefault?.[fromSlotType]) {
      let nodeNewType: string | Record<string, unknown> | false = false
      if (typeof slotTypesDefault[fromSlotType] == 'object') {
        for (const typeX in slotTypesDefault[fromSlotType]) {
          if (
            opts.nodeType == slotTypesDefault[fromSlotType][typeX] ||
            opts.nodeType == 'AUTO'
          ) {
            nodeNewType = slotTypesDefault[fromSlotType][typeX]
            break
          }
        }
      } else if (
        opts.nodeType == slotTypesDefault[fromSlotType] ||
        opts.nodeType == 'AUTO'
      ) {
        nodeNewType = slotTypesDefault[fromSlotType]
      }
      if (nodeNewType) {
        let nodeNewOpts: SlotTypeDefaultNodeOpts | undefined
        let nodeTypeStr: string
        if (typeof nodeNewType == 'object') {
          nodeNewOpts = nodeNewType as SlotTypeDefaultNodeOpts
          nodeTypeStr = nodeNewOpts.node ?? ''
        } else {
          nodeTypeStr = nodeNewType
        }

        // that.graph.beforeChange();
        const xSizeFix = opts.posSizeFix[0] * LiteGraph.NODE_WIDTH
        const ySizeFix = opts.posSizeFix[1] * LiteGraph.NODE_SLOT_HEIGHT
        const nodeX = opts.position[0] + opts.posAdd[0] + xSizeFix
        const nodeY = opts.position[1] + opts.posAdd[1] + ySizeFix
        const pos = [nodeX, nodeY]
        const newNode = LiteGraph.createNode(nodeTypeStr, nodeNewOpts?.title, {
          pos
        })
        if (newNode) {
          // if is object pass options
          if (nodeNewOpts) {
            if (nodeNewOpts.properties) {
              for (const i in nodeNewOpts.properties) {
                newNode.addProperty(i, nodeNewOpts.properties[i])
              }
            }
            if (nodeNewOpts.inputs) {
              newNode.inputs = []
              for (const input of nodeNewOpts.inputs) {
                newNode.addInput(input[0], input[1])
              }
            }
            if (nodeNewOpts.outputs) {
              newNode.outputs = []
              for (const output of nodeNewOpts.outputs) {
                newNode.addOutput(output[0], output[1])
              }
            }
            if (nodeNewOpts.json) {
              newNode.configure(nodeNewOpts.json)
            }
          }

          // add the node
          if (!this.graph) throw new NullGraphError()

          this.graph.add(newNode)

          // Interim API - allow the link connection to be canceled.
          // TODO: https://github.com/Comfy-Org/litegraph.js/issues/946
          const detail = { node: newNode, opts }
          const mayConnectLinks = this.canvas.dispatchEvent(
            new CustomEvent('connect-new-default-node', {
              detail,
              cancelable: true
            })
          )
          if (!mayConnectLinks) return true

          // connect the two!
          if (isFrom) {
            if (!opts.nodeFrom)
              throw new TypeError(
                'createDefaultNodeForSlot - nodeFrom was null'
              )
            opts.nodeFrom.connectByType(iSlotConn, newNode, fromSlotType, {
              afterRerouteId
            })
          } else {
            if (!opts.nodeTo)
              throw new TypeError('createDefaultNodeForSlot - nodeTo was null')
            opts.nodeTo.connectByTypeOutput(iSlotConn, newNode, fromSlotType, {
              afterRerouteId
            })
          }

          // if connecting in between
          if (isFrom && isTo) {
            // TODO
          }

          return true
        }
        console.error(`failed creating ${nodeNewType}`)
      }
    }
    return false
  }

  showConnectionMenu(
    optPass: Partial<ICreateNodeOptions & { e: MouseEvent }>
  ): ContextMenu<string> | undefined {
    const opts = Object.assign<
      ICreateNodeOptions & HasShowSearchCallback,
      ICreateNodeOptions
    >(
      {
        nodeFrom: null,
        slotFrom: null,
        nodeTo: null,
        slotTo: null,
        e: undefined,
        allow_searchbox: this.allow_searchbox,
        showSearchBox: this.showSearchBox
      },
      optPass || {}
    )
    const dirty = () => this._dirty()

    const that = this
    const { graph } = this
    const { afterRerouteId } = opts

    const isFrom = opts.nodeFrom && opts.slotFrom
    const isTo = !isFrom && opts.nodeTo && opts.slotTo

    if (!isFrom && !isTo) {
      console.warn('No data passed to showConnectionMenu')
      return
    }

    const nodeX = isFrom ? opts.nodeFrom : opts.nodeTo
    if (!nodeX)
      throw new TypeError('nodeX was null when creating default node for slot.')
    let slotX = isFrom ? opts.slotFrom : opts.slotTo

    let iSlotConn: number
    if (nodeX instanceof SubgraphIONodeBase) {
      if (typeof slotX !== 'object' || !slotX) {
        console.warn('Cant get slot information', slotX)
        return
      }
      const { name } = slotX
      iSlotConn = nodeX.slots.findIndex((s) => s.name === name)
      // If it's not found in the main slots, it might be the empty slot from a Subgraph node.
      // In that case, the original `slotX` object is the correct one, so don't overwrite it.
      if (iSlotConn !== -1) {
        slotX = nodeX.slots[iSlotConn]
      }
      if (!slotX) {
        console.warn('Cant get slot information', slotX)
        return
      }
    } else {
      switch (typeof slotX) {
        case 'string':
          iSlotConn = isFrom
            ? nodeX.findOutputSlot(slotX, false)
            : nodeX.findInputSlot(slotX, false)
          slotX = isFrom ? nodeX.outputs[slotX] : nodeX.inputs[slotX]
          break
        case 'object':
          if (slotX === null) {
            console.warn('Cant get slot information', slotX)
            return
          }

          // ok slotX
          iSlotConn = isFrom
            ? nodeX.findOutputSlot(slotX.name)
            : nodeX.findInputSlot(slotX.name)
          break
        case 'number':
          iSlotConn = slotX
          slotX = isFrom ? nodeX.outputs[slotX] : nodeX.inputs[slotX]
          break
        default:
          console.warn('Cant get slot information', slotX)
          return
      }
    }

    const options = ['Add Node', 'Add Reroute', null]

    if (opts.allow_searchbox) {
      options.push('Search', null)
    }

    // get defaults nodes for this slottype
    const fromSlotType = slotX.type == LiteGraph.EVENT ? '_event_' : slotX.type
    const slotTypesDefault = isFrom
      ? LiteGraph.slot_types_default_out
      : LiteGraph.slot_types_default_in
    if (slotTypesDefault?.[fromSlotType]) {
      if (typeof slotTypesDefault[fromSlotType] == 'object') {
        for (const typeX in slotTypesDefault[fromSlotType]) {
          options.push(slotTypesDefault[fromSlotType][typeX])
        }
      } else {
        options.push(slotTypesDefault[fromSlotType])
      }
    }

    // build menu
    const menu = new LiteGraph.ContextMenu<string>(options, {
      event: opts.e,
      extra: slotX,
      title:
        (slotX && slotX.name != ''
          ? slotX.name + (fromSlotType ? ' | ' : '')
          : '') + (slotX && fromSlotType ? fromSlotType : ''),
      callback: inner_clicked
    })

    return menu

    // callback
    function inner_clicked(
      v: string | undefined,
      options: IContextMenuOptions<string, INodeInputSlot | INodeOutputSlot>,
      e: MouseEvent
    ) {
      switch (v) {
        case 'Add Node':
          LGraphCanvas.onMenuAdd(null, null, e, menu, function (node) {
            if (!node) return

            if (isFrom) {
              if (!opts.nodeFrom)
                throw new TypeError(
                  'Cannot add node to SubgraphInputNode: nodeFrom was null'
                )
              const slot = opts.nodeFrom.connectByType(
                iSlotConn,
                node,
                fromSlotType,
                { afterRerouteId }
              )
              if (!slot) console.warn('Failed to make new connection.')
              // }
            } else {
              if (!opts.nodeTo)
                throw new TypeError(
                  'Cannot add node to SubgraphInputNode: nodeTo was null'
                )
              opts.nodeTo.connectByTypeOutput(iSlotConn, node, fromSlotType, {
                afterRerouteId
              })
            }
          })
          break
        case 'Add Reroute': {
          const node = isFrom ? opts.nodeFrom : opts.nodeTo
          const slot = options.extra

          if (!graph) throw new NullGraphError()
          if (!node) throw new TypeError('Cannot add reroute: node was null')
          if (!slot) throw new TypeError('Cannot add reroute: slot was null')
          if (!opts.e)
            throw new TypeError(
              'Cannot add reroute: CanvasPointerEvent was null'
            )

          if (node instanceof SubgraphIONodeBase) {
            throw new TypeError(
              'Cannot add floating reroute to Subgraph IO Nodes'
            )
          } else {
            const reroute = node.connectFloatingReroute(
              [opts.e.canvasX, opts.e.canvasY],
              slot,
              afterRerouteId
            )
            if (!reroute) throw new Error('Failed to create reroute')
          }

          dirty()
          break
        }
        case 'Search':
          if (isFrom) {
            opts.showSearchBox(e, {
              // @ts-expect-error - Subgraph types
              node_from: opts.nodeFrom,
              // @ts-expect-error - Subgraph types
              slot_from: slotX,
              type_filter_in: fromSlotType
            })
          } else {
            opts.showSearchBox(e, {
              // @ts-expect-error - Subgraph types
              node_to: opts.nodeTo,
              // @ts-expect-error - Subgraph types
              slot_from: slotX,
              type_filter_out: fromSlotType
            })
          }
          break
        default: {
          const customProps = {
            position: [opts.e?.canvasX ?? 0, opts.e?.canvasY ?? 0],
            nodeType: v,
            afterRerouteId
          } satisfies Partial<ICreateDefaultNodeOptions>

          const options = Object.assign(opts, customProps)
          if (!that.createDefaultNodeForSlot(options)) break
        }
      }
    }
  }

  // refactor: there are different dialogs, some uses createDialog some dont
  prompt(
    title: string,
    value: string | number,
    callback: (value: string) => void,
    event: CanvasPointerEvent,
    multiline?: boolean
  ): HTMLDivElement {
    const that = this
    title = title || ''

    const customProperties = {
      is_modified: false,
      className: 'graphdialog rounded',
      innerHTML: multiline
        ? "<span class='name'></span> <textarea autofocus class='value'></textarea><button class='rounded'>OK</button>"
        : "<span class='name'></span> <input autofocus type='text' class='value'/><button class='rounded'>OK</button>",
      close() {
        that.prompt_box = null
        if (dialog.parentNode) {
          dialog.remove()
        }
      }
    } satisfies Partial<IDialog>

    const div = document.createElement('div')
    const dialog: PromptDialog = Object.assign(div, customProperties)

    const graphcanvas = LGraphCanvas.active_canvas
    const { canvas } = graphcanvas
    if (!canvas.parentNode)
      throw new TypeError(
        'canvas element parentNode was null when opening a prompt.'
      )
    canvas.parentNode.append(dialog)

    if (this.ds.scale > 1) dialog.style.transform = `scale(${this.ds.scale})`

    let dialogCloseTimer: ReturnType<typeof setTimeout> | undefined
    let prevent_timeout = 0
    LiteGraph.pointerListenerAdd(dialog, 'leave', function () {
      if (prevent_timeout) return
      if (LiteGraph.dialog_close_on_mouse_leave) {
        if (!dialog.is_modified && LiteGraph.dialog_close_on_mouse_leave) {
          dialogCloseTimer = setTimeout(
            dialog.close,
            LiteGraph.dialog_close_on_mouse_leave_delay
          )
        }
      }
    })
    LiteGraph.pointerListenerAdd(dialog, 'enter', function () {
      if (LiteGraph.dialog_close_on_mouse_leave && dialogCloseTimer)
        clearTimeout(dialogCloseTimer)
    })
    const selInDia = dialog.querySelectorAll('select')
    if (selInDia) {
      // if filtering, check focus changed to comboboxes and prevent closing
      for (const selIn of selInDia) {
        selIn.addEventListener('click', function () {
          prevent_timeout++
        })
        selIn.addEventListener('blur', function () {
          prevent_timeout = 0
        })
        selIn.addEventListener('change', function () {
          prevent_timeout = -1
        })
      }
    }
    this.prompt_box?.close()
    this.prompt_box = dialog

    const name_element: HTMLSpanElement | null = dialog.querySelector('.name')
    if (!name_element) throw new TypeError('name_element was null')

    name_element.textContent = title
    const value_element: HTMLInputElement | null =
      dialog.querySelector('.value')
    if (!value_element) throw new TypeError('value_element was null')

    value_element.value = String(value)
    value_element.select()

    const input = value_element
    input.addEventListener('keydown', function (e: KeyboardEvent) {
      dialog.is_modified = true
      if (e.key == 'Escape') {
        // ESC
        dialog.close()
      } else if (
        e.key == 'Enter' &&
        (e.target as Element).localName != 'textarea'
      ) {
        if (callback) {
          callback(this.value)
        }
        dialog.close()
      } else {
        return
      }
      e.preventDefault()
      e.stopPropagation()
    })

    const button = dialog.querySelector('button')
    if (!button) throw new TypeError('button was null when opening prompt')

    button.addEventListener('click', function () {
      callback?.(input.value)
      that.setDirty(true)
      dialog.close()
    })

    const rect = canvas.getBoundingClientRect()
    let offsetx = -20
    let offsety = -20
    if (rect) {
      offsetx -= rect.left
      offsety -= rect.top
    }

    if (event) {
      dialog.style.left = `${event.clientX + offsetx}px`
      dialog.style.top = `${event.clientY + offsety}px`
    } else {
      dialog.style.left = `${canvas.width * 0.5 + offsetx}px`
      dialog.style.top = `${canvas.height * 0.5 + offsety}px`
    }

    setTimeout(function () {
      input.focus()
      const clickTime = Date.now()
      function handleOutsideClick(e: Event) {
        if (e.target === canvas && Date.now() - clickTime > 256) {
          dialog.close()
          canvas.parentElement?.removeEventListener('click', handleOutsideClick)
          canvas.parentElement?.removeEventListener(
            'touchend',
            handleOutsideClick
          )
        }
      }
      canvas.parentElement?.addEventListener('click', handleOutsideClick)
      canvas.parentElement?.addEventListener('touchend', handleOutsideClick)
    }, 10)

    return dialog
  }

  showSearchBox(
    event: MouseEvent | null,
    searchOptions?: IShowSearchOptions
  ): HTMLDivElement {
    // proposed defaults
    const options: IShowSearchOptions = {
      slot_from: null,
      node_from: null,
      node_to: null,
      // TODO check for registered_slot_[in/out]_types not empty
      // this will be checked for functionality enabled : filter on slot type, in and out
      do_type_filter: LiteGraph.search_filter_enabled,

      // these are default: pass to set initially set values
      // @ts-expect-error Property missing from interface definition
      type_filter_in: false,

      type_filter_out: false,
      show_general_if_none_on_typefilter: true,
      show_general_after_typefiltered: true,
      hide_on_mouse_leave: LiteGraph.search_hide_on_mouse_leave,
      show_all_if_empty: true,
      show_all_on_open: LiteGraph.search_show_all_on_open
    }
    Object.assign(options, searchOptions)

    // console.log(options);

    const that = this
    const graphcanvas = LGraphCanvas.active_canvas
    const { canvas } = graphcanvas
    const root_document = canvas.ownerDocument || document

    const div = document.createElement('div')
    const dialog = Object.assign(div, {
      close(this: typeof div) {
        that.search_box = undefined
        this.blur()
        canvas.focus()
        root_document.body.style.overflow = ''

        // important, if canvas loses focus keys won't be captured
        setTimeout(() => canvas.focus(), 20)
        dialog.remove()
      }
    } satisfies Partial<HTMLDivElement> & ICloseable)
    dialog.className = 'litegraph litesearchbox graphdialog rounded'
    dialog.innerHTML =
      "<span class='name'>Search</span> <input autofocus type='text' class='value rounded'/>"
    if (options.do_type_filter) {
      dialog.innerHTML +=
        "<select class='slot_in_type_filter'><option value=''></option></select>"
      dialog.innerHTML +=
        "<select class='slot_out_type_filter'><option value=''></option></select>"
    }
    const helper = document.createElement('div')
    helper.className = 'helper'
    dialog.append(helper)

    if (root_document.fullscreenElement) {
      root_document.fullscreenElement.append(dialog)
    } else {
      root_document.body.append(dialog)
      root_document.body.style.overflow = 'hidden'
    }

    // dialog element has been appended
    let selIn
    let selOut
    if (options.do_type_filter) {
      selIn = dialog.querySelector('.slot_in_type_filter')
      selOut = dialog.querySelector('.slot_out_type_filter')
    }

    if (this.ds.scale > 1) {
      dialog.style.transform = `scale(${this.ds.scale})`
    }

    // hide on mouse leave
    if (options.hide_on_mouse_leave) {
      let prevent_timeout = 0
      let timeout_close: ReturnType<typeof setTimeout> | null = null
      LiteGraph.pointerListenerAdd(dialog, 'enter', function () {
        if (timeout_close) {
          clearTimeout(timeout_close)
          timeout_close = null
        }
      })
      dialog.addEventListener('pointerleave', function () {
        if (prevent_timeout) return

        const hideDelay = options.hide_on_mouse_leave
        const delay = typeof hideDelay === 'number' ? hideDelay : 500
        timeout_close = setTimeout(dialog.close, delay)
      })
      // if filtering, check focus changed to comboboxes and prevent closing
      if (options.do_type_filter) {
        if (!selIn)
          throw new TypeError('selIn was null when showing search box')
        if (!selOut)
          throw new TypeError('selOut was null when showing search box')

        selIn.addEventListener('click', function () {
          prevent_timeout++
        })
        selIn.addEventListener('blur', function () {
          prevent_timeout = 0
        })
        selIn.addEventListener('change', function () {
          prevent_timeout = -1
        })
        selOut.addEventListener('click', function () {
          prevent_timeout++
        })
        selOut.addEventListener('blur', function () {
          prevent_timeout = 0
        })
        selOut.addEventListener('change', function () {
          prevent_timeout = -1
        })
      }
    }

    // @ts-expect-error Panel?
    that.search_box?.close()
    that.search_box = dialog

    let first: string | null = null
    let timeout: ReturnType<typeof setTimeout> | null = null
    let selected: ChildNode | null = null

    const maybeInput = dialog.querySelector('input')
    if (!maybeInput) throw new TypeError('Could not create search input box.')

    const input = maybeInput

    if (input) {
      input.addEventListener('blur', function () {
        this.focus()
      })
      input.addEventListener('keydown', function (e) {
        if (e.key == 'ArrowUp') {
          // UP
          changeSelection(false)
        } else if (e.key == 'ArrowDown') {
          // DOWN
          changeSelection(true)
        } else if (e.key == 'Escape') {
          // ESC
          dialog.close()
        } else if (e.key == 'Enter') {
          if (selected instanceof HTMLElement) {
            select(unescape(String(selected.dataset['type'])))
          } else if (first) {
            select(first)
          } else {
            dialog.close()
          }
        } else {
          if (timeout) {
            clearInterval(timeout)
          }
          timeout = setTimeout(refreshHelper, 10)
          return
        }
        e.preventDefault()
        e.stopPropagation()
        e.stopImmediatePropagation()
        return true
      })
    }

    // if should filter on type, load and fill selected and choose elements if passed
    if (options.do_type_filter) {
      if (selIn) {
        const aSlots = LiteGraph.slot_types_in
        const nSlots = aSlots.length

        if (
          options.type_filter_in == LiteGraph.EVENT ||
          options.type_filter_in == LiteGraph.ACTION
        ) {
          options.type_filter_in = '_event_'
        }
        for (let iK = 0; iK < nSlots; iK++) {
          const opt = document.createElement('option')
          opt.value = aSlots[iK]
          opt.innerHTML = aSlots[iK]
          selIn.append(opt)
          if (
            // @ts-expect-error Property missing from interface definition
            options.type_filter_in !== false &&
            String(options.type_filter_in).toLowerCase() ==
              String(aSlots[iK]).toLowerCase()
          ) {
            opt.selected = true
          }
        }
        selIn.addEventListener('change', function () {
          refreshHelper()
        })
      }
      if (selOut) {
        const aSlots = LiteGraph.slot_types_out

        if (
          options.type_filter_out == LiteGraph.EVENT ||
          options.type_filter_out == LiteGraph.ACTION
        ) {
          options.type_filter_out = '_event_'
        }
        for (const aSlot of aSlots) {
          const opt = document.createElement('option')
          opt.value = aSlot
          opt.innerHTML = aSlot
          selOut.append(opt)
          if (
            options.type_filter_out !== false &&
            String(options.type_filter_out).toLowerCase() ==
              String(aSlot).toLowerCase()
          ) {
            opt.selected = true
          }
        }
        selOut.addEventListener('change', function () {
          refreshHelper()
        })
      }
    }

    // compute best position
    const rect = canvas.getBoundingClientRect()

    // Handles cases where the searchbox is initiated by
    // non-click events. e.g. Keyboard shortcuts
    const safeEvent =
      event ??
      new MouseEvent('click', {
        clientX: rect.left + rect.width * 0.5,
        clientY: rect.top + rect.height * 0.5,
        // @ts-expect-error layerY is a nonstandard property
        layerY: rect.top + rect.height * 0.5
      })

    const left = safeEvent.clientX - 80
    const top = safeEvent.clientY - 20
    dialog.style.left = `${left}px`
    dialog.style.top = `${top}px`

    // To avoid out of screen problems
    if (safeEvent.layerY > rect.height - 200) {
      helper.style.maxHeight = `${rect.height - safeEvent.layerY - 20}px`
    }
    requestAnimationFrame(function () {
      input.focus()
    })
    if (options.show_all_on_open) refreshHelper()

    function select(name: string) {
      if (name) {
        if (that.onSearchBoxSelection) {
          that.onSearchBoxSelection(name, safeEvent, graphcanvas)
        } else {
          if (!graphcanvas.graph) throw new NullGraphError()

          graphcanvas.graph.beforeChange()
          const node = LiteGraph.createNode(name)
          if (node) {
            node.pos = graphcanvas.convertEventToCanvasOffset(safeEvent)
            graphcanvas.graph.add(node, false)
          }

          // join node after inserting
          if (options.node_from) {
            let iS: number | false = false
            switch (typeof options.slot_from) {
              case 'string':
                iS = options.node_from.findOutputSlot(options.slot_from)
                break
              case 'object':
                if (options.slot_from == null)
                  throw new TypeError(
                    'options.slot_from was null when showing search box'
                  )

                iS = options.slot_from.name
                  ? options.node_from.findOutputSlot(options.slot_from.name)
                  : -1
                // @ts-expect-error - slot_index property
                if (iS == -1 && options.slot_from.slot_index !== undefined)
                  // @ts-expect-error - slot_index property
                  iS = options.slot_from.slot_index
                break
              case 'number':
                iS = options.slot_from
                break
              default:
                // try with first if no name set
                iS = 0
            }
            if (iS !== false && options.node_from.outputs[iS] !== undefined) {
              if (iS > -1) {
                if (node == null)
                  throw new TypeError(
                    'options.slot_from was null when showing search box'
                  )

                options.node_from.connectByType(
                  iS,
                  node,
                  options.node_from.outputs[iS].type
                )
              }
            } else {
              // console.warn("can't find slot " + options.slot_from);
            }
          }
          if (options.node_to) {
            let iS: number | false = false
            switch (typeof options.slot_from) {
              case 'string':
                iS = options.node_to.findInputSlot(options.slot_from)
                break
              case 'object':
                if (options.slot_from == null)
                  throw new TypeError(
                    'options.slot_from was null when showing search box'
                  )

                iS = options.slot_from.name
                  ? options.node_to.findInputSlot(options.slot_from.name)
                  : -1
                // @ts-expect-error - slot_index property
                if (iS == -1 && options.slot_from.slot_index !== undefined)
                  // @ts-expect-error - slot_index property
                  iS = options.slot_from.slot_index
                break
              case 'number':
                iS = options.slot_from
                break
              default:
                // try with first if no name set
                iS = 0
            }
            if (iS !== false && options.node_to.inputs[iS] !== undefined) {
              if (iS > -1) {
                if (node == null)
                  throw new TypeError(
                    'options.slot_from was null when showing search box'
                  )
                // try connection
                options.node_to.connectByTypeOutput(
                  iS,
                  node,
                  options.node_to.inputs[iS].type
                )
              }
            } else {
              // console.warn("can't find slot_nodeTO " + options.slot_from);
            }
          }

          graphcanvas.graph.afterChange()
        }
      }

      dialog.close()
    }

    function changeSelection(forward: boolean) {
      const prev = selected
      if (!selected) {
        selected = forward
          ? helper.childNodes[0]
          : helper.childNodes[helper.childNodes.length]
      } else if (selected instanceof Element) {
        selected.classList.remove('selected')
        selected = forward ? selected.nextSibling : selected.previousSibling
        selected ||= prev
      }

      if (selected instanceof Element) {
        selected.classList.add('selected')
        selected.scrollIntoView({ block: 'end', behavior: 'smooth' })
      }
    }

    function refreshHelper() {
      timeout = null
      let str = input.value
      first = null
      helper.innerHTML = ''
      if (!str && !options.show_all_if_empty) return

      if (that.onSearchBox) {
        const list = that.onSearchBox(helper, str, graphcanvas)
        if (list) {
          for (const item of list) {
            addResult(item)
          }
        }
      } else {
        let c = 0
        str = str.toLowerCase()
        if (!graphcanvas.graph) throw new NullGraphError()

        const filter = graphcanvas.filter || graphcanvas.graph.filter

        // filter by type preprocess
        let sIn: HTMLSelectElement | null = null
        let sOut: HTMLSelectElement | null = null
        if (options.do_type_filter && that.search_box) {
          sIn = that.search_box.querySelector<HTMLSelectElement>(
            '.slot_in_type_filter'
          )
          sOut = that.search_box.querySelector<HTMLSelectElement>(
            '.slot_out_type_filter'
          )
        }

        const keys = Object.keys(LiteGraph.registered_node_types)
        const filtered = keys.filter((x) => inner_test_filter(x))

        for (const item of filtered) {
          addResult(item)
          if (
            LGraphCanvas.search_limit !== -1 &&
            c++ > LGraphCanvas.search_limit
          )
            break
        }

        // add general type if filtering
        if (
          options.show_general_after_typefiltered &&
          (sIn?.value || sOut?.value)
        ) {
          const filtered_extra: string[] = []
          for (const i in LiteGraph.registered_node_types) {
            if (
              inner_test_filter(i, {
                inTypeOverride: sIn && sIn.value ? '*' : false,
                outTypeOverride: sOut && sOut.value ? '*' : false
              })
            ) {
              filtered_extra.push(i)
            }
          }
          for (const extraItem of filtered_extra) {
            addResult(extraItem, 'generic_type')
            if (
              LGraphCanvas.search_limit !== -1 &&
              c++ > LGraphCanvas.search_limit
            )
              break
          }
        }

        // check il filtering gave no results
        if (
          (sIn?.value || sOut?.value) &&
          helper.childNodes.length == 0 &&
          options.show_general_if_none_on_typefilter
        ) {
          const filtered_extra: string[] = []
          for (const i in LiteGraph.registered_node_types) {
            if (inner_test_filter(i, { skipFilter: true }))
              filtered_extra.push(i)
          }
          for (const extraItem of filtered_extra) {
            addResult(extraItem, 'not_in_filter')
            if (
              LGraphCanvas.search_limit !== -1 &&
              c++ > LGraphCanvas.search_limit
            )
              break
          }
        }

        function inner_test_filter(
          type: string,
          optsIn?: {
            inTypeOverride?: string | boolean
            outTypeOverride?: string | boolean
            skipFilter?: boolean
          }
        ): boolean {
          optsIn = optsIn || {}
          const optsDef = {
            skipFilter: false,
            inTypeOverride: false,
            outTypeOverride: false
          }
          const opts = Object.assign(optsDef, optsIn)
          const ctor = LiteGraph.registered_node_types[type]
          if (filter && ctor.filter != filter) return false
          if (
            (!options.show_all_if_empty || str) &&
            !type.toLowerCase().includes(str) &&
            (!ctor.title || !ctor.title.toLowerCase().includes(str))
          ) {
            return false
          }

          // filter by slot IN, OUT types
          if (options.do_type_filter && !opts.skipFilter) {
            const sType = type

            let sV: string | undefined =
              typeof opts.inTypeOverride === 'string'
                ? opts.inTypeOverride
                : sIn?.value
            // type is stored
            if (sIn && sV && LiteGraph.registered_slot_in_types[sV]?.nodes) {
              const doesInc =
                LiteGraph.registered_slot_in_types[sV].nodes.includes(sType)
              if (doesInc === false) return false
            }

            sV = sOut?.value
            if (typeof opts.outTypeOverride === 'string')
              sV = opts.outTypeOverride
            // type is stored
            if (sOut && sV && LiteGraph.registered_slot_out_types[sV]?.nodes) {
              const doesInc =
                LiteGraph.registered_slot_out_types[sV].nodes.includes(sType)
              if (doesInc === false) return false
            }
          }
          return true
        }
      }

      function addResult(type: string, className?: string): void {
        const help = document.createElement('div')
        first ||= type

        const nodeType = LiteGraph.registered_node_types[type]
        if (nodeType?.title) {
          help.textContent = nodeType?.title
          const typeEl = document.createElement('span')
          typeEl.className = 'litegraph lite-search-item-type'
          typeEl.textContent = type
          help.append(typeEl)
        } else {
          help.textContent = type
        }

        help.dataset['type'] = escape(type)
        help.className = 'litegraph lite-search-item'
        if (className) {
          help.className += ` ${className}`
        }
        help.addEventListener('click', function () {
          select(unescape(String(this.dataset['type'])))
        })
        helper.append(help)
      }
    }

    return dialog
  }

  showEditPropertyValue(
    node: LGraphNode,
    property: string,
    options: IDialogOptions
  ): IDialog | undefined {
    if (!node || node.properties[property] === undefined) return

    options = options || {}

    const info = node.getPropertyInfo(property)
    const { type } = info

    let input_html = ''

    if (
      type == 'string' ||
      type == 'number' ||
      type == 'array' ||
      type == 'object'
    ) {
      input_html = "<input autofocus type='text' class='value'/>"
    } else if ((type == 'enum' || type == 'combo') && info.values) {
      input_html = "<select autofocus type='text' class='value'>"
      for (const i in info.values) {
        const v = Array.isArray(info.values) ? info.values[i] : i

        const selected = v == node.properties[property] ? 'selected' : ''
        input_html += `<option value='${v}' ${selected}>${info.values[i]}</option>`
      }
      input_html += '</select>'
    } else if (type == 'boolean' || type == 'toggle') {
      const checked = node.properties[property] ? 'checked' : ''
      input_html = `<input autofocus type='checkbox' class='value' ${checked}/>`
    } else {
      console.warn(`unknown type: ${type}`)
      return
    }

    const dialog = this.createDialog(
      `<span class='name'>${info.label || property}</span>${input_html}<button>OK</button>`,
      options
    )

    let input: HTMLInputElement | HTMLSelectElement | null
    if ((type == 'enum' || type == 'combo') && info.values) {
      input = dialog.querySelector('select')
      input?.addEventListener('change', function (e) {
        dialog.modified()
        setValue((e.target as HTMLSelectElement)?.value)
      })
    } else if (type == 'boolean' || type == 'toggle') {
      input = dialog.querySelector('input')
      input?.addEventListener('click', function () {
        dialog.modified()
        // @ts-expect-error setValue function signature not strictly typed
        setValue(!!input.checked)
      })
    } else {
      input = dialog.querySelector('input')
      if (input) {
        input.addEventListener('blur', function () {
          this.focus()
        })

        let v =
          node.properties[property] !== undefined
            ? node.properties[property]
            : ''
        if (type !== 'string') {
          v = JSON.stringify(v)
        }

        // @ts-expect-error HTMLInputElement.value expects string but v can be other types
        input.value = v
        input.addEventListener('keydown', function (e) {
          if (e.key == 'Escape') {
            // ESC
            dialog.close()
          } else if (e.key == 'Enter') {
            // ENTER
            // save
            inner()
          } else {
            dialog.modified()
            return
          }
          e.preventDefault()
          e.stopPropagation()
        })
      }
    }
    input?.focus()

    const button = dialog.querySelector('button')
    if (!button)
      throw new TypeError('Show edit property value button was null.')
    button.addEventListener('click', inner)

    function inner() {
      setValue(input?.value)
    }
    const dirty = () => this._dirty()

    function setValue(value: string | number | undefined) {
      if (
        info?.values &&
        typeof info.values === 'object' &&
        info.values[value] != undefined
      ) {
        value = info.values[value]
      }

      if (typeof node.properties[property] == 'number') {
        value = Number(value)
      }
      if (type == 'array' || type == 'object') {
        // @ts-expect-error JSON.parse doesn't care.
        value = JSON.parse(value)
      }
      node.properties[property] = value
      if (node.graph) {
        node.graph._version++
      }
      node.onPropertyChanged?.(property, value)
      options.onclose?.()
      dialog.close()
      dirty()
    }

    return dialog
  }

  // TODO refactor, there are different dialog, some uses createDialog, some dont
  createDialog(html: string, options: IDialogOptions): IDialog {
    const def_options = {
      checkForInput: false,
      closeOnLeave: true,
      closeOnLeave_checkModified: true
    }
    options = Object.assign(def_options, options || {})

    const customProperties = {
      className: 'graphdialog',
      innerHTML: html,
      is_modified: false,
      modified() {
        this.is_modified = true
      },
      close(this: IDialog) {
        this.remove()
      }
    } satisfies Partial<IDialog>

    const div = document.createElement('div')
    const dialog: IDialog = Object.assign(div, customProperties)

    const rect = this.canvas.getBoundingClientRect()
    let offsetx = -20
    let offsety = -20
    if (rect) {
      offsetx -= rect.left
      offsety -= rect.top
    }

    if (options.position) {
      offsetx += options.position[0]
      offsety += options.position[1]
    } else if (options.event) {
      offsetx += options.event.clientX
      offsety += options.event.clientY
    } else {
      // centered
      offsetx += this.canvas.width * 0.5
      offsety += this.canvas.height * 0.5
    }

    dialog.style.left = `${offsetx}px`
    dialog.style.top = `${offsety}px`

    if (!this.canvas.parentNode)
      throw new TypeError('Canvas parent element was null.')
    this.canvas.parentNode.append(dialog)

    // acheck for input and use default behaviour: save on enter, close on esc
    if (options.checkForInput) {
      const aI = dialog.querySelectorAll('input')
      if (aI) {
        for (const iX of aI) {
          iX.addEventListener('keydown', function (e) {
            dialog.modified()
            if (e.key == 'Escape') {
              dialog.close()
            } else if (e.key != 'Enter') {
              return
            }
            e.preventDefault()
            e.stopPropagation()
          })
          iX.focus()
        }
      }
    }

    let dialogCloseTimer: ReturnType<typeof setTimeout> | undefined
    let prevent_timeout = 0
    dialog.addEventListener('mouseleave', function () {
      if (prevent_timeout) return

      if (!dialog.is_modified && LiteGraph.dialog_close_on_mouse_leave) {
        dialogCloseTimer = setTimeout(
          dialog.close,
          LiteGraph.dialog_close_on_mouse_leave_delay
        )
      }
    })
    dialog.addEventListener('mouseenter', function () {
      if (options.closeOnLeave || LiteGraph.dialog_close_on_mouse_leave) {
        if (dialogCloseTimer) clearTimeout(dialogCloseTimer)
      }
    })
    const selInDia = dialog.querySelectorAll('select')
    // if filtering, check focus changed to comboboxes and prevent closing
    if (selInDia) {
      for (const selIn of selInDia) {
        selIn.addEventListener('click', function () {
          prevent_timeout++
        })
        selIn.addEventListener('blur', function () {
          prevent_timeout = 0
        })
        selIn.addEventListener('change', function () {
          prevent_timeout = -1
        })
      }
    }

    return dialog
  }

  createPanel(title: string, options: ICreatePanelOptions): Panel {
    options = options || {}

    const root = document.createElement('div') as Panel
    root.className = 'litegraph dialog'
    root.innerHTML =
      "<div class='dialog-header'><span class='dialog-title'></span></div><div class='dialog-content'></div><div style='display:none;' class='dialog-alt-content'></div><div class='dialog-footer'></div>"
    root.header = root.querySelector('.dialog-header')!

    if (options.width)
      root.style.width =
        options.width + (typeof options.width === 'number' ? 'px' : '')
    if (options.height)
      root.style.height =
        options.height + (typeof options.height === 'number' ? 'px' : '')
    if (options.closable) {
      const close = document.createElement('span')
      close.innerHTML = '&#10005;'
      close.classList.add('close')
      close.addEventListener('click', function () {
        root.close()
      })
      root.header.append(close)
    }
    root.title_element = root.querySelector('.dialog-title')!
    root.title_element.textContent = title
    root.content = root.querySelector('.dialog-content')!
    root.alt_content = root.querySelector('.dialog-alt-content')!
    root.footer = root.querySelector('.dialog-footer')!
    root.footer.style.marginTop = '-96px'

    root.close = function () {
      if (typeof root.onClose == 'function') root.onClose()
      root.remove()
      this.remove()
    }

    // function to swap panel content
    root.toggleAltContent = function (force?: boolean) {
      let vTo: string
      let vAlt: string
      if (force !== undefined) {
        vTo = force ? 'block' : 'none'
        vAlt = force ? 'none' : 'block'
      } else {
        vTo = root.alt_content.style.display != 'block' ? 'block' : 'none'
        vAlt = root.alt_content.style.display != 'block' ? 'none' : 'block'
      }
      root.alt_content.style.display = vTo
      root.content.style.display = vAlt
    }

    root.toggleFooterVisibility = function (force?: boolean) {
      let vTo: string
      if (force !== undefined) {
        vTo = force ? 'block' : 'none'
      } else {
        vTo = root.footer.style.display != 'block' ? 'block' : 'none'
      }
      root.footer.style.display = vTo
    }

    root.clear = function () {
      this.content.innerHTML = ''
    }

    root.addHTML = function (
      code: string,
      classname?: string,
      on_footer?: boolean
    ) {
      const elem = document.createElement('div')
      if (classname) elem.className = classname
      elem.innerHTML = code
      if (on_footer) root.footer.append(elem)
      else root.content.append(elem)
      return elem
    }

    root.addButton = function (
      name: string,
      callback: () => void,
      options?: unknown
    ): PanelButton {
      const elem = document.createElement('button') as PanelButton
      elem.textContent = name
      elem.options = options
      elem.classList.add('btn')
      elem.addEventListener('click', callback)
      root.footer.append(elem)
      return elem
    }

    root.addSeparator = function () {
      const elem = document.createElement('div')
      elem.className = 'separator'
      root.content.append(elem)
    }

    root.addWidget = function (
      type: string,
      name: string,
      value: TWidgetValue,
      options?: PanelWidgetOptions,
      callback?: PanelWidgetCallback
    ): PanelWidget {
      options = options || {}
      let str_value = String(value)
      type = type.toLowerCase()
      if (type == 'number' && typeof value === 'number')
        str_value = value.toFixed(3)

      const elem: PanelWidget = document.createElement('div') as PanelWidget
      elem.className = 'property'
      elem.innerHTML =
        "<span class='property_name'></span><span class='property_value'></span>"
      const nameSpan = elem.querySelector('.property_name')
      if (!nameSpan) throw new TypeError('Property name element was null.')

      nameSpan.textContent = options.label || name
      const value_element: HTMLSpanElement | null =
        elem.querySelector('.property_value')
      if (!value_element) throw new TypeError('Property name element was null.')
      value_element.textContent = str_value
      elem.dataset['property'] = name
      elem.dataset['type'] = options.type || type
      elem.options = options
      elem.value = value

      if (type == 'code') {
        elem.addEventListener('click', function () {
          const property = this.dataset['property']
          if (property) root.inner_showCodePad?.(property)
        })
      } else if (type == 'boolean') {
        elem.classList.add('boolean')
        if (value) elem.classList.add('bool-on')
        elem.addEventListener('click', () => {
          const propname = elem.dataset['property']
          elem.value = !elem.value
          elem.classList.toggle('bool-on')
          if (!value_element)
            throw new TypeError('Property name element was null.')

          value_element.textContent = elem.value ? 'true' : 'false'
          innerChange(propname, elem.value)
        })
      } else if (type == 'string' || type == 'number') {
        if (!value_element)
          throw new TypeError('Property name element was null.')
        value_element.setAttribute('contenteditable', 'true')
        value_element.addEventListener('keydown', function (e) {
          // allow for multiline
          if (e.code == 'Enter' && (type != 'string' || !e.shiftKey)) {
            e.preventDefault()
            this.blur()
          }
        })
        value_element.addEventListener('blur', function () {
          let v: string | number | null = this.textContent
          const propname = this.parentElement?.dataset['property']
          const proptype = this.parentElement?.dataset['type']
          if (proptype == 'number') v = Number(v)
          innerChange(propname, v)
        })
      } else if (type == 'enum' || type == 'combo') {
        const str_value = LGraphCanvas.getPropertyPrintableValue(
          value,
          options.values
        )
        if (!value_element)
          throw new TypeError('Property name element was null.')
        value_element.textContent = str_value ?? ''

        value_element.addEventListener('click', function (event) {
          const values = options?.values || []
          const propname = this.parentElement?.dataset['property']
          const inner_clicked = (v?: string) => {
            this.textContent = v ?? null
            innerChange(propname, v)
            return false
          }
          new LiteGraph.ContextMenu(values, {
            event,
            className: 'dark',
            callback: inner_clicked
          })
        })
      }

      root.content.append(elem)

      function innerChange(name: string | undefined, value: TWidgetValue) {
        const opts = options || {}
        opts.callback?.(name, value, opts)
        callback?.(name, value, opts)
      }

      return elem
    }

    if (typeof root.onOpen == 'function') root.onOpen()

    return root
  }

  closePanels(): void {
    type MightHaveClose = HTMLDivElement & Partial<ICloseable>
    document.querySelector<MightHaveClose>('#node-panel')?.close?.()
    document.querySelector<MightHaveClose>('#option-panel')?.close?.()
  }

  showShowNodePanel(node: LGraphNode): void {
    this.SELECTED_NODE = node
    this.closePanels()
    const ref_window = this.getCanvasWindow()
    const panel = this.createPanel(node.title || '', {
      closable: true,
      window: ref_window,
      onOpen: () => {
        this.NODEPANEL_IS_OPEN = true
      },
      onClose: () => {
        this.NODEPANEL_IS_OPEN = false
        this.node_panel = undefined
      }
    })
    this.node_panel = panel
    panel.id = 'node-panel'
    panel.node = node
    panel.classList.add('settings')
    panel.style.position = 'absolute'
    panel.style.top = '96px'
    panel.style.left = '65px'

    const inner_refresh = () => {
      // clear
      panel.content.innerHTML = ''
      panel.addHTML(
        // @ts-expect-error - desc property
        `<span class='node_type'>${node.type}</span><span class='node_desc'>${node.constructor.desc || ''}</span><span class='separator'></span>`
      )

      panel.addHTML('<h3>Properties</h3>')

      const fUpdate: PanelWidgetCallback = (name, value) => {
        if (!this.graph) throw new NullGraphError()
        if (!name) return
        this.graph.beforeChange(node)
        switch (name) {
          case 'Title':
            if (typeof value !== 'string')
              throw new TypeError(
                'Attempting to set title to non-string value.'
              )

            node.title = value
            break
          case 'Mode': {
            if (typeof value !== 'string')
              throw new TypeError('Attempting to set mode to non-string value.')

            const kV = Object.values(LiteGraph.NODE_MODES).indexOf(value)
            if (kV !== -1 && LiteGraph.NODE_MODES[kV]) {
              node.changeMode(kV)
            } else {
              console.warn(`unexpected mode: ${value}`)
            }
            break
          }
          case 'Color':
            if (typeof value !== 'string')
              throw new TypeError(
                'Attempting to set colour to non-string value.'
              )

            if (LGraphCanvas.node_colors[value]) {
              node.color = LGraphCanvas.node_colors[value].color
              node.bgcolor = LGraphCanvas.node_colors[value].bgcolor
            } else {
              console.warn(`unexpected color: ${value}`)
            }
            break
          default:
            node.setProperty(name, value)
            break
        }
        this.graph.afterChange()
        this.dirty_canvas = true
      }

      panel.addWidget('string', 'Title', node.title, {}, fUpdate)

      const mode =
        node.mode == null ? undefined : LiteGraph.NODE_MODES[node.mode]
      panel.addWidget(
        'combo',
        'Mode',
        mode,
        { values: LiteGraph.NODE_MODES },
        fUpdate
      )

      const nodeCol =
        node.color !== undefined
          ? Object.keys(LGraphCanvas.node_colors).filter(function (nK) {
              return LGraphCanvas.node_colors[nK].color == node.color
            })
          : ''

      panel.addWidget(
        'combo',
        'Color',
        nodeCol,
        { values: Object.keys(LGraphCanvas.node_colors) },
        fUpdate
      )

      for (const pName in node.properties) {
        const value = node.properties[pName]
        const info = node.getPropertyInfo(pName)

        // in case the user wants control over the side panel widget
        if (node.onAddPropertyToPanel?.(pName, panel)) continue

        panel.addWidget(info.widget || info.type, pName, value, info, fUpdate)
      }

      panel.addSeparator()

      node.onShowCustomPanelInfo?.(panel)

      // clear
      panel.footer.innerHTML = ''
      panel
        .addButton('Delete', function () {
          if (node.block_delete) return
          if (!node.graph) throw new NullGraphError()

          node.graph.remove(node)
          panel.close()
        })
        .classList.add('delete')
    }

    panel.inner_showCodePad = function (propname: string) {
      panel.classList.remove('settings')
      panel.classList.add('centered')

      panel.alt_content.innerHTML = "<textarea class='code'></textarea>"
      const textarea: HTMLTextAreaElement =
        panel.alt_content.querySelector('textarea')!
      const fDoneWith = function () {
        panel.toggleAltContent(false)
        panel.toggleFooterVisibility(true)
        textarea.remove()
        panel.classList.add('settings')
        panel.classList.remove('centered')
        inner_refresh()
      }
      textarea.value = String(node.properties[propname])
      textarea.addEventListener('keydown', function (e: KeyboardEvent) {
        if (e.code == 'Enter' && e.ctrlKey) {
          node.setProperty(propname, textarea.value)
          fDoneWith()
        }
      })
      panel.toggleAltContent(true)
      panel.toggleFooterVisibility(false)
      textarea.style.height = 'calc(100% - 40px)'

      const assign = panel.addButton('Assign', function () {
        node.setProperty(propname, textarea.value)
        fDoneWith()
      })
      panel.alt_content.append(assign)
      const button = panel.addButton('Close', fDoneWith)
      button.style.float = 'right'
      panel.alt_content.append(button)
    }

    inner_refresh()

    if (!this.canvas.parentNode)
      throw new TypeError('showNodePanel - this.canvas.parentNode was null')
    this.canvas.parentNode.append(panel)
  }

  checkPanels(): void {
    if (!this.canvas) return

    if (!this.canvas.parentNode)
      throw new TypeError('checkPanels - this.canvas.parentNode was null')
    const panels = this.canvas.parentNode.querySelectorAll('.litegraph.dialog')
    for (const panel of panels) {
      // @ts-expect-error Panel
      if (!panel.node) continue
      // @ts-expect-error Panel
      if (!panel.node.graph || panel.graph != this.graph) panel.close()
    }
  }

  getCanvasMenuOptions(): (IContextMenuValue | null)[] {
    let options: (IContextMenuValue<string> | null)[]
    if (this.getMenuOptions) {
      options = this.getMenuOptions()
    } else {
      options = [
        {
          content: 'Add Node',
          has_submenu: true,
          callback: LGraphCanvas.onMenuAdd
        },
        { content: 'Add Group', callback: LGraphCanvas.onGroupAdd },
        {
          content: 'Paste',
          callback: () => {
            this.pasteFromClipboard()
          }
        }
        // { content: "Arrange", callback: that.graph.arrange },
        // {content:"Collapse All", callback: LGraphCanvas.onMenuCollapseAll }
      ]
      if (Object.keys(this.selected_nodes).length > 1) {
        options.push(
          {
            content: 'Convert to Subgraph',
            callback: () => {
              if (!this.selectedItems.size)
                throw new Error('Convert to Subgraph: Nothing selected.')
              this._graph.convertToSubgraph(this.selectedItems)
            }
          },
          {
            content: 'Align',
            has_submenu: true,
            callback: LGraphCanvas.onGroupAlign
          }
        )
      }
    }

    const extra = this.getExtraMenuOptions?.(this, options)
    return Array.isArray(extra) ? options.concat(extra) : options
  }

  // called by processContextMenu to extract the menu list
  getNodeMenuOptions(node: LGraphNode) {
    let options: (
      | IContextMenuValue<string>
      | IContextMenuValue<string | null>
      | IContextMenuValue<INodeSlotContextItem>
      | IContextMenuValue<unknown, LGraphNode>
      | IContextMenuValue<(typeof LiteGraph.VALID_SHAPES)[number]>
      | null
    )[]
    if (node.getMenuOptions) {
      options = node.getMenuOptions(this)
    } else {
      options = [
        {
          content: 'Convert to Subgraph',
          callback: () => {
            // find groupnodes, degroup and select children
            if (this.selectedItems.size) {
              let hasGroups = false
              for (const item of this.selectedItems) {
                const node = item as LGraphNode
                const isGroup =
                  typeof node.type === 'string' &&
                  node.type.startsWith(`${PREFIX}${SEPARATOR}`)
                if (isGroup && node.convertToNodes) {
                  hasGroups = true
                  const nodes = node.convertToNodes()

                  requestAnimationFrame(() => {
                    this.selectItems(nodes, true)

                    if (!this.selectedItems.size)
                      throw new Error('Convert to Subgraph: Nothing selected.')
                    this._graph.convertToSubgraph(this.selectedItems)
                  })
                  return
                }
              }

              // If no groups were found, continue normally
              if (!hasGroups) {
                if (!this.selectedItems.size)
                  throw new Error('Convert to Subgraph: Nothing selected.')
                this._graph.convertToSubgraph(this.selectedItems)
              }
            } else {
              throw new Error('Convert to Subgraph: Nothing selected.')
            }
          }
        },
        {
          content: 'Properties',
          has_submenu: true,
          callback: LGraphCanvas.onShowMenuNodeProperties
        },
        {
          content: 'Properties Panel',
          callback: function (
            _item: Positionable,
            _options: IContextMenuOptions | undefined,
            _e: MouseEvent | undefined,
            _menu: ContextMenu<unknown> | undefined,
            node: LGraphNode
          ) {
            LGraphCanvas.active_canvas.showShowNodePanel(node)
          }
        },
        null,
        {
          content: 'Title',
          callback: LGraphCanvas.onShowPropertyEditor
        },
        {
          content: 'Mode',
          has_submenu: true,
          callback: LGraphCanvas.onMenuNodeMode
        }
      ]
      if (node.resizable !== false) {
        options.push({
          content: 'Resize',
          callback: LGraphCanvas.onMenuResizeNode
        })
      }
      if (node.collapsible) {
        options.push({
          content: node.collapsed ? 'Expand' : 'Collapse',
          callback: LGraphCanvas.onMenuNodeCollapse
        })
      }
      if (node.hasAdvancedWidgets()) {
        options.push({
          content: node.showAdvanced ? 'Hide Advanced' : 'Show Advanced',
          callback: LGraphCanvas.onMenuToggleAdvanced
        })
      }
      options.push(
        {
          content: node.pinned ? 'Unpin' : 'Pin',
          callback: () => {
            for (const i in this.selected_nodes) {
              const node = this.selected_nodes[i]
              node.pin()
            }
            this.setDirty(true, true)
          }
        },
        {
          content: 'Colors',
          has_submenu: true,
          callback: LGraphCanvas.onMenuNodeColors
        },
        {
          content: 'Shapes',
          has_submenu: true,
          callback: LGraphCanvas.onMenuNodeShapes
        },
        null
      )
    }

    const extra = node.getExtraMenuOptions?.(this, options)
    if (Array.isArray(extra) && extra.length > 0) {
      extra.push(null)
      options = extra.concat(options)
    }

    if (node.clonable !== false) {
      options.push({
        content: 'Clone',
        callback: LGraphCanvas.onMenuNodeClone
      })
    }

    if (Object.keys(this.selected_nodes).length > 1) {
      options.push(
        {
          content: 'Align Selected To',
          has_submenu: true,
          callback: LGraphCanvas.onNodeAlign
        },
        {
          content: 'Distribute Nodes',
          has_submenu: true,
          callback: LGraphCanvas.createDistributeMenu
        }
      )
    }

    options.push(null, {
      content: 'Remove',
      disabled: !(node.removable !== false && !node.block_delete),
      callback: LGraphCanvas.onMenuNodeRemove
    })

    node.graph?.onGetNodeMenuOptions?.(options, node)

    return options
  }

  /** @deprecated */
  getGroupMenuOptions(group: LGraphGroup) {
    console.warn(
      'LGraphCanvas.getGroupMenuOptions is deprecated, use LGraphGroup.getMenuOptions instead'
    )
    return group.getMenuOptions()
  }

  processContextMenu(
    node: LGraphNode | undefined,
    event: CanvasPointerEvent
  ): void {
    // TODO: Remove type kludge
    let menu_info: (IContextMenuValue | string | null)[]
    const options: IContextMenuOptions = {
      event,
      callback: inner_option_clicked,
      extra: node
    }

    if (node) {
      options.title = node.displayType ?? node.type ?? undefined
      LGraphCanvas.active_node = node

      // check if mouse is in input
      const slot = node.getSlotInPosition(event.canvasX, event.canvasY)
      if (slot) {
        // on slot
        menu_info = []
        if (node.getSlotMenuOptions) {
          menu_info = node.getSlotMenuOptions(slot)
        } else {
          if (slot.output?.links?.length || slot.input?.link != null) {
            menu_info.push({ content: 'Disconnect Links', slot })
          }

          const _slot = slot.input || slot.output
          if (!_slot)
            throw new TypeError(
              'Both in put and output slots were null when processing context menu.'
            )

          if (!_slot.nameLocked && !('link' in _slot && _slot.widget)) {
            menu_info.push({ content: 'Rename Slot', slot })
          }

          if (_slot.removable) {
            menu_info.push(null)
            menu_info.push(
              _slot.locked
                ? 'Cannot remove'
                : { content: 'Remove Slot', slot, className: 'danger' }
            )
          }

          if (node.getExtraSlotMenuOptions) {
            menu_info.push(...node.getExtraSlotMenuOptions(slot))
          }
        }
        // @ts-expect-error Slot type can be number and has number checks
        options.title = (slot.input ? slot.input.type : slot.output.type) || '*'
        if (slot.input && slot.input.type == LiteGraph.ACTION)
          options.title = 'Action'

        if (slot.output && slot.output.type == LiteGraph.EVENT)
          options.title = 'Event'
      } else {
        // on node
        menu_info = this.getNodeMenuOptions(node)
      }
    } else {
      menu_info = this.getCanvasMenuOptions()
      if (!this.graph) throw new NullGraphError()

      // Check for reroutes
      if (this.links_render_mode !== LinkRenderType.HIDDEN_LINK) {
        // Try layout store first, fallback to old method
        const rerouteLayout = layoutStore.queryRerouteAtPoint({
          x: event.canvasX,
          y: event.canvasY
        })

        let reroute: Reroute | undefined
        if (rerouteLayout) {
          reroute = this.graph.getReroute(rerouteLayout.id)
        } else {
          reroute = this.graph.getRerouteOnPos(
            event.canvasX,
            event.canvasY,
            this._visibleReroutes
          )
        }
        if (reroute) {
          menu_info.unshift(
            {
              content: 'Delete Reroute',
              callback: () => {
                if (!this.graph) throw new NullGraphError()

                this.graph.removeReroute(reroute.id)
              }
            },
            null
          )
        }
      }

      const group = this.graph.getGroupOnPos(event.canvasX, event.canvasY)
      if (group) {
        // on group
        menu_info.push(null, {
          content: 'Edit Group',
          has_submenu: true,
          submenu: {
            title: 'Group',
            extra: group,
            options: group.getMenuOptions()
          }
        })
      }
    }

    // show menu
    if (!menu_info) return

    new LiteGraph.ContextMenu(menu_info, options)

    const createDialog = (options: IDialogOptions) =>
      this.createDialog(
        "<span class='name'>Name</span><input autofocus type='text'/><button>OK</button>",
        options
      )
    const setDirty = () => this.setDirty(true)

    function inner_option_clicked(
      v: IContextMenuValue<unknown>,
      options: IDialogOptions
    ) {
      if (!v) return

      if (v.content == 'Remove Slot') {
        if (!node?.graph) throw new NullGraphError()

        const info = v.slot
        if (!info)
          throw new TypeError(
            'Found-slot info was null when processing context menu.'
          )

        node.graph.beforeChange()
        if (info.input) {
          node.removeInput(info.slot)
        } else if (info.output) {
          node.removeOutput(info.slot)
        }
        node.graph.afterChange()
        return
      } else if (v.content == 'Disconnect Links') {
        if (!node?.graph) throw new NullGraphError()

        const info = v.slot
        if (!info)
          throw new TypeError(
            'Found-slot info was null when processing context menu.'
          )

        node.graph.beforeChange()
        if (info.output) {
          node.disconnectOutput(info.slot)
        } else if (info.input) {
          node.disconnectInput(info.slot, true)
        }
        node.graph.afterChange()
        return
      } else if (v.content == 'Rename Slot') {
        if (!node)
          throw new TypeError(
            '`node` was null when processing the context menu.'
          )

        const info = v.slot
        if (!info)
          throw new TypeError(
            'Found-slot info was null when processing context menu.'
          )

        const slot_info = info.input
          ? node.getInputInfo(info.slot)
          : node.getOutputInfo(info.slot)
        const dialog = createDialog(options)

        const input = dialog.querySelector('input')
        if (input && slot_info) {
          input.value = slot_info.label || ''
        }
        const inner = function () {
          if (!node.graph) throw new NullGraphError()

          node.graph.beforeChange()
          if (input?.value) {
            if (slot_info) {
              slot_info.label = input.value
            }
            setDirty()
          }
          dialog.close()
          node.graph.afterChange()
        }
        dialog.querySelector('button')?.addEventListener('click', inner)
        if (!input)
          throw new TypeError(
            'Input element was null when processing context menu.'
          )

        input.addEventListener('keydown', function (e) {
          dialog.is_modified = true
          if (e.key == 'Escape') {
            // ESC
            dialog.close()
          } else if (e.key == 'Enter') {
            // save
            inner()
          } else if ((e.target as Element).localName != 'textarea') {
            return
          }
          e.preventDefault()
          e.stopPropagation()
        })
        input.focus()
      }
    }
  }

  /**
   * Starts an animation to fit the view around the specified selection of nodes.
   * @param bounds The bounds to animate the view to, defined by a rectangle.
   */
  animateToBounds(bounds: ReadOnlyRect, options: AnimationOptions = {}) {
    const setDirty = () => this.setDirty(true, true)
    this.ds.animateToBounds(bounds, setDirty, options)
  }

  /**
   * Fits the view to the selected nodes with animation.
   * If nothing is selected, the view is fitted around all items in the graph.
   */
  fitViewToSelectionAnimated(options: AnimationOptions = {}) {
    const items = this.selectedItems.size
      ? Array.from(this.selectedItems)
      : this.positionableItems
    const bounds = createBounds(items)
    if (!bounds)
      throw new TypeError(
        'Attempted to fit to view but could not calculate bounds.'
      )

    const setDirty = () => this.setDirty(true, true)
    this.ds.animateToBounds(bounds, setDirty, options)
  }

  /**
   * Calculate new position with delta
   */
  private calculateNewPosition(
    node: LGraphNode,
    deltaX: number,
    deltaY: number
  ): { x: number; y: number } {
    return {
      x: node.pos[0] + deltaX,
      y: node.pos[1] + deltaY
    }
  }

  /**
   * Apply batched node position updates
   */
  private applyNodePositionUpdates(
    nodesToMove: Array<{ node: LGraphNode; newPos: { x: number; y: number } }>
  ): void {
    for (const { node, newPos } of nodesToMove) {
      // setPos automatically syncs to layout store
      node.setPos(newPos.x, newPos.y)
    }
  }

  /**
   * Collect all nodes that are children of groups in the selection
   */
  private collectNodesInGroups(items: Set<Positionable>): Set<LGraphNode> {
    const nodesInGroups = new Set<LGraphNode>()
    for (const item of items) {
      if (item instanceof LGraphGroup) {
        for (const child of item._children) {
          if (child instanceof LGraphNode) {
            nodesInGroups.add(child)
          }
        }
      }
    }
    return nodesInGroups
  }

  /**
   * Move group children (both nodes and non-nodes)
   */
  private moveGroupChildren(
    group: LGraphGroup,
    deltaX: number,
    deltaY: number,
    nodesToMove: Array<{ node: LGraphNode; newPos: { x: number; y: number } }>
  ): void {
    for (const child of group._children) {
      if (child instanceof LGraphNode) {
        const node = child as LGraphNode
        nodesToMove.push({
          node,
          newPos: this.calculateNewPosition(node, deltaX, deltaY)
        })
      } else if (!(child instanceof LGraphGroup)) {
        // Non-node, non-group children (reroutes, etc.)
        // Skip groups here - they're already in allItems and will be
        // processed in the main loop of moveChildNodesInGroupVueMode
        child.move(deltaX, deltaY, true)
      }
    }
  }

  moveChildNodesInGroupVueMode(
    allItems: Set<Positionable>,
    deltaX: number,
    deltaY: number
  ) {
    const nodesInMovingGroups = this.collectNodesInGroups(allItems)
    const nodesToMove: NewNodePosition[] = []

    // First, collect all the moves we need to make
    for (const item of allItems) {
      const isNode = item instanceof LGraphNode
      if (isNode) {
        const node = item as LGraphNode
        if (nodesInMovingGroups.has(node)) {
          continue
        }
        nodesToMove.push({
          node,
          newPos: this.calculateNewPosition(node, deltaX, deltaY)
        })
      } else if (item instanceof LGraphGroup) {
        item.move(deltaX, deltaY, true)
        this.moveGroupChildren(item, deltaX, deltaY, nodesToMove)
      } else {
        // Other items (reroutes, etc.)
        item.move(deltaX, deltaY, true)
      }
    }

    // Now apply all the node moves at once
    this.applyNodePositionUpdates(nodesToMove)
  }

  repositionNodesVueMode(nodesToReposition: NewNodePosition[]) {
    this.applyNodePositionUpdates(nodesToReposition)
  }

  /**
   * Custom JSON serialization to prevent circular reference errors.
   * LGraphCanvas should not be serialized directly - serialize the graph instead.
   */
  toJSON(): { ds: { scale: number; offset: [number, number] } } {
    return {
      ds: {
        scale: this.ds.scale,
        offset: [...this.ds.offset] as [number, number]
      }
    }
  }
}

function patchLinkNodeIds(
  links: { origin_id: NodeId; target_id: NodeId }[] | undefined,
  remappedIds: Map<NodeId, NodeId>
) {
  if (!links?.length) return

  for (const link of links) {
    const newOriginId = remappedIds.get(link.origin_id)
    if (newOriginId !== undefined) link.origin_id = newOriginId

    const newTargetId = remappedIds.get(link.target_id)
    if (newTargetId !== undefined) link.target_id = newTargetId
  }
}

function remapNodeId(
  nodeId: string,
  remappedIds: Map<NodeId, NodeId>
): NodeId | undefined {
  const directMatch = remappedIds.get(nodeId)
  if (directMatch !== undefined) return directMatch
  if (!/^-?\d+$/.test(nodeId)) return undefined

  const numericId = Number(nodeId)
  if (!Number.isSafeInteger(numericId)) return undefined

  return remappedIds.get(numericId)
}

function remapProxyWidgets(
  info: ISerialisedNode,
  remappedIds: Map<NodeId, NodeId> | undefined
) {
  if (!remappedIds || remappedIds.size === 0) return

  const proxyWidgets = info.properties?.proxyWidgets
  if (!Array.isArray(proxyWidgets)) return

  for (const entry of proxyWidgets) {
    if (!Array.isArray(entry)) continue

    const [nodeId] = entry
    if (typeof nodeId !== 'string' || nodeId === '-1') continue

    const remappedNodeId = remapNodeId(nodeId, remappedIds)
    if (remappedNodeId !== undefined) entry[0] = String(remappedNodeId)
  }
}

/**
 * Remaps pasted subgraph interior node IDs that would collide with existing
 * node IDs in the root graph. Also patches subgraph link node IDs and
 * SubgraphNode `properties.proxyWidgets` references so promoted widget
 * associations stay aligned with remapped interior IDs.
 */
export function remapClipboardSubgraphNodeIds(
  parsed: ClipboardItems,
  rootGraph: LGraph
): void {
  const usedNodeIds = new Set<number>()
  forEachNode(rootGraph, (node) => {
    if (typeof node.id !== 'number') return
    usedNodeIds.add(node.id)
    if (rootGraph.state.lastNodeId < node.id)
      rootGraph.state.lastNodeId = node.id
  })

  function nextUniqueNodeId() {
    while (usedNodeIds.has(++rootGraph.state.lastNodeId));
    const nextId = rootGraph.state.lastNodeId
    usedNodeIds.add(nextId)
    return nextId
  }

  const subgraphNodeIdMap = new Map<UUID, Map<NodeId, NodeId>>()
  for (const subgraphInfo of parsed.subgraphs ?? []) {
    const remappedIds = new Map<NodeId, NodeId>()
    const interiorNodes = subgraphInfo.nodes ?? []

    for (const nodeInfo of interiorNodes) {
      if (typeof nodeInfo.id !== 'number') continue

      if (usedNodeIds.has(nodeInfo.id)) {
        const oldId = nodeInfo.id
        const newId = nextUniqueNodeId()
        remappedIds.set(oldId, newId)
        nodeInfo.id = newId
        continue
      }

      usedNodeIds.add(nodeInfo.id)
      if (rootGraph.state.lastNodeId < nodeInfo.id)
        rootGraph.state.lastNodeId = nodeInfo.id
    }

    if (remappedIds.size > 0) {
      patchLinkNodeIds(subgraphInfo.links, remappedIds)
      subgraphNodeIdMap.set(subgraphInfo.id, remappedIds)
    }
  }

  const allNodeInfo: ISerialisedNode[] = [
    parsed.nodes ? [parsed.nodes] : [],
    parsed.subgraphs ? parsed.subgraphs.map((s) => s.nodes ?? []) : []
  ].flat(2)

  for (const nodeInfo of allNodeInfo) {
    if (typeof nodeInfo.type !== 'string') continue
    remapProxyWidgets(nodeInfo, subgraphNodeIdMap.get(nodeInfo.type))
  }
}
