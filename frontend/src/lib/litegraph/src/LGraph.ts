import { toString } from 'es-toolkit/compat'

import {
  SUBGRAPH_INPUT_ID,
  SUBGRAPH_OUTPUT_ID
} from '@/lib/litegraph/src/constants'
import { isNodeBindable } from '@/lib/litegraph/src/utils/type'
import type { UUID } from '@/lib/litegraph/src/utils/uuid'
import { createUuidv4, zeroUuid } from '@/lib/litegraph/src/utils/uuid'
import { useLayoutMutations } from '@/renderer/core/layout/operations/layoutMutations'
import { LayoutSource } from '@/renderer/core/layout/types'
import { usePromotionStore } from '@/stores/promotionStore'
import { useWidgetValueStore } from '@/stores/widgetValueStore'
import { forEachNode } from '@/utils/graphTraversalUtil'

import type { DragAndScaleState } from './DragAndScale'
import { LGraphCanvas } from './LGraphCanvas'
import { LGraphGroup } from './LGraphGroup'
import { LGraphNode } from './LGraphNode'
import type { NodeId } from './LGraphNode'
import { LLink } from './LLink'
import type { LinkId, SerialisedLLinkArray } from './LLink'
import { MapProxyHandler } from './MapProxyHandler'
import { Reroute } from './Reroute'
import type { RerouteId } from './Reroute'
import { CustomEventTarget } from './infrastructure/CustomEventTarget'
import type { LGraphEventMap } from './infrastructure/LGraphEventMap'
import type { SubgraphEventMap } from './infrastructure/SubgraphEventMap'
import type {
  DefaultConnectionColors,
  Dictionary,
  HasBoundingRect,
  IContextMenuValue,
  INodeInputSlot,
  INodeOutputSlot,
  LinkNetwork,
  LinkSegment,
  MethodNames,
  OptionalProps,
  Point,
  Positionable,
  Size
} from './interfaces'
import { LiteGraph, SubgraphNode } from './litegraph'
import {
  alignOutsideContainer,
  alignToContainer,
  createBounds
} from './measure'
import { SubgraphInput } from './subgraph/SubgraphInput'
import { SubgraphInputNode } from './subgraph/SubgraphInputNode'
import { SubgraphOutput } from './subgraph/SubgraphOutput'
import { SubgraphOutputNode } from './subgraph/SubgraphOutputNode'
import {
  findUsedSubgraphIds,
  getBoundaryLinks,
  groupResolvedByOutput,
  mapSubgraphInputsAndLinks,
  mapSubgraphOutputsAndLinks,
  multiClone,
  splitPositionables
} from './subgraph/subgraphUtils'
import { Alignment, LGraphEventMode } from './types/globalEnums'
import type {
  LGraphTriggerAction,
  LGraphTriggerEvent,
  LGraphTriggerHandler,
  LGraphTriggerParam
} from './types/graphTriggers'
import type {
  ExportedSubgraph,
  ExposedWidget,
  ISerialisedGraph,
  ISerialisedNode,
  Serialisable,
  SerialisableGraph,
  SerialisableReroute
} from './types/serialisation'
import { getAllNestedItems } from './utils/collections'

export type {
  LGraphTriggerAction,
  LGraphTriggerParam
} from './types/graphTriggers'

export type RendererType = 'LG' | 'Vue'

export interface LGraphState {
  lastGroupId: number
  lastNodeId: number
  lastLinkId: number
  lastRerouteId: number
}

type ParamsArray<T, K extends MethodNames<T>> = Parameters<
  Extract<T[K], (...args: never[]) => unknown>
>[1] extends undefined
  ?
      | Parameters<Extract<T[K], (...args: never[]) => unknown>>
      | Parameters<Extract<T[K], (...args: never[]) => unknown>>[0]
  : Parameters<Extract<T[K], (...args: never[]) => unknown>>

/** Configuration used by {@link LGraph} `config`. */
export interface LGraphConfig {
  /** @deprecated Legacy config - unused */
  align_to_grid?: boolean
  links_ontop?: boolean
}

/** Options for {@link LGraph.add} method. */
export interface GraphAddOptions {
  /** If true, skip recomputing execution order after adding the node. */
  skipComputeOrder?: boolean
  /** If true, the node will be semi-transparent and follow the cursor until placed or cancelled. */
  ghost?: boolean
  /** Mouse event for ghost placement. Used to position node under cursor. */
  dragEvent?: MouseEvent
}

export interface GroupNodeConfigEntry {
  input?: Record<string, { name?: string; visible?: boolean }>
  output?: Record<number, { name?: string; visible?: boolean }>
}

export interface GroupNodeWorkflowData {
  external: (number | string)[][]
  links: SerialisedLLinkArray[]
  nodes: {
    index?: number
    type?: string
    title?: string
    inputs?: unknown[]
    outputs?: unknown[]
    widgets_values?: unknown[]
  }[]
  config?: Record<number, GroupNodeConfigEntry>
}

export interface LGraphExtra extends Dictionary<unknown> {
  reroutes?: SerialisableReroute[]
  linkExtensions?: { id: number; parentId: number | undefined }[]
  ds?: DragAndScaleState
  workflowRendererVersion?: RendererType
  groupNodes?: Record<string, GroupNodeWorkflowData>
}

export interface BaseLGraph {
  /** The root graph. */
  readonly rootGraph: LGraph
}

/**
 * LGraph is the class that contain a full graph. We instantiate one and add nodes to it, and then we can run the execution loop.
 * supported callbacks:
 * + onNodeAdded: when a new node is added to the graph
 * + onNodeRemoved: when a node inside this graph is removed
 */
export class LGraph
  implements LinkNetwork, BaseLGraph, Serialisable<SerialisableGraph>
{
  static serialisedSchemaVersion = 1 as const

  static STATUS_STOPPED = 1
  static STATUS_RUNNING = 2

  /** List of LGraph properties that are manually handled by {@link LGraph.configure}. */
  static readonly ConfigureProperties = new Set([
    'nodes',
    'groups',
    'links',
    'state',
    'reroutes',
    'floatingLinks',
    'id',
    'subgraphs',
    'definitions',
    'inputs',
    'outputs',
    'widgets',
    'inputNode',
    'outputNode',
    'extra'
  ])

  id: UUID = zeroUuid
  revision: number = 0

  _version: number = -1
  /** The backing store for links.  Keys are wrapped in String() */
  _links: Map<LinkId, LLink> = new Map()
  /**
   * Indexed property access is deprecated.
   * Backwards compatibility with a Proxy has been added, but will eventually be removed.
   *
   * Use {@link Map} methods:
   * ```
   * const linkId = 123
   * const link = graph.links.get(linkId)
   * // Deprecated: const link = graph.links[linkId]
   * ```
   */
  links: Map<LinkId, LLink> & Record<LinkId, LLink>
  list_of_graphcanvas: LGraphCanvas[] | null
  status: number = LGraph.STATUS_STOPPED

  private _state: LGraphState = {
    lastGroupId: 0,
    lastNodeId: 0,
    lastLinkId: 0,
    lastRerouteId: 0
  }

  get state(): LGraphState {
    return this._state
  }

  set state(value: LGraphState) {
    this._state = value
  }

  readonly events = new CustomEventTarget<LGraphEventMap>()
  readonly _subgraphs: Map<UUID, Subgraph> = new Map()

  _nodes: (LGraphNode | SubgraphNode)[] = []
  _nodes_by_id: Record<NodeId, LGraphNode> = {}
  _nodes_in_order: LGraphNode[] = []
  _nodes_executable: LGraphNode[] | null = null
  _groups: LGraphGroup[] = []
  iteration: number = 0
  globaltime: number = 0
  /** @deprecated Unused */
  runningtime: number = 0
  fixedtime: number = 0
  fixedtime_lapse: number = 0.01
  elapsed_time: number = 0.01
  last_update_time: number = 0
  starttime: number = 0
  catch_errors: boolean = true
  execution_timer_id?: number | null
  errors_in_execution?: boolean
  /** @deprecated Unused */
  execution_time!: number
  _last_trigger_time?: number
  filter?: string
  /** Must contain serialisable values, e.g. primitive types */
  config: LGraphConfig = {}
  vars: Dictionary<unknown> = {}
  nodes_executing: boolean[] = []
  nodes_actioning: (string | boolean)[] = []
  nodes_executedAction: string[] = []
  extra: LGraphExtra = {}

  /** @deprecated Deserialising a workflow sets this unused property. */
  version?: number

  /** @returns Whether the graph has no items */
  get empty(): boolean {
    return this._nodes.length + this._groups.length + this.reroutes.size === 0
  }

  /** @returns All items on the canvas that can be selected */
  *positionableItems(): Generator<LGraphNode | LGraphGroup | Reroute> {
    for (const node of this._nodes) yield node
    for (const group of this._groups) yield group
    for (const reroute of this.reroutes.values()) yield reroute
    return
  }

  /** Internal only.  Not required for serialisation; calculated on deserialise. */
  private _lastFloatingLinkId: number = 0

  private readonly floatingLinksInternal: Map<LinkId, LLink> = new Map()
  get floatingLinks(): ReadonlyMap<LinkId, LLink> {
    return this.floatingLinksInternal
  }

  private readonly reroutesInternal = new Map<RerouteId, Reroute>()
  /** All reroutes in this graph. */
  public get reroutes(): Map<RerouteId, Reroute> {
    return this.reroutesInternal
  }

  get rootGraph(): LGraph {
    return this
  }

  get isRootGraph(): boolean {
    return this.rootGraph === this
  }

  /** @deprecated See {@link state}.{@link LGraphState.lastNodeId lastNodeId} */
  get last_node_id() {
    return this.state.lastNodeId
  }

  set last_node_id(value) {
    this.state.lastNodeId = value
  }

  /** @deprecated See {@link state}.{@link LGraphState.lastLinkId lastLinkId} */
  get last_link_id() {
    return this.state.lastLinkId
  }

  set last_link_id(value) {
    this.state.lastLinkId = value
  }

  onAfterStep?(): void
  onBeforeStep?(): void
  onPlayEvent?(): void
  onStopEvent?(): void
  onAfterExecute?(): void
  onExecuteStep?(): void
  onNodeAdded?(node: LGraphNode): void
  onNodeRemoved?(node: LGraphNode): void
  onTrigger?: LGraphTriggerHandler
  onBeforeChange?(graph: LGraph, info?: LGraphNode): void
  onAfterChange?(graph: LGraph, info?: LGraphNode | null): void
  onConnectionChange?(node: LGraphNode): void
  on_change?(graph: LGraph): void
  onSerialize?(data: ISerialisedGraph | SerialisableGraph): void
  onConfigure?(data: ISerialisedGraph | SerialisableGraph): void
  onGetNodeMenuOptions?(
    options: (IContextMenuValue<unknown> | null)[],
    node: LGraphNode
  ): void

  // @ts-expect-error - Private property type needs fixing
  private _input_nodes?: LGraphNode[]

  /**
   * See {@link LGraph}
   * @param o data from previous serialization [optional]
   */
  constructor(o?: ISerialisedGraph | SerialisableGraph) {
    /** @see MapProxyHandler */
    const links = this._links
    MapProxyHandler.bindAllMethods(links)
    const handler = new MapProxyHandler<LLink>()
    this.links = new Proxy(links, handler) as Map<LinkId, LLink> &
      Record<LinkId, LLink>

    this.list_of_graphcanvas = null
    this.clear()

    if (o) this.configure(o)
  }

  /**
   * Removes all nodes from this graph
   */
  clear(): void {
    this.stop()
    this.status = LGraph.STATUS_STOPPED

    const graphId = this.id
    if (this.isRootGraph && graphId !== zeroUuid) {
      usePromotionStore().clearGraph(graphId)
      useWidgetValueStore().clearGraph(graphId)
    }

    this.id = zeroUuid
    this.revision = 0

    this.state = {
      lastGroupId: 0,
      lastNodeId: 0,
      lastLinkId: 0,
      lastRerouteId: 0
    }

    // used to detect changes
    this._version = -1
    this._subgraphs.clear()

    // safe clear
    if (this._nodes) {
      for (const _node of this._nodes) {
        _node.onRemoved?.()
        this.onNodeRemoved?.(_node)
      }
    }

    // nodes
    this._nodes = []
    this._nodes_by_id = {}
    // nodes sorted in execution order
    this._nodes_in_order = []
    // nodes that contain onExecute sorted in execution order
    this._nodes_executable = null

    this._links.clear()
    this.reroutes.clear()
    this.floatingLinksInternal.clear()

    this._lastFloatingLinkId = 0

    // other scene stuff
    this._groups = []

    // iterations
    this.iteration = 0

    // custom data
    this.config = {}
    this.vars = {}
    // to store custom data
    this.extra = {}

    // timing
    this.globaltime = 0
    this.runningtime = 0
    this.fixedtime = 0
    this.fixedtime_lapse = 0.01
    this.elapsed_time = 0.01
    this.last_update_time = 0
    this.starttime = 0

    this.catch_errors = true

    this.nodes_executing = []
    this.nodes_actioning = []
    this.nodes_executedAction = []

    // notify canvas to redraw
    this.change()

    this.canvasAction((c) => c.clear())
  }

  get subgraphs(): Map<UUID, Subgraph> {
    return this.rootGraph._subgraphs
  }

  get nodes() {
    return this._nodes
  }

  get groups() {
    return this._groups
  }

  /**
   * Attach Canvas to this graph
   */
  attachCanvas(canvas: LGraphCanvas): void {
    if (!(canvas instanceof LGraphCanvas)) {
      throw new TypeError('attachCanvas expects an LGraphCanvas instance')
    }

    this.primaryCanvas = canvas

    this.list_of_graphcanvas ??= []
    if (!this.list_of_graphcanvas.includes(canvas)) {
      this.list_of_graphcanvas.push(canvas)
    }

    if (canvas.graph === this) return

    canvas.graph?.detachCanvas(canvas)
    canvas.graph = this
    canvas.subgraph = undefined
  }

  /**
   * Detach Canvas from this graph
   */
  detachCanvas(canvas: LGraphCanvas): void {
    canvas.graph = null
    const canvases = this.list_of_graphcanvas
    if (canvases) {
      const pos = canvases.indexOf(canvas)
      if (pos !== -1) canvases.splice(pos, 1)
    }
  }

  /**
   * @deprecated Will be removed in 0.9
   * Starts running this graph every interval milliseconds.
   * @param interval amount of milliseconds between executions, if 0 then it renders to the monitor refresh rate
   */
  start(interval?: number): void {
    if (this.status == LGraph.STATUS_RUNNING) return
    this.status = LGraph.STATUS_RUNNING

    this.onPlayEvent?.()
    this.sendEventToAllNodes('onStart')

    // launch
    this.starttime = LiteGraph.getTime()
    this.last_update_time = this.starttime
    interval ||= 0

    // execute once per frame
    if (
      interval == 0 &&
      typeof window != 'undefined' &&
      window.requestAnimationFrame
    ) {
      const on_frame = () => {
        if (this.execution_timer_id != -1) return

        window.requestAnimationFrame(on_frame)
        this.onBeforeStep?.()
        this.runStep(1, !this.catch_errors)
        this.onAfterStep?.()
      }
      this.execution_timer_id = -1
      on_frame()
    } else {
      // execute every 'interval' ms
      // @ts-expect-error - Timer ID type mismatch needs fixing
      this.execution_timer_id = setInterval(() => {
        // execute
        this.onBeforeStep?.()
        this.runStep(1, !this.catch_errors)
        this.onAfterStep?.()
      }, interval)
    }
  }

  /**
   * @deprecated Will be removed in 0.9
   * Stops the execution loop of the graph
   */
  stop(): void {
    if (this.status == LGraph.STATUS_STOPPED) return

    this.status = LGraph.STATUS_STOPPED

    this.onStopEvent?.()

    if (this.execution_timer_id != null) {
      if (this.execution_timer_id != -1) {
        clearInterval(this.execution_timer_id)
      }
      this.execution_timer_id = null
    }

    this.sendEventToAllNodes('onStop')
  }

  /**
   * Run N steps (cycles) of the graph
   * @param num number of steps to run, default is 1
   * @param do_not_catch_errors [optional] if you want to try/catch errors
   * @param limit max number of nodes to execute (used to execute from start to a node)
   */
  runStep(num: number, do_not_catch_errors: boolean, limit?: number): void {
    num = num || 1

    const start = LiteGraph.getTime()
    this.globaltime = 0.001 * (start - this.starttime)

    const nodes = this._nodes_executable || this._nodes
    if (!nodes) return

    limit = limit || nodes.length

    if (do_not_catch_errors) {
      // iterations
      for (let i = 0; i < num; i++) {
        for (let j = 0; j < limit; ++j) {
          const node = nodes[j]
          // FIXME: Looks like copy/paste broken logic - checks for "on", executes "do"
          if (node.mode == LGraphEventMode.ALWAYS && node.onExecute) {
            // wrap node.onExecute();
            node.doExecute?.()
          }
        }

        this.fixedtime += this.fixedtime_lapse
        this.onExecuteStep?.()
      }

      this.onAfterExecute?.()
    } else {
      try {
        // iterations
        for (let i = 0; i < num; i++) {
          for (let j = 0; j < limit; ++j) {
            const node = nodes[j]
            if (node.mode == LGraphEventMode.ALWAYS) {
              node.onExecute?.()
            }
          }

          this.fixedtime += this.fixedtime_lapse
          this.onExecuteStep?.()
        }

        this.onAfterExecute?.()
        this.errors_in_execution = false
      } catch (error) {
        this.errors_in_execution = true
        if (LiteGraph.throw_errors) throw error

        if (LiteGraph.debug) console.error('Error during execution:', error)
        this.stop()
      }
    }

    const now = LiteGraph.getTime()
    let elapsed = now - start
    if (elapsed == 0) elapsed = 1

    this.execution_time = 0.001 * elapsed
    this.globaltime += 0.001 * elapsed
    this.iteration += 1
    this.elapsed_time = (now - this.last_update_time) * 0.001
    this.last_update_time = now
    this.nodes_executing = []
    this.nodes_actioning = []
    this.nodes_executedAction = []
  }

  /**
   * Updates the graph execution order according to relevance of the nodes (nodes with only outputs have more relevance than
   * nodes with only inputs.
   */
  updateExecutionOrder(): void {
    this._nodes_in_order = this.computeExecutionOrder(false)
    this._nodes_executable = []
    for (const node of this._nodes_in_order) {
      if (node.onExecute) {
        this._nodes_executable.push(node)
      }
    }
  }

  // This is more internal, it computes the executable nodes in order and returns it
  computeExecutionOrder(
    only_onExecute: boolean,
    set_level?: boolean
  ): LGraphNode[] {
    const L: LGraphNode[] = []
    const S: LGraphNode[] = []
    const M: Dictionary<LGraphNode> = {}
    // to avoid repeating links
    const visited_links: Record<NodeId, boolean> = {}
    const remaining_links: Record<NodeId, number> = {}

    // search for the nodes without inputs (starting nodes)
    for (const node of this._nodes) {
      if (only_onExecute && !node.onExecute) {
        continue
      }

      // add to pending nodes
      M[node.id] = node

      // num of input connections
      let num = 0
      if (node.inputs) {
        for (const input of node.inputs) {
          if (input?.link != null) {
            num += 1
          }
        }
      }

      if (num == 0) {
        // is a starting node
        S.push(node)
        if (set_level) node._level = 1
      } else {
        // num of input links
        if (set_level) node._level = 0
        remaining_links[node.id] = num
      }
    }

    while (true) {
      // get an starting node
      const node = S.shift()
      if (node === undefined) break

      // add to ordered list
      L.push(node)
      // remove from the pending nodes
      delete M[node.id]

      if (!node.outputs) continue

      // for every output
      for (const output of node.outputs) {
        // not connected
        // TODO: Confirm functionality, clean condition
        if (output?.links == null || output.links.length == 0) continue

        // for every connection
        for (const link_id of output.links) {
          const link = this._links.get(link_id)
          if (!link) continue

          // already visited link (ignore it)
          if (visited_links[link.id]) continue

          const target_node = this.getNodeById(link.target_id)
          if (target_node == null) {
            visited_links[link.id] = true
            continue
          }

          if (set_level) {
            node._level ??= 0
            if (!target_node._level || target_node._level <= node._level) {
              target_node._level = node._level + 1
            }
          }

          // mark as visited
          visited_links[link.id] = true
          // reduce the number of links remaining
          remaining_links[target_node.id] -= 1

          // if no more links, then add to starters array
          if (remaining_links[target_node.id] == 0) S.push(target_node)
        }
      }
    }

    // the remaining ones (loops)
    for (const i in M) {
      L.push(M[i])
    }

    if (L.length != this._nodes.length && LiteGraph.debug)
      console.warn('something went wrong, nodes missing')

    /** Ensure type is set */
    type OrderedLGraphNode = LGraphNode & { order: number }

    /** Sets the order property of each provided node to its index in {@link nodes}. */
    function setOrder(
      nodes: LGraphNode[]
    ): asserts nodes is OrderedLGraphNode[] {
      const l = nodes.length
      for (let i = 0; i < l; ++i) {
        nodes[i].order = i
      }
    }

    // save order number in the node
    setOrder(L)

    // sort now by priority
    L.sort(function (A, B) {
      // @ts-expect-error ctor props
      const Ap = A.constructor.priority || A.priority || 0
      // @ts-expect-error ctor props
      const Bp = B.constructor.priority || B.priority || 0
      // if same priority, sort by order

      return Ap == Bp ? A.order - B.order : Ap - Bp
    })

    // save order number in the node, again...
    setOrder(L)

    return L
  }

  /**
   * Positions every node in a more readable manner
   */
  arrange(margin?: number, layout?: string): void {
    margin = margin || 100

    const nodes = this.computeExecutionOrder(false, true)
    const columns: LGraphNode[][] = []
    for (const node of nodes) {
      const col = node._level || 1
      columns[col] ||= []
      columns[col].push(node)
    }

    let x = margin

    for (const column of columns) {
      if (!column) continue

      let max_size = 100
      let y = margin + LiteGraph.NODE_TITLE_HEIGHT
      for (const node of column) {
        node.setPos(
          layout == LiteGraph.VERTICAL_LAYOUT ? y : x,
          layout == LiteGraph.VERTICAL_LAYOUT ? x : y
        )
        const max_size_index = layout == LiteGraph.VERTICAL_LAYOUT ? 1 : 0
        if (node.size[max_size_index] > max_size) {
          max_size = node.size[max_size_index]
        }
        const node_size_index = layout == LiteGraph.VERTICAL_LAYOUT ? 0 : 1
        y += node.size[node_size_index] + margin + LiteGraph.NODE_TITLE_HEIGHT
      }
      x += max_size + margin
    }

    this.setDirtyCanvas(true, true)
  }

  /**
   * Returns the amount of time the graph has been running in milliseconds
   * @returns number of milliseconds the graph has been running
   */
  getTime(): number {
    return this.globaltime
  }

  /**
   * Returns the amount of time accumulated using the fixedtime_lapse var.
   * This is used in context where the time increments should be constant
   * @returns number of milliseconds the graph has been running
   */
  getFixedTime(): number {
    return this.fixedtime
  }

  /**
   * Returns the amount of time it took to compute the latest iteration.
   * Take into account that this number could be not correct
   * if the nodes are using graphical actions
   * @returns number of milliseconds it took the last cycle
   */
  getElapsedTime(): number {
    return this.elapsed_time
  }

  /**
   * @deprecated Will be removed in 0.9
   * Sends an event to all the nodes, useful to trigger stuff
   * @param eventname the name of the event (function to be called)
   * @param params parameters in array format
   */
  sendEventToAllNodes(
    eventname: string,
    params?: object | object[],
    mode?: LGraphEventMode
  ): void {
    mode = mode || LGraphEventMode.ALWAYS

    const nodes = this._nodes_in_order || this._nodes
    if (!nodes) return

    for (const node of nodes) {
      // @ts-expect-error deprecated
      if (!node[eventname] || node.mode != mode) continue
      if (params === undefined) {
        // @ts-expect-error deprecated
        node[eventname]()
      } else if (params && params.constructor === Array) {
        // @ts-expect-error deprecated
        // eslint-disable-next-line prefer-spread
        node[eventname].apply(node, params)
      } else {
        // @ts-expect-error deprecated
        node[eventname](params)
      }
    }
  }

  /**
   * Runs an action on every canvas registered to this graph.
   * @param action Action to run for every canvas
   */
  canvasAction(action: (canvas: LGraphCanvas) => void): void {
    const canvases = this.list_of_graphcanvas
    if (!canvases) return
    for (const canvas of canvases) action(canvas)
  }

  /** @deprecated See {@link LGraph.canvasAction} */
  sendActionToCanvas<T extends MethodNames<LGraphCanvas>>(
    action: T,
    params?: ParamsArray<LGraphCanvas, T>
  ): void {
    const { list_of_graphcanvas } = this
    if (!list_of_graphcanvas) return

    for (const c of list_of_graphcanvas) {
      const method = c[action]

      if (typeof method === 'function') {
        const args =
          params == null ? [] : Array.isArray(params) ? params : [params]
        ;(method as (...args: unknown[]) => unknown).apply(c, args)
      }
    }
  }

  /**
   * Adds a new node instance to this graph
   * @param node the instance of the node
   * @param options Additional options for adding the node
   */
  add(
    node: LGraphNode | LGraphGroup,
    options?: GraphAddOptions
  ): LGraphNode | null | undefined
  /**
   * Adds a new node instance to this graph
   * @param node the instance of the node
   * @param skipComputeOrder If true, skip recomputing execution order
   * @deprecated Use options object instead
   */
  add(
    node: LGraphNode | LGraphGroup | null,
    skipComputeOrder?: boolean
  ): LGraphNode | null | undefined
  add(
    node: LGraphNode | LGraphGroup,
    skipComputeOrderOrOptions?: boolean | GraphAddOptions
  ): LGraphNode | null | undefined {
    if (!node) return

    // Handle backwards compatibility: 2nd arg can be boolean or options
    const opts: GraphAddOptions =
      typeof skipComputeOrderOrOptions === 'object'
        ? skipComputeOrderOrOptions
        : { skipComputeOrder: skipComputeOrderOrOptions ?? false }
    const shouldSkipComputeOrder = opts.skipComputeOrder ?? false

    const { state } = this

    // Ensure created items are snapped
    if (LiteGraph.alwaysSnapToGrid) {
      const snapTo = this.getSnapToGridSize()
      if (snapTo) node.snapToGrid(snapTo)
    }

    // LEGACY: This was changed from constructor === LGraphGroup
    // groups
    if (node instanceof LGraphGroup) {
      // Assign group ID
      if (node.id == null || node.id === -1) node.id = ++state.lastGroupId
      if (node.id > state.lastGroupId) state.lastGroupId = node.id

      this._groups.push(node)
      this.setDirtyCanvas(true)
      this.change()
      node.graph = this
      this._version++
      return
    }

    // nodes
    if (node.id != -1 && this._nodes_by_id[node.id] != null) {
      console.warn(
        'LiteGraph: there is already a node with this ID, changing it'
      )
      node.id = LiteGraph.use_uuids ? LiteGraph.uuidv4() : ++state.lastNodeId
    }

    if (this._nodes.length >= LiteGraph.MAX_NUMBER_OF_NODES) {
      throw 'LiteGraph: max number of nodes in a graph reached'
    }

    // give him an id
    if (LiteGraph.use_uuids) {
      if (node.id == null || node.id == -1) node.id = LiteGraph.uuidv4()
    } else {
      if (node.id == null || node.id == -1) {
        node.id = ++state.lastNodeId
      } else if (typeof node.id === 'number' && state.lastNodeId < node.id) {
        state.lastNodeId = node.id
      }
    }

    // Set ghost flag before registration so VueNodeData picks it up
    if (opts.ghost) {
      node.flags.ghost = true
    }

    node.graph = this
    this._version++

    // Register all widgets with the WidgetValueStore now that node has a
    // valid ID and graph reference.
    if (node.widgets) {
      for (const widget of node.widgets) {
        if (isNodeBindable(widget)) widget.setNodeId(node.id)
      }
    }

    this._nodes.push(node)
    this._nodes_by_id[node.id] = node

    node.onAdded?.(this)

    if (this.config.align_to_grid) node.alignToGrid()

    if (!shouldSkipComputeOrder) this.updateExecutionOrder()

    this.onNodeAdded?.(node)

    this.setDirtyCanvas(true)
    this.change()

    if (opts.ghost) {
      this.canvasAction((c) => c.startGhostPlacement(node, opts.dragEvent))
    }

    if (node.isSubgraphNode?.()) {
      forEachNode(node.subgraph, (innerNode) => {
        if (innerNode.isSubgraphNode())
          this.subgraphs.set(innerNode.subgraph.id, innerNode.subgraph)
      })
    }

    // to chain actions
    return node
  }

  /**
   * Removes a node from the graph
   * @param node the instance of the node
   */
  remove(node: LGraphNode | LGraphGroup): void {
    // LEGACY: This was changed from constructor === LiteGraph.LGraphGroup
    if (node instanceof LGraphGroup) {
      this.canvasAction((c) => c.deselect(node))

      const index = this._groups.indexOf(node)
      if (index != -1) {
        this._groups.splice(index, 1)
      }
      node.graph = undefined
      this._version++
      this.setDirtyCanvas(true, true)
      this.change()
      return
    }

    // not found
    if (this._nodes_by_id[node.id] == null) {
      console.warn('LiteGraph: node not found', node)
      return
    }
    // cannot be removed
    if (node.ignore_remove) {
      console.warn('LiteGraph: node cannot be removed', node)
      return
    }

    // sure? - almost sure is wrong
    this.beforeChange()

    const { inputs, outputs } = node

    // disconnect inputs
    if (inputs) {
      for (const [i, slot] of inputs.entries()) {
        if (slot.link != null) node.disconnectInput(i, true)
      }
    }

    // disconnect outputs
    if (outputs) {
      for (const [i, slot] of outputs.entries()) {
        if (slot.links?.length) node.disconnectOutput(i)
      }
    }

    // Floating links
    for (const link of this.floatingLinks.values()) {
      if (link.origin_id === node.id || link.target_id === node.id) {
        this.removeFloatingLink(link)
      }
    }

    if (node.isSubgraphNode()) {
      forEachNode(node.subgraph, (innerNode) => {
        innerNode.onRemoved?.()
        innerNode.graph?.onNodeRemoved?.(innerNode)
        if (innerNode.isSubgraphNode())
          this.rootGraph.subgraphs.delete(innerNode.subgraph.id)
      })
      this.rootGraph.subgraphs.delete(node.subgraph.id)
    }

    // callback
    node.onRemoved?.()

    node.graph = null
    this._version++

    // remove from canvas render
    const { list_of_graphcanvas } = this
    if (list_of_graphcanvas) {
      for (const canvas of list_of_graphcanvas) {
        if (canvas.selected_nodes[node.id])
          delete canvas.selected_nodes[node.id]

        canvas.deselect(node)
      }
    }

    // remove from containers
    const pos = this._nodes.indexOf(node)
    if (pos != -1) this._nodes.splice(pos, 1)

    delete this._nodes_by_id[node.id]

    this.onNodeRemoved?.(node)

    // close panels
    this.canvasAction((c) => c.checkPanels())

    this.setDirtyCanvas(true, true)
    // sure? - almost sure is wrong
    this.afterChange()
    this.change()

    this.updateExecutionOrder()
  }

  /**
   * Returns a node by its id.
   */
  getNodeById(id: NodeId | null | undefined): LGraphNode | null {
    return id != null ? this._nodes_by_id[id] : null
  }

  /**
   * Returns a list of nodes that matches a class
   * @param classObject the class itself (not an string)
   * @returns a list with all the nodes of this type
   */
  // eslint-disable-next-line @typescript-eslint/no-unsafe-function-type
  findNodesByClass(classObject: Function, result?: LGraphNode[]): LGraphNode[] {
    result = result || []
    result.length = 0
    const { _nodes } = this
    for (const node of _nodes) {
      if (node.constructor === classObject) result.push(node)
    }
    return result
  }

  /**
   * Returns a list of nodes that matches a type
   * @param type the name of the node type
   * @returns a list with all the nodes of this type
   */
  findNodesByType(type: string, result: LGraphNode[]): LGraphNode[] {
    const matchType = type.toLowerCase()
    result = result || []
    result.length = 0
    const { _nodes } = this
    for (const node of _nodes) {
      if (node.type?.toLowerCase() == matchType) result.push(node)
    }
    return result
  }

  /**
   * Returns the first node that matches a name in its title
   * @param title the name of the node to search
   * @returns the node or null
   */
  findNodeByTitle(title: string): LGraphNode | null {
    const { _nodes } = this
    for (const node of _nodes) {
      if (node.title == title) return node
    }
    return null
  }

  /**
   * Returns a list of nodes that matches a name
   * @param title the name of the node to search
   * @returns a list with all the nodes with this name
   */
  findNodesByTitle(title: string): LGraphNode[] {
    const result: LGraphNode[] = []
    const { _nodes } = this
    for (const node of _nodes) {
      if (node.title == title) result.push(node)
    }
    return result
  }

  /**
   * Returns the top-most node in this position of the canvas
   * @param x the x coordinate in canvas space
   * @param y the y coordinate in canvas space
   * @param nodeList a list with all the nodes to search from, by default is all the nodes in the graph
   * @returns the node at this position or null
   */
  getNodeOnPos(
    x: number,
    y: number,
    nodeList?: LGraphNode[]
  ): LGraphNode | null {
    const nodes = nodeList || this._nodes
    let i = nodes.length
    while (--i >= 0) {
      const node = nodes[i]
      if (node.isPointInside(x, y)) return node
    }
    return null
  }

  /**
   * Returns the top-most group in that position
   * @param x The x coordinate in canvas space
   * @param y The y coordinate in canvas space
   * @returns The group or null
   */
  getGroupOnPos(x: number, y: number): LGraphGroup | undefined {
    // Iterate backwards through groups to find top-most
    for (let i = this._groups.length - 1; i >= 0; i--) {
      const group = this._groups[i]
      if (group.isPointInside(x, y)) {
        return group
      }
    }
    return undefined
  }

  /**
   * Returns the top-most group with a titlebar in the provided position.
   * @param x The x coordinate in canvas space
   * @param y The y coordinate in canvas space
   * @returns The group or null
   */
  getGroupTitlebarOnPos(x: number, y: number): LGraphGroup | undefined {
    // Iterate backwards through groups to find top-most
    for (let i = this._groups.length - 1; i >= 0; i--) {
      const group = this._groups[i]
      if (group.isPointInTitlebar(x, y)) {
        return group
      }
    }
    return undefined
  }

  /**
   * Finds a reroute a the given graph point
   * @param x X co-ordinate in graph space
   * @param y Y co-ordinate in graph space
   * @returns The first reroute under the given co-ordinates, or undefined
   */
  getRerouteOnPos(
    x: number,
    y: number,
    reroutes?: Iterable<Reroute>
  ): Reroute | undefined {
    for (const reroute of reroutes ?? this.reroutes.values()) {
      if (reroute.containsPoint([x, y])) return reroute
    }
  }

  /**
   * Snaps the provided items to a grid.
   *
   * Item positions are rounded to the nearest multiple of {@link LiteGraph.CANVAS_GRID_SIZE}.
   *
   * When {@link LiteGraph.alwaysSnapToGrid} is enabled
   * and the grid size is falsy, a default of 1 is used.
   * @param items The items to be snapped to the grid
   * @todo Currently only snaps nodes.
   */
  snapToGrid(items: Set<Positionable>): void {
    const snapTo = this.getSnapToGridSize()
    if (!snapTo) return

    for (const item of getAllNestedItems(items)) {
      if (!item.pinned) item.snapToGrid(snapTo)
    }
  }

  /**
   * Finds the size of the grid that items should be snapped to when moved.
   * @returns The size of the grid that items should be snapped to
   */
  getSnapToGridSize(): number {
    // Default to 1 when always snapping
    return LiteGraph.alwaysSnapToGrid
      ? LiteGraph.CANVAS_GRID_SIZE || 1
      : LiteGraph.CANVAS_GRID_SIZE
  }

  /**
   * @deprecated Will be removed in 0.9
   * Checks that the node type matches the node type registered,
   * used when replacing a nodetype by a newer version during execution
   * this replaces the ones using the old version with the new version
   */
  checkNodeTypes() {
    const { _nodes } = this
    for (const [i, node] of _nodes.entries()) {
      const ctor = LiteGraph.registered_node_types[node.type]
      if (node.constructor == ctor) continue

      console.warn('node being replaced by newer version:', node.type)
      const newnode = LiteGraph.createNode(node.type)
      if (!newnode) continue
      _nodes[i] = newnode
      newnode.configure(node.serialize())
      newnode.graph = this
      this._nodes_by_id[newnode.id] = newnode

      if (node.inputs) newnode.inputs = [...node.inputs]
      if (node.outputs) newnode.outputs = [...node.outputs]
    }
    this.updateExecutionOrder()
  }

  // ********** GLOBALS *****************
  trigger<A extends LGraphTriggerAction>(
    action: A,
    param: LGraphTriggerParam<A>
  ): void
  trigger(action: string, param: unknown): void
  trigger(action: string, param: unknown) {
    // Convert to discriminated union format for typed handlers
    const validEventTypes = new Set([
      'node:slot-links:changed',
      'node:slot-errors:changed',
      'node:property:changed'
    ])

    if (validEventTypes.has(action) && param && typeof param === 'object') {
      this.onTrigger?.({ type: action, ...param } as LGraphTriggerEvent)
    }
    // Don't handle unknown events - just ignore them
  }

  /** @todo Clean up - never implemented. */
  triggerInput(name: string, value: unknown): void {
    const nodes = this.findNodesByTitle(name)
    for (const node of nodes) {
      // @ts-expect-error - onTrigger method may not exist on all node types
      node.onTrigger(value)
    }
  }

  /** @todo Clean up - never implemented. */
  setCallback(name: string, func?: () => void): void {
    const nodes = this.findNodesByTitle(name)
    for (const node of nodes) {
      // @ts-expect-error - setTrigger method may not exist on all node types
      node.setTrigger(func)
    }
  }

  // used for undo, called before any change is made to the graph
  beforeChange(info?: LGraphNode): void {
    this.onBeforeChange?.(this, info)
    this.canvasAction((c) => c.onBeforeChange?.(this))
  }

  // used to resend actions, called after any change is made to the graph
  afterChange(info?: LGraphNode | null): void {
    this.onAfterChange?.(this, info)
    this.canvasAction((c) => c.onAfterChange?.(this))
  }

  /**
   * clears the triggered slot animation in all links (stop visual animation)
   */
  clearTriggeredSlots(): void {
    for (const link_info of this._links.values()) {
      if (!link_info) continue

      if (link_info._last_time) link_info._last_time = 0
    }
  }

  /* Called when something visually changed (not the graph!) */
  change(): void {
    this.canvasAction((c) => c.setDirty(true, true))
    this.on_change?.(this)
  }

  setDirtyCanvas(fg: boolean, bg?: boolean): void {
    this.canvasAction((c) => c.setDirty(fg, bg))
  }

  addFloatingLink(link: LLink): LLink {
    if (link.id === -1) {
      link.id = ++this._lastFloatingLinkId
    }
    this.floatingLinksInternal.set(link.id, link)

    const slot =
      link.target_id !== -1
        ? this.getNodeById(link.target_id)?.inputs?.[link.target_slot]
        : this.getNodeById(link.origin_id)?.outputs?.[link.origin_slot]
    if (slot) {
      slot._floatingLinks ??= new Set()
      slot._floatingLinks.add(link)
    } else {
      console.warn(
        `Adding invalid floating link: target/slot: [${link.target_id}/${link.target_slot}] origin/slot: [${link.origin_id}/${link.origin_slot}]`
      )
    }

    const reroutes = LLink.getReroutes(this, link)
    for (const reroute of reroutes) {
      reroute.floatingLinkIds.add(link.id)
    }
    return link
  }

  removeFloatingLink(link: LLink): void {
    this.floatingLinksInternal.delete(link.id)

    const slot =
      link.target_id !== -1
        ? this.getNodeById(link.target_id)?.inputs?.[link.target_slot]
        : this.getNodeById(link.origin_id)?.outputs?.[link.origin_slot]
    if (slot) {
      slot._floatingLinks?.delete(link)
    }

    const reroutes = LLink.getReroutes(this, link)
    for (const reroute of reroutes) {
      reroute.floatingLinkIds.delete(link.id)
      if (reroute.floatingLinkIds.size === 0) {
        delete reroute.floating
      }

      if (reroute.totalLinks === 0) this.removeReroute(reroute.id)
    }
  }

  /**
   * Finds the link with the provided ID.
   * @param id ID of link to find
   * @returns The link with the provided {@link id}, otherwise `undefined`. Always returns `undefined` if `id` is nullish.
   */
  getLink(id: null | undefined): undefined
  getLink(id: LinkId | null | undefined): LLink | undefined
  getLink(id: LinkId | null | undefined): LLink | undefined {
    return id == null ? undefined : this._links.get(id)
  }

  /**
   * Finds the reroute with the provided ID.
   * @param id ID of reroute to find
   * @returns The reroute with the provided {@link id}, otherwise `undefined`. Always returns `undefined` if `id` is nullish.
   */
  getReroute(id: null | undefined): undefined
  getReroute(id: RerouteId | null | undefined): Reroute | undefined
  getReroute(id: RerouteId | null | undefined): Reroute | undefined {
    return id == null ? undefined : this.reroutes.get(id)
  }

  /**
   * Configures a reroute on the graph where ID is already known (probably deserialisation).
   * Creates the object if it does not exist.
   * @param serialisedReroute See {@link SerialisableReroute}
   */
  setReroute({
    id,
    parentId,
    pos,
    linkIds,
    floating
  }: OptionalProps<SerialisableReroute, 'id'>): Reroute {
    id ??= ++this.state.lastRerouteId
    if (id > this.state.lastRerouteId) this.state.lastRerouteId = id

    const reroute = this.reroutes.get(id) ?? new Reroute(id, this)
    reroute.update(parentId, pos, linkIds, floating)
    this.reroutes.set(id, reroute)
    return reroute
  }

  /**
   * Creates a new reroute and adds it to the graph.
   * @param pos Position in graph space
   * @param before The existing link segment (reroute, link) that will be after this reroute,
   * going from the node output to input.
   * @returns The newly created reroute - typically ignored.
   */
  createReroute(pos: Point, before: LinkSegment): Reroute {
    const layoutMutations = useLayoutMutations()
    const rerouteId = ++this.state.lastRerouteId
    const linkIds = before instanceof Reroute ? before.linkIds : [before.id]
    const floatingLinkIds =
      before instanceof Reroute ? before.floatingLinkIds : [before.id]
    const reroute = new Reroute(
      rerouteId,
      this,
      pos,
      before.parentId,
      linkIds,
      floatingLinkIds
    )
    this.reroutes.set(rerouteId, reroute)

    // Register reroute in Layout Store for spatial tracking
    layoutMutations.setSource(LayoutSource.Canvas)
    layoutMutations.createReroute(
      rerouteId,
      { x: pos[0], y: pos[1] },
      before.parentId,
      Array.from(linkIds)
    )

    for (const linkId of linkIds) {
      const link = this._links.get(linkId)
      if (!link) continue
      if (link.parentId === before.parentId) link.parentId = rerouteId

      const reroutes = LLink.getReroutes(this, link)
      for (const x of reroutes.filter((x) => x.parentId === before.parentId)) {
        x.parentId = rerouteId
      }
    }

    for (const linkId of floatingLinkIds) {
      const link = this.floatingLinks.get(linkId)
      if (!link) continue
      if (link.parentId === before.parentId) link.parentId = rerouteId

      const reroutes = LLink.getReroutes(this, link)
      for (const x of reroutes.filter((x) => x.parentId === before.parentId)) {
        x.parentId = rerouteId
      }
    }

    return reroute
  }

  /**
   * Removes a reroute from the graph
   * @param id ID of reroute to remove
   */
  removeReroute(id: RerouteId): void {
    const layoutMutations = useLayoutMutations()
    const { reroutes } = this
    const reroute = reroutes.get(id)
    if (!reroute) return

    this.canvasAction((c) => c.deselect(reroute))

    // Extract reroute from the reroute chain
    const { parentId, linkIds, floatingLinkIds } = reroute
    for (const reroute of reroutes.values()) {
      if (reroute.parentId === id) reroute.parentId = parentId
    }

    for (const linkId of linkIds) {
      const link = this._links.get(linkId)
      if (link && link.parentId === id) link.parentId = parentId
    }

    for (const linkId of floatingLinkIds) {
      const link = this.floatingLinks.get(linkId)
      if (!link) {
        console.warn(
          `Removed reroute had floating link ID that did not exist [${linkId}]`
        )
        continue
      }

      // A floating link is a unique branch; if there is no parent reroute, or
      // the parent reroute has any other links, remove this floating link.
      const floatingReroutes = LLink.getReroutes(this, link)
      const lastReroute = floatingReroutes.at(-1)
      const secondLastReroute = floatingReroutes.at(-2)

      if (reroute !== lastReroute) {
        continue
      } else if (secondLastReroute?.totalLinks !== 1) {
        this.removeFloatingLink(link)
      } else if (link.parentId === id) {
        link.parentId = parentId
        secondLastReroute.floating = reroute.floating
      }
    }

    reroutes.delete(id)

    // Delete reroute from Layout Store
    layoutMutations.setSource(LayoutSource.Canvas)
    layoutMutations.deleteReroute(id)

    // This does not belong here; it should be handled by the caller, or run by a remove-many API.
    // https://github.com/Comfy-Org/litegraph.js/issues/898
    this.setDirtyCanvas(false, true)
  }

  /**
   * Destroys a link
   */
  removeLink(link_id: LinkId): void {
    const link = this._links.get(link_id)
    if (!link) return

    const node = this.getNodeById(link.target_id)
    node?.disconnectInput(link.target_slot, false)

    link.disconnect(this)
  }

  /**
   * Creates a new subgraph definition, and adds it to the graph.
   * @param data Exported data (typically serialised) to configure the new subgraph with
   * @returns The newly created subgraph definition.
   */
  createSubgraph(data: ExportedSubgraph): Subgraph {
    const { id } = data

    const subgraph = new Subgraph(this.rootGraph, data)
    this.subgraphs.set(id, subgraph)

    // FE: Create node defs
    this.rootGraph.events.dispatch('subgraph-created', { subgraph, data })
    return subgraph
  }

  convertToSubgraph(items: Set<Positionable>): {
    subgraph: Subgraph
    node: SubgraphNode
  } {
    if (items.size === 0)
      throw new Error('Cannot convert to subgraph: nothing to convert')

    // Record state before conversion for proper undo support
    this.beforeChange()

    try {
      return this._convertToSubgraphImpl(items)
    } finally {
      // Mark state change complete for proper undo support
      this.afterChange()
    }
  }

  private _convertToSubgraphImpl(items: Set<Positionable>): {
    subgraph: Subgraph
    node: SubgraphNode
  } {
    const { state, revision, config } = this
    const firstChild = [...items][0]
    if (items.size === 1 && firstChild instanceof LGraphGroup) {
      items = new Set([firstChild])
      firstChild.recomputeInsideNodes()
      firstChild.children.forEach((n) => items.add(n))
    }

    const {
      boundaryLinks,
      boundaryFloatingLinks,
      internalLinks,
      boundaryInputLinks,
      boundaryOutputLinks
    } = getBoundaryLinks(this, items)
    const { nodes, reroutes, groups } = splitPositionables(items)

    const boundingRect = createBounds(items)
    if (!boundingRect)
      throw new Error('Failed to create bounding rect for subgraph')

    const resolvedInputLinks = boundaryInputLinks.map((x) => x.resolve(this))
    const resolvedOutputLinks = boundaryOutputLinks.map((x) => x.resolve(this))

    const clonedNodes = multiClone(nodes)

    // Inputs, outputs, and links
    const links = internalLinks.map((x) => x.asSerialisable())

    const internalReroutes = new Map([...reroutes].map((r) => [r.id, r]))
    const externalReroutes = new Map(
      [...this.reroutes].filter(([id]) => !internalReroutes.has(id))
    )
    const inputs = mapSubgraphInputsAndLinks(
      resolvedInputLinks,
      links,
      internalReroutes
    )
    const outputs = mapSubgraphOutputsAndLinks(
      resolvedOutputLinks,
      links,
      externalReroutes
    )

    // Prepare subgraph data
    const data = {
      id: createUuidv4(),
      name: 'New Subgraph',
      inputNode: {
        id: SUBGRAPH_INPUT_ID,
        bounding: [0, 0, 75, 100]
      },
      outputNode: {
        id: SUBGRAPH_OUTPUT_ID,
        bounding: [0, 0, 75, 100]
      },
      inputs,
      outputs,
      widgets: [],
      version: LGraph.serialisedSchemaVersion,
      state,
      revision,
      config,
      links,
      nodes: clonedNodes,
      reroutes: structuredClone(
        [...reroutes].map((reroute) => reroute.asSerialisable())
      ),
      groups: structuredClone([...groups].map((group) => group.serialize()))
    } satisfies ExportedSubgraph

    const subgraph = this.createSubgraph(data)
    subgraph.configure(data)
    for (const node of subgraph.nodes) node.onGraphConfigured?.()
    for (const node of subgraph.nodes) node.onAfterGraphConfigured?.()

    // Position the subgraph input nodes
    subgraph.inputNode.arrange()
    subgraph.outputNode.arrange()
    const { boundingRect: inputRect } = subgraph.inputNode
    const { boundingRect: outputRect } = subgraph.outputNode
    alignOutsideContainer(inputRect, Alignment.MidLeft, boundingRect, [50, 0])
    alignOutsideContainer(outputRect, Alignment.MidRight, boundingRect, [50, 0])

    // Remove items converted to subgraph
    for (const resolved of resolvedInputLinks)
      resolved.inputNode?.disconnectInput(
        resolved.inputNode.inputs.indexOf(resolved.input!),
        true
      )
    for (const resolved of resolvedOutputLinks)
      resolved.outputNode?.disconnectOutput(
        resolved.outputNode.outputs.indexOf(resolved.output!),
        resolved.inputNode
      )

    for (const node of nodes) this.remove(node)
    for (const reroute of reroutes) this.removeReroute(reroute.id)
    for (const group of groups) this.remove(group)

    this.rootGraph.events.dispatch('convert-to-subgraph', {
      subgraph,
      bounds: boundingRect,
      exportedSubgraph: data,
      boundaryLinks,
      resolvedInputLinks,
      resolvedOutputLinks,
      boundaryFloatingLinks,
      internalLinks
    })

    // Create subgraph node object
    const subgraphNode = LiteGraph.createNode(subgraph.id, subgraph.name, {
      outputs: structuredClone(outputs)
    })
    if (!subgraphNode) throw new Error('Failed to create subgraph node')
    for (let i = 0; i < inputs.length; i++) {
      Object.assign(subgraphNode.inputs[i], inputs[i])
    }

    // Resize to inputs/outputs
    subgraphNode.setSize(subgraphNode.computeSize())

    // Center the subgraph node
    alignToContainer(
      subgraphNode._posSize,
      Alignment.Centre | Alignment.Middle,
      boundingRect
    )

    //Correct for title height. It's included in bounding box, but not _posSize
    subgraphNode.setPos(
      subgraphNode.pos[0],
      subgraphNode.pos[1] + LiteGraph.NODE_TITLE_HEIGHT / 2
    )

    // Add the subgraph node to the graph
    this.add(subgraphNode)

    // Group matching input links
    const groupedByOutput = groupResolvedByOutput(resolvedInputLinks)

    // Reconnect input links in parent graph
    let i = 0
    for (const [, connections] of groupedByOutput.entries()) {
      const [firstResolved, ...others] = connections
      const { output, outputNode, link, subgraphInput } = firstResolved

      // Special handling: Subgraph input node
      i++
      if (link.origin_id === SUBGRAPH_INPUT_ID) {
        link.target_id = subgraphNode.id
        link.target_slot = i - 1
        if (subgraphInput instanceof SubgraphInput) {
          subgraphInput.connect(
            subgraphNode.findInputSlotByType(link.type, true, true),
            subgraphNode,
            link.parentId
          )
        } else {
          throw new TypeError('Subgraph input node is not a SubgraphInput')
        }

        for (const resolved of others) {
          resolved.link.disconnect(this)
        }
        continue
      }

      if (!output || !outputNode) {
        console.warn(
          'Convert to Subgraph reconnect: Failed to resolve input link',
          connections[0]
        )
        continue
      }

      const input = subgraphNode.inputs[i - 1]
      outputNode.connectSlots(output, subgraphNode, input, link.parentId)
    }

    // Group matching links
    const outputsGroupedByOutput = groupResolvedByOutput(resolvedOutputLinks)

    // Reconnect output links in parent graph
    i = 0
    for (const [, connections] of outputsGroupedByOutput.entries()) {
      i++
      for (const connection of connections) {
        const { input, inputNode, link, subgraphOutput } = connection
        // Special handling: Subgraph output node
        if (link.target_id === SUBGRAPH_OUTPUT_ID) {
          link.origin_id = subgraphNode.id
          link.origin_slot = i - 1
          this.links.set(link.id, link)
          if (subgraphOutput instanceof SubgraphOutput) {
            subgraphOutput.connect(
              subgraphNode.findOutputSlotByType(link.type, true, true),
              subgraphNode,
              link.parentId
            )
          } else {
            throw new TypeError('Subgraph input node is not a SubgraphInput')
          }
          continue
        }

        if (!input || !inputNode) {
          console.warn(
            'Convert to Subgraph reconnect: Failed to resolve output link',
            connection
          )
          continue
        }

        const output = subgraphNode.outputs[i - 1]
        subgraphNode.connectSlots(output, inputNode, input, link.parentId)
      }
    }

    subgraphNode._setConcreteSlots()
    subgraphNode.arrange()

    this.canvasAction((c) =>
      c.canvas.dispatchEvent(
        new CustomEvent('subgraph-converted', {
          bubbles: true,
          detail: { subgraphNode: subgraphNode as SubgraphNode }
        })
      )
    )
    return { subgraph, node: subgraphNode as SubgraphNode }
  }

  unpackSubgraph(
    subgraphNode: SubgraphNode,
    options?: { skipMissingNodes?: boolean }
  ) {
    if (!(subgraphNode instanceof SubgraphNode))
      throw new Error('Can only unpack Subgraph Nodes')

    // Record state before unpacking for proper undo support
    this.beforeChange()

    try {
      this._unpackSubgraphImpl(subgraphNode, options)
    } finally {
      // Mark state change complete for proper undo support
      this.afterChange()
    }
  }

  private _unpackSubgraphImpl(
    subgraphNode: SubgraphNode,
    options?: { skipMissingNodes?: boolean }
  ) {
    const skipMissingNodes = options?.skipMissingNodes ?? false

    //NOTE: Create bounds can not be called on positionables directly as the subgraph is not being displayed and boundingRect is not initialized.
    //NOTE: NODE_TITLE_HEIGHT is explicitly excluded here
    const positionables = [
      ...subgraphNode.subgraph.nodes,
      ...subgraphNode.subgraph.reroutes.values(),
      ...subgraphNode.subgraph.groups
    ].map((p: { pos: Point; size?: Size }): HasBoundingRect => {
      return {
        boundingRect: [p.pos[0], p.pos[1], p.size?.[0] ?? 0, p.size?.[1] ?? 0]
      }
    })
    const bounds = createBounds(positionables) ?? [0, 0, 0, 0]
    const center = [bounds[0] + bounds[2] / 2, bounds[1] + bounds[3] / 2]

    const toSelect: Positionable[] = []
    const offsetX = subgraphNode.pos[0] - center[0] + subgraphNode.size[0] / 2
    const offsetY = subgraphNode.pos[1] - center[1] + subgraphNode.size[1] / 2
    const movedNodes = multiClone(subgraphNode.subgraph.nodes)
    const nodeIdMap = new Map<NodeId, NodeId>()
    for (const n_info of movedNodes) {
      let node = LiteGraph.createNode(String(n_info.type), n_info.title)
      if (!node) {
        if (skipMissingNodes) {
          console.warn(
            `Cannot unpack node of type "${n_info.type}" - node type not found. Creating placeholder node.`
          )
          node = new LGraphNode(n_info.title || n_info.type || 'Missing Node')
          node.last_serialization = n_info
          node.has_errors = true
          node.type = String(n_info.type)
        } else {
          throw new Error(
            `Cannot unpack: node type "${n_info.type}" is not registered`
          )
        }
      }

      nodeIdMap.set(n_info.id, ++this.last_node_id)
      node.id = this.last_node_id
      n_info.id = this.last_node_id

      // Strip links from serialized data before configure to prevent
      // onConnectionsChange from resolving subgraph-internal link IDs
      // against the parent graph's link map (which may contain unrelated
      // links with the same numeric IDs).
      for (const input of n_info.inputs ?? []) {
        input.link = null
      }
      for (const output of n_info.outputs ?? []) {
        output.links = []
      }

      this.add(node, true)
      node.configure(n_info)
      node.setPos(node.pos[0] + offsetX, node.pos[1] + offsetY)
      toSelect.push(node)
    }
    const groups = structuredClone(
      [...subgraphNode.subgraph.groups].map((g) => g.serialize())
    )
    for (const g_info of groups) {
      const group = new LGraphGroup(g_info.title, g_info.id)
      this.add(group, true)
      group.configure(g_info)
      group.pos[0] += offsetX
      group.pos[1] += offsetY
      toSelect.push(group)
    }
    //cleanup reoute.linkIds now, but leave link.parentIds dangling
    for (const islot of subgraphNode.inputs) {
      if (!islot.link) continue
      const link = this.links.get(islot.link)
      if (!link) {
        console.warn('Broken link', islot, islot.link)
        continue
      }
      for (const reroute of LLink.getReroutes(this, link)) {
        reroute.linkIds.delete(link.id)
      }
    }
    for (const oslot of subgraphNode.outputs) {
      for (const linkId of oslot.links ?? []) {
        const link = this.links.get(linkId)
        if (!link) {
          console.warn('Broken link', oslot, linkId)
          continue
        }
        for (const reroute of LLink.getReroutes(this, link)) {
          reroute.linkIds.delete(link.id)
        }
      }
    }
    const newLinks: {
      oid: NodeId
      oslot: number
      tid: NodeId
      tslot: number
      id: LinkId
      iparent?: RerouteId
      eparent?: RerouteId
      externalFirst: boolean
    }[] = []
    for (const [, link] of subgraphNode.subgraph._links) {
      let externalParentId: RerouteId | undefined
      if (link.origin_id === SUBGRAPH_INPUT_ID) {
        const outerLinkId = subgraphNode.inputs[link.origin_slot].link
        if (!outerLinkId) {
          console.error('Missing Link ID when unpacking')
          continue
        }
        const outerLink = this.links[outerLinkId]
        link.origin_id = outerLink.origin_id
        link.origin_slot = outerLink.origin_slot
        externalParentId = outerLink.parentId
      } else {
        const origin_id = nodeIdMap.get(link.origin_id)
        if (!origin_id) {
          console.error('Missing Link ID when unpacking')
          continue
        }
        link.origin_id = origin_id
      }
      if (link.target_id === SUBGRAPH_OUTPUT_ID) {
        for (const linkId of subgraphNode.outputs[link.target_slot].links ??
          []) {
          const sublink = this.links[linkId]
          newLinks.push({
            oid: link.origin_id,
            oslot: link.origin_slot,
            tid: sublink.target_id,
            tslot: sublink.target_slot,
            id: link.id,
            iparent: link.parentId,
            eparent: sublink.parentId,
            externalFirst: true
          })
          sublink.parentId = undefined
        }
        continue
      } else {
        const target_id = nodeIdMap.get(link.target_id)
        if (!target_id) {
          console.error('Missing Link ID when unpacking')
          continue
        }
        link.target_id = target_id
      }
      newLinks.push({
        oid: link.origin_id,
        oslot: link.origin_slot,
        tid: link.target_id,
        tslot: link.target_slot,
        id: link.id,
        iparent: link.parentId,
        eparent: externalParentId,
        externalFirst: false
      })
    }
    this.remove(subgraphNode)
    this.subgraphs.delete(subgraphNode.subgraph.id)

    // Deduplicate links by (oid, oslot, tid, tslot) to prevent repeated
    // disconnect/reconnect cycles on widget inputs that can shift slot indices.
    const seenLinks = new Set<string>()
    const dedupedNewLinks = newLinks.filter((link) => {
      const key = `${link.oid}:${link.oslot}:${link.tid}:${link.tslot}`
      if (seenLinks.has(key)) return false
      seenLinks.add(key)
      return true
    })

    const linkIdMap = new Map<LinkId, LinkId[]>()
    for (const newLink of dedupedNewLinks) {
      let created: LLink | null | undefined
      if (newLink.oid == SUBGRAPH_INPUT_ID) {
        if (!(this instanceof Subgraph)) {
          console.error('Ignoring link to subgraph outside subgraph')
          continue
        }
        const tnode = this._nodes_by_id[newLink.tid]
        created = this.inputNode.slots[newLink.oslot].connect(
          tnode.inputs[newLink.tslot],
          tnode
        )
      } else if (newLink.tid == SUBGRAPH_OUTPUT_ID) {
        if (!(this instanceof Subgraph)) {
          console.error('Ignoring link to subgraph outside subgraph')
          continue
        }
        const tnode = this._nodes_by_id[newLink.oid]
        created = this.outputNode.slots[newLink.tslot].connect(
          tnode.outputs[newLink.oslot],
          tnode
        )
      } else {
        created = this._nodes_by_id[newLink.oid].connect(
          newLink.oslot,
          this._nodes_by_id[newLink.tid],
          newLink.tslot
        )
      }
      if (!created) {
        console.error('Failed to create link')
        continue
      }
      //This is a little unwieldy since Map.has isn't a type guard
      const linkIds = linkIdMap.get(newLink.id) ?? []
      linkIds.push(created.id)
      if (!linkIdMap.has(newLink.id)) {
        linkIdMap.set(newLink.id, linkIds)
      }
      newLink.id = created.id
    }
    const rerouteIdMap = new Map<RerouteId, RerouteId>()
    for (const reroute of subgraphNode.subgraph.reroutes.values()) {
      if (
        reroute.parentId !== undefined &&
        rerouteIdMap.get(reroute.parentId) === undefined
      ) {
        console.error('Missing Parent ID')
      }
      const migratedReroute = new Reroute(++this.state.lastRerouteId, this, [
        reroute.pos[0] + offsetX,
        reroute.pos[1] + offsetY
      ])
      rerouteIdMap.set(reroute.id, migratedReroute.id)
      this.reroutes.set(migratedReroute.id, migratedReroute)
      toSelect.push(migratedReroute)
    }
    //iterate over newly created links to update reroute parentIds
    for (const newLink of dedupedNewLinks) {
      const linkInstance = this.links.get(newLink.id)
      if (!linkInstance) {
        continue
      }
      let instance: Reroute | LLink | undefined = linkInstance
      let parentId: RerouteId | undefined = undefined
      if (newLink.externalFirst) {
        parentId = newLink.eparent
        //TODO: recursion check/helper method? Probably exists, but wouldn't mesh with the reference tracking used by this implementation
        while (parentId) {
          instance.parentId = parentId
          instance = this.reroutes.get(parentId)
          if (!instance) {
            console.error('Broken Id link when unpacking')
            break
          }
          if (instance.linkIds.has(linkInstance.id))
            throw new Error('Infinite parentId loop')
          instance.linkIds.add(linkInstance.id)
          parentId = instance.parentId
        }
      }
      if (!instance) continue
      parentId = newLink.iparent
      while (parentId) {
        const migratedId = rerouteIdMap.get(parentId)
        if (!migratedId) {
          console.error('Broken Id link when unpacking')
          break
        }
        instance.parentId = migratedId
        instance = this.reroutes.get(migratedId)
        if (!instance) {
          console.error('Broken Id link when unpacking')
          break
        }
        if (instance.linkIds.has(linkInstance.id))
          throw new Error('Infinite parentId loop')
        instance.linkIds.add(linkInstance.id)
        const oldReroute = subgraphNode.subgraph.reroutes.get(parentId)
        if (!oldReroute) {
          console.error('Broken Id link when unpacking')
          break
        }
        parentId = oldReroute.parentId
      }
      if (!instance) break
      if (!newLink.externalFirst) {
        parentId = newLink.eparent
        while (parentId) {
          instance.parentId = parentId
          instance = this.reroutes.get(parentId)
          if (!instance) {
            console.error('Broken Id link when unpacking')
            break
          }
          if (instance.linkIds.has(linkInstance.id))
            throw new Error('Infinite parentId loop')
          instance.linkIds.add(linkInstance.id)
          parentId = instance.parentId
        }
      }
    }

    for (const nodeId of nodeIdMap.values()) {
      const node = this._nodes_by_id[nodeId]
      node._setConcreteSlots()
      node.arrange()
    }

    this.canvasAction((c) => c.selectItems(toSelect))
  }

  /**
   * Resolve a path of subgraph node IDs into a list of subgraph nodes.
   * Not intended to be run from subgraphs.
   * @param nodeIds An ordered list of node IDs, from the root graph to the most nested subgraph node
   * @returns An ordered list of nested subgraph nodes.
   */
  resolveSubgraphIdPath(nodeIds: readonly NodeId[]): SubgraphNode[] {
    const result: SubgraphNode[] = []
    let currentGraph: GraphOrSubgraph = this.rootGraph

    for (const nodeId of nodeIds) {
      const node: LGraphNode | null = currentGraph.getNodeById(nodeId)
      if (!node)
        throw new Error(
          `Node [${nodeId}] not found.  ID Path: ${nodeIds.join(':')}`
        )
      if (!node.isSubgraphNode())
        throw new Error(
          `Node [${nodeId}] is not a SubgraphNode.  ID Path: ${nodeIds.join(':')}`
        )

      result.push(node)
      currentGraph = node.subgraph
    }

    return result
  }

  /**
   * Creates a Object containing all the info about this graph, it can be serialized
   * @deprecated Use {@link asSerialisable}, which returns the newer schema version.
   * @returns value of the node
   */
  serialize(option?: { sortNodes: boolean }): ISerialisedGraph {
    const {
      config,
      state,
      groups,
      nodes,
      reroutes,
      extra,
      floatingLinks,
      definitions
    } = this.asSerialisable(option)
    const linkArray = [...this._links.values()]
    const links = linkArray.map((x) => x.serialize())

    if (reroutes?.length) {
      // Link parent IDs cannot go in 0.4 schema arrays
      extra.linkExtensions = linkArray
        .filter((x) => x.parentId !== undefined)
        .map((x) => ({ id: x.id, parentId: x.parentId }))
    }

    extra.reroutes = reroutes?.length ? reroutes : undefined
    return {
      id: this.id,
      revision: this.revision,
      last_node_id: state.lastNodeId,
      last_link_id: state.lastLinkId,
      nodes,
      links,
      floatingLinks,
      groups,
      definitions,
      config,
      extra,
      version: LiteGraph.VERSION
    }
  }

  /**
   * Custom JSON serialization to prevent circular reference errors.
   * Called automatically by JSON.stringify().
   */
  toJSON(): ISerialisedGraph {
    return this.serialize()
  }

  /** @returns The drag and scale state of the first attached canvas, otherwise `undefined`. */
  private _getDragAndScale(): DragAndScaleState | undefined {
    const ds = this.list_of_graphcanvas?.at(0)?.ds
    if (ds) return { scale: ds.scale, offset: ds.offset }
  }

  /**
   * Prepares a shallow copy of this object for immediate serialisation or structuredCloning.
   * The return value should be discarded immediately.
   * @param options Serialise options = currently `sortNodes: boolean`, whether to sort nodes by ID.
   * @returns A shallow copy of parts of this graph, with shallow copies of its serialisable objects.
   * Mutating the properties of the return object may result in changes to your graph.
   * It is intended for use with {@link structuredClone} or {@link JSON.stringify}.
   */
  asSerialisable(options?: {
    sortNodes: boolean
  }): SerialisableGraph &
    Required<Pick<SerialisableGraph, 'nodes' | 'groups' | 'extra'>> {
    const { id, revision, config, state } = this

    const nodeList =
      !LiteGraph.use_uuids && options?.sortNodes
        ? // @ts-expect-error If LiteGraph.use_uuids is false, ids are numbers.
          [...this._nodes].sort((a, b) => a.id - b.id)
        : this._nodes

    const nodes = nodeList.map((node) => node.serialize())
    const groups = this._groups.map((x) => x.serialize())

    const links = this._links.size
      ? [...this._links.values()].map((x) => x.asSerialisable())
      : undefined
    const floatingLinks = this.floatingLinks.size
      ? [...this.floatingLinks.values()].map((x) => x.asSerialisable())
      : undefined
    const reroutes = this.reroutes.size
      ? [...this.reroutes.values()].map((x) => x.asSerialisable())
      : undefined

    // Save scale and offset
    const extra = { ...this.extra }
    if (LiteGraph.saveViewportWithGraph) extra.ds = this._getDragAndScale()
    if (!extra.ds) delete extra.ds

    const data: ReturnType<typeof this.asSerialisable> = {
      id,
      revision,
      version: LGraph.serialisedSchemaVersion,
      config,
      state,
      groups,
      nodes,
      links,
      floatingLinks,
      reroutes,
      extra
    }

    if (this.isRootGraph && this._subgraphs.size) {
      const usedSubgraphIds = findUsedSubgraphIds(this, this._subgraphs)
      const usedSubgraphs = [...this._subgraphs.values()]
        .filter((subgraph) => usedSubgraphIds.has(subgraph.id))
        .map((x) => x.asSerialisable())

      if (usedSubgraphs.length > 0) {
        data.definitions = { subgraphs: usedSubgraphs }
      }
    }

    this.onSerialize?.(data)
    return data
  }

  protected _configureBase(data: ISerialisedGraph | SerialisableGraph): void {
    const { id, extra } = data

    // Create a new graph ID if none is provided
    if (id) {
      this.id = id
    } else if (this.id === zeroUuid) {
      this.id = createUuidv4()
    }

    // Extra
    this.extra = extra ? structuredClone(extra) : {}

    // Ensure auto-generated serialisation data is removed from extra
    delete this.extra.linkExtensions
  }

  /**
   * Configure a graph from a JSON string
   * @param data The deserialised object to configure this graph from
   * @param keep_old If `true`, the graph will not be cleared prior to
   * adding the configuration.
   */
  configure(
    data: ISerialisedGraph | SerialisableGraph,
    keep_old?: boolean
  ): boolean | undefined {
    const layoutMutations = useLayoutMutations()
    const options: LGraphEventMap['configuring'] = {
      data,
      clearGraph: !keep_old
    }
    const mayContinue = this.events.dispatch('configuring', options)
    if (!mayContinue) return

    try {
      // TODO: Finish typing configure()
      if (!data) return
      if (options.clearGraph) this.clear()

      this._configureBase(data)

      let reroutes: SerialisableReroute[] | undefined

      // TODO: Determine whether this should this fall back to 0.4.
      if (data.version === 0.4) {
        const { extra } = data
        // Deprecated - old schema version, links are arrays
        if (Array.isArray(data.links)) {
          for (const linkData of data.links) {
            const link = LLink.createFromArray(linkData)
            this._links.set(link.id, link)
          }
        }
        // #region `extra` embeds for v0.4

        // LLink parentIds
        if (Array.isArray(extra?.linkExtensions)) {
          for (const linkEx of extra.linkExtensions) {
            const link = this._links.get(linkEx.id)
            if (link) link.parentId = linkEx.parentId
          }
        }

        // Reroutes
        reroutes = extra?.reroutes

        // #endregion `extra` embeds for v0.4
      } else {
        // New schema - one version so far, no check required.

        // State - use max to prevent ID collisions across root and subgraphs
        if (data.state) {
          const { lastGroupId, lastLinkId, lastNodeId, lastRerouteId } =
            data.state
          const { state } = this
          if (lastGroupId != null)
            state.lastGroupId = Math.max(state.lastGroupId, lastGroupId)
          if (lastLinkId != null)
            state.lastLinkId = Math.max(state.lastLinkId, lastLinkId)
          if (lastNodeId != null)
            state.lastNodeId = Math.max(state.lastNodeId, lastNodeId)
          if (lastRerouteId != null)
            state.lastRerouteId = Math.max(state.lastRerouteId, lastRerouteId)
        }

        // Links
        if (Array.isArray(data.links)) {
          for (const linkData of data.links) {
            const link = LLink.create(linkData)
            this._links.set(link.id, link)
          }
        }

        reroutes = data.reroutes
      }

      // Reroutes
      if (Array.isArray(reroutes)) {
        for (const rerouteData of reroutes) {
          this.setReroute(rerouteData)
        }
      }

      const nodesData = data.nodes

      // copy all stored fields
      for (const i in data) {
        if (LGraph.ConfigureProperties.has(i)) continue

        // @ts-expect-error #574 Legacy property assignment
        this[i] = data[i]
      }

      // Subgraph definitions
      const subgraphs = data.definitions?.subgraphs
      if (subgraphs) {
        for (const subgraph of subgraphs) this.createSubgraph(subgraph)
        for (const subgraph of subgraphs)
          this.subgraphs.get(subgraph.id)?.configure(subgraph)
      }

      if (this.isRootGraph) {
        const reservedNodeIds = nodesData
          ?.map((n) => n.id)
          .filter((id): id is number => typeof id === 'number')
        this.ensureGlobalIdUniqueness(reservedNodeIds)
      }

      let error = false
      const nodeDataMap = new Map<NodeId, ISerialisedNode>()

      // create nodes
      this._nodes = []
      if (nodesData) {
        for (const n_info of nodesData) {
          // stored info
          let node = LiteGraph.createNode(String(n_info.type), n_info.title)
          if (!node) {
            if (LiteGraph.debug)
              console.warn('Node not found or has errors:', n_info.type)

            // in case of error we create a replacement node to avoid losing info
            node = new LGraphNode('')
            node.last_serialization = n_info
            node.has_errors = true
            error = true
            // continue;
          }

          // id it or it will create a new id
          node.id = n_info.id
          // add before configure, otherwise configure cannot create links
          this.add(node, true)
          nodeDataMap.set(node.id, n_info)
        }

        // configure nodes afterwards so they can reach each other
        for (const [id, nodeData] of nodeDataMap) {
          this.getNodeById(id)?.configure(nodeData)
        }
      }

      // Floating links
      if (Array.isArray(data.floatingLinks)) {
        for (const linkData of data.floatingLinks) {
          const floatingLink = LLink.create(linkData)
          this.addFloatingLink(floatingLink)

          if (floatingLink.id > this._lastFloatingLinkId)
            this._lastFloatingLinkId = floatingLink.id
        }
      }

      // Drop broken reroutes
      for (const reroute of this.reroutes.values()) {
        // Drop broken links, and ignore reroutes with no valid links
        if (!reroute.validateLinks(this._links, this.floatingLinks)) {
          this.reroutes.delete(reroute.id)
          // Clean up layout store
          layoutMutations.setSource(LayoutSource.Canvas)
          layoutMutations.deleteReroute(reroute.id)
        }
      }

      // groups
      this._groups.length = 0
      const groupData = data.groups
      if (groupData) {
        for (const data of groupData) {
          // TODO: Search/remove these global object refs
          const group = new LiteGraph.LGraphGroup()
          group.configure(data)
          this.add(group)
        }
      }

      this.updateExecutionOrder()

      this.onConfigure?.(data)
      this._version++

      // Ensure the primary canvas is set to the correct graph
      const { primaryCanvas } = this
      const subgraphId = primaryCanvas?.subgraph?.id
      if (subgraphId) {
        const subgraph = this.subgraphs.get(subgraphId)
        if (subgraph) {
          primaryCanvas.setGraph(subgraph)
        } else {
          primaryCanvas.setGraph(this)
        }
      }

      this.setDirtyCanvas(true, true)
      return error
    } finally {
      this.events.dispatch('configured')
    }
  }

  /**
   * Ensures all node IDs are globally unique across the root graph and all
   * subgraphs. Reassigns any colliding IDs found in subgraphs, preserving
   * root graph IDs as canonical. Updates link references (`origin_id`,
   * `target_id`) within the affected graph to match the new node IDs.
   */
  ensureGlobalIdUniqueness(reservedNodeIds?: Iterable<number>): void {
    const { state } = this

    const allGraphs: LGraph[] = [this, ...this._subgraphs.values()]

    const usedNodeIds = new Set<number>(reservedNodeIds)
    for (const graph of allGraphs) {
      const remappedIds = new Map<NodeId, NodeId>()

      for (const node of graph._nodes) {
        if (typeof node.id !== 'number') continue

        if (usedNodeIds.has(node.id)) {
          const oldId = node.id
          while (usedNodeIds.has(++state.lastNodeId));
          const newId = state.lastNodeId
          delete graph._nodes_by_id[oldId]
          node.id = newId
          graph._nodes_by_id[newId] = node
          usedNodeIds.add(newId)
          remappedIds.set(oldId, newId)
          console.warn(
            `LiteGraph: duplicate node ID ${oldId} reassigned to ${newId} in graph ${graph.id}`
          )
        } else {
          usedNodeIds.add(node.id as number)
          if ((node.id as number) > state.lastNodeId)
            state.lastNodeId = node.id as number
        }
      }

      if (remappedIds.size > 0) {
        patchLinkNodeIds(graph._links, remappedIds)
        patchLinkNodeIds(graph.floatingLinksInternal, remappedIds)
      }
    }
  }

  private _canvas?: LGraphCanvas
  get primaryCanvas(): LGraphCanvas | undefined {
    return this.rootGraph._canvas
  }

  set primaryCanvas(canvas: LGraphCanvas) {
    this.rootGraph._canvas = canvas
  }

  load(url: string | Blob | URL | File, callback: () => void) {
    // from file
    if (url instanceof Blob || url instanceof File) {
      const reader = new FileReader()
      reader.addEventListener('load', (event) => {
        const result = toString(event.target?.result)
        const data = JSON.parse(result)
        this.configure(data)
        callback?.()
      })

      reader.readAsText(url)
      return
    }

    // is a string, then an URL
    const req = new XMLHttpRequest()
    req.open('GET', url, true)
    req.send(null)
    req.addEventListener('load', () => {
      if (req.status !== 200) {
        console.error('Error loading graph:', req.status, req.response)
        return
      }
      const data = JSON.parse(req.response)
      this.configure(data)
      callback?.()
    })
    req.addEventListener('error', (err) => {
      console.error('Error loading graph:', err)
    })
  }
}

/** Internal; simplifies type definitions. */
export type GraphOrSubgraph = LGraph | Subgraph

// ============================================================================
// TEMPORARY: Subgraph class moved here to resolve circular dependency
// This is a temporary solution until the architecture can be refactored
// TODO: Move back to separate file once circular dependencies are resolved
// ============================================================================

/** A subgraph definition. */
export class Subgraph
  extends LGraph
  implements BaseLGraph, Serialisable<ExportedSubgraph>
{
  override readonly events = new CustomEventTarget<SubgraphEventMap>()

  /** Limits the number of levels / depth that subgraphs may be nested.  Prevents uncontrolled programmatic nesting. */
  static MAX_NESTED_SUBGRAPHS = 1000

  /** The display name of the subgraph. */
  name: string = 'Unnamed Subgraph'
  /** Optional description shown as tooltip when hovering over the subgraph node. */
  description?: string

  readonly inputNode = new SubgraphInputNode(this)
  readonly outputNode = new SubgraphOutputNode(this)

  /** Ordered list of inputs to the subgraph itself. Similar to a reroute, with the input side in the graph, and the output side in the subgraph. */
  readonly inputs: SubgraphInput[] = []
  /** Ordered list of outputs from the subgraph itself. Similar to a reroute, with the input side in the subgraph, and the output side in the graph. */
  readonly outputs: SubgraphOutput[] = []
  /** A list of node widgets displayed in the parent graph, on the subgraph object. */
  readonly widgets: ExposedWidget[] = []

  private _rootGraph: LGraph
  override get rootGraph(): LGraph {
    return this._rootGraph
  }

  override get state(): LGraphState {
    return this._rootGraph.state
  }

  override set state(_value: LGraphState) {
    // No-op: subgraphs share the root graph's state.
  }

  constructor(rootGraph: LGraph, data: ExportedSubgraph) {
    if (!rootGraph) throw new Error('Root graph is required')

    super()

    this._rootGraph = rootGraph

    const cloned = structuredClone(data)
    this._configureBase(cloned)
    this._configureSubgraph(cloned)
  }

  getIoNodeOnPos(
    x: number,
    y: number
  ): SubgraphInputNode | SubgraphOutputNode | undefined {
    const { inputNode, outputNode } = this
    if (inputNode.containsPoint([x, y])) return inputNode
    if (outputNode.containsPoint([x, y])) return outputNode
  }

  private _configureSubgraph(
    data:
      | (ISerialisedGraph & ExportedSubgraph)
      | (SerialisableGraph & ExportedSubgraph)
  ): void {
    const { name, description, inputs, outputs, widgets } = data

    this.name = name
    this.description = description
    if (inputs) {
      this.inputs.length = 0
      for (const input of inputs) {
        const subgraphInput = new SubgraphInput(input, this.inputNode)
        this.inputs.push(subgraphInput)
        this.events.dispatch('input-added', { input: subgraphInput })
      }
    }

    if (outputs) {
      this.outputs.length = 0
      for (const output of outputs) {
        this.outputs.push(new SubgraphOutput(output, this.outputNode))
      }
    }

    if (widgets) {
      this.widgets.length = 0
      for (const widget of widgets) {
        this.widgets.push(widget)
      }
    }

    this.inputNode.configure(data.inputNode)
    this.outputNode.configure(data.outputNode)
    for (const node of this.nodes) node.updateComputedDisabled()
  }

  override configure(
    data:
      | (ISerialisedGraph & ExportedSubgraph)
      | (SerialisableGraph & ExportedSubgraph),
    keep_old?: boolean
  ): boolean | undefined {
    const r = super.configure(data, keep_old)

    this._configureSubgraph(data)
    return r
  }

  override attachCanvas(canvas: LGraphCanvas): void {
    super.attachCanvas(canvas)
    canvas.subgraph = this
  }

  addInput(name: string, type: string): SubgraphInput {
    if (name === null || type === null) {
      throw new Error('Name and type are required for subgraph input')
    }

    this.events.dispatch('adding-input', { name, type })

    const input = new SubgraphInput(
      {
        id: createUuidv4(),
        name,
        type
      },
      this.inputNode
    )

    this.inputs.push(input)
    this.events.dispatch('input-added', { input })

    return input
  }

  addOutput(name: string, type: string): SubgraphOutput {
    if (name === null || type === null) {
      throw new Error('Name and type are required for subgraph output')
    }

    this.events.dispatch('adding-output', { name, type })

    const output = new SubgraphOutput(
      {
        id: createUuidv4(),
        name,
        type
      },
      this.outputNode
    )

    this.outputs.push(output)
    this.events.dispatch('output-added', { output })

    return output
  }

  /**
   * Renames an input slot in the subgraph.
   * @param input The input slot to rename.
   * @param name The new name for the input slot.
   */
  renameInput(input: SubgraphInput, name: string): void {
    const index = this.inputs.indexOf(input)
    if (index === -1) throw new Error('Input not found')

    const oldName = input.displayName
    this.events.dispatch('renaming-input', {
      input,
      index,
      oldName,
      newName: name
    })

    input.label = name
  }

  /**
   * Renames an output slot in the subgraph.
   * @param output The output slot to rename.
   * @param name The new name for the output slot.
   */
  renameOutput(output: SubgraphOutput, name: string): void {
    const index = this.outputs.indexOf(output)
    if (index === -1) throw new Error('Output not found')

    const oldName = output.displayName
    this.events.dispatch('renaming-output', {
      output,
      index,
      oldName,
      newName: name
    })

    output.label = name
  }

  /**
   * Removes an input slot from the subgraph.
   * @param input The input slot to remove.
   */
  removeInput(input: SubgraphInput): void {
    input.disconnect()

    const index = this.inputs.indexOf(input)
    if (index === -1) throw new Error('Input not found')

    const mayContinue = this.events.dispatch('removing-input', { input, index })
    if (!mayContinue) return

    this.inputs.splice(index, 1)

    const { length } = this.inputs
    for (let i = index; i < length; i++) {
      this.inputs[i].decrementSlots('inputs')
    }
  }

  /**
   * Removes an output slot from the subgraph.
   * @param output The output slot to remove.
   */
  removeOutput(output: SubgraphOutput): void {
    output.disconnect()

    const index = this.outputs.indexOf(output)
    if (index === -1) throw new Error('Output not found')

    const mayContinue = this.events.dispatch('removing-output', {
      output,
      index
    })
    if (!mayContinue) return

    this.outputs.splice(index, 1)

    const { length } = this.outputs
    for (let i = index; i < length; i++) {
      this.outputs[i].decrementSlots('outputs')
    }
  }

  draw(
    ctx: CanvasRenderingContext2D,
    colorContext: DefaultConnectionColors,
    fromSlot?:
      | INodeInputSlot
      | INodeOutputSlot
      | SubgraphInput
      | SubgraphOutput,
    editorAlpha?: number
  ): void {
    this.inputNode.draw(ctx, colorContext, fromSlot, editorAlpha)
    this.outputNode.draw(ctx, colorContext, fromSlot, editorAlpha)
  }

  /**
   * Clones the subgraph, creating an identical copy with a new ID.
   * @returns A new subgraph with the same configuration, but a new ID.
   */
  clone(keepId: boolean = false): Subgraph {
    const exported = this.asSerialisable()
    if (!keepId) exported.id = createUuidv4()

    const subgraph = new Subgraph(this.rootGraph, exported)
    subgraph.configure(exported)
    return subgraph
  }

  override asSerialisable(): ExportedSubgraph &
    Required<Pick<SerialisableGraph, 'nodes' | 'groups' | 'extra'>> {
    return {
      id: this.id,
      version: LGraph.serialisedSchemaVersion,
      state: this.state,
      revision: this.revision,
      config: this.config,
      name: this.name,
      ...(this.description && { description: this.description }),
      inputNode: this.inputNode.asSerialisable(),
      outputNode: this.outputNode.asSerialisable(),
      inputs: this.inputs.map((x) => x.asSerialisable()),
      outputs: this.outputs.map((x) => x.asSerialisable()),
      widgets: [...this.widgets],
      nodes: this.nodes.map((node) => node.serialize()),
      groups: this.groups.map((group) => group.serialize()),
      links: [...this.links.values()].map((x) => x.asSerialisable()),
      reroutes: this.reroutes.size
        ? [...this.reroutes.values()].map((x) => x.asSerialisable())
        : undefined,
      extra: this.extra
    }
  }
}

function patchLinkNodeIds(
  links: Map<LinkId, LLink>,
  remappedIds: Map<NodeId, NodeId>
): void {
  for (const link of links.values()) {
    const newOrigin = remappedIds.get(link.origin_id)
    if (newOrigin !== undefined) link.origin_id = newOrigin

    const newTarget = remappedIds.get(link.target_id)
    if (newTarget !== undefined) link.target_id = newTarget
  }
}
