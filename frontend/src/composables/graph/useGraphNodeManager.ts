/**
 * Vue node lifecycle management for LiteGraph integration
 * Provides event-driven reactivity with performance optimizations
 */
import { reactiveComputed } from '@vueuse/core'
import { reactive, shallowReactive } from 'vue'

import { useChainCallback } from '@/composables/functional/useChainCallback'
import { isPromotedWidgetView } from '@/core/graph/subgraph/promotedWidgetTypes'
import { resolvePromotedWidgetSource } from '@/core/graph/subgraph/resolvePromotedWidgetSource'
import type {
  INodeInputSlot,
  INodeOutputSlot
} from '@/lib/litegraph/src/interfaces'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import { useLayoutMutations } from '@/renderer/core/layout/operations/layoutMutations'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import { LayoutSource } from '@/renderer/core/layout/types'
import type { NodeId } from '@/renderer/core/layout/types'
import type { InputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import { isDOMWidget } from '@/scripts/domWidget'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import type { WidgetValue, SafeControlWidget } from '@/types/simplifiedWidget'
import { normalizeControlOption } from '@/types/simplifiedWidget'

import type {
  LGraph,
  LGraphBadge,
  LGraphNode,
  LGraphTriggerAction,
  LGraphTriggerEvent,
  LGraphTriggerParam
} from '@/lib/litegraph/src/litegraph'
import type { TitleMode } from '@/lib/litegraph/src/types/globalEnums'
import { NodeSlotType } from '@/lib/litegraph/src/types/globalEnums'
import { app } from '@/scripts/app'

export interface WidgetSlotMetadata {
  index: number
  linked: boolean
}

/**
 * Minimal render-specific widget data extracted from LiteGraph widgets.
 * Value and metadata (label, hidden, disabled, etc.) are accessed via widgetValueStore.
 */
export interface SafeWidgetData {
  nodeId?: NodeId
  name: string
  type: string
  /** Callback to invoke when widget value changes (wraps LiteGraph callback + triggerDraw) */
  callback?: ((value: unknown) => void) | undefined
  /** Control widget for seed randomization/increment/decrement */
  controlWidget?: SafeControlWidget
  /** Whether widget has custom layout size computation */
  hasLayoutSize?: boolean
  /** Whether widget is a DOM widget */
  isDOMWidget?: boolean
  /**
   * Widget options needed for render decisions.
   * Note: Most metadata should be accessed via widgetValueStore.getWidget().
   */
  options?: {
    canvasOnly?: boolean
    advanced?: boolean
    hidden?: boolean
    read_only?: boolean
  }
  /** Input specification from node definition */
  spec?: InputSpec
  /** Input slot metadata (index and link status) */
  slotMetadata?: WidgetSlotMetadata
  /**
   * Original LiteGraph widget name used for slot metadata matching.
   * For promoted widgets, `name` is `sourceWidgetName` (interior widget name)
   * which differs from the subgraph node's input slot widget name.
   */
  slotName?: string
}

export interface VueNodeData {
  executing: boolean
  id: NodeId
  mode: number
  selected: boolean
  title: string
  type: string
  apiNode?: boolean
  badges?: (LGraphBadge | (() => LGraphBadge))[]
  bgcolor?: string
  color?: string
  flags?: {
    collapsed?: boolean
    ghost?: boolean
    pinned?: boolean
  }
  hasErrors?: boolean
  inputs?: INodeInputSlot[]
  outputs?: INodeOutputSlot[]
  resizable?: boolean
  shape?: number
  showAdvanced?: boolean
  subgraphId?: string | null
  titleMode?: TitleMode
  widgets?: SafeWidgetData[]
}

export interface GraphNodeManager {
  // Reactive state - safe data extracted from LiteGraph nodes
  vueNodeData: ReadonlyMap<string, VueNodeData>

  // Access to original LiteGraph nodes (non-reactive)
  getNode(id: string): LGraphNode | undefined

  // Lifecycle methods
  cleanup(): void
}

function isPromotedDOMWidget(widget: IBaseWidget): boolean {
  if (!isPromotedWidgetView(widget)) return false
  const sourceWidget = resolvePromotedWidgetSource(widget.node, widget)
  if (!sourceWidget) return false

  const innerWidget = sourceWidget.widget
  return (
    ('element' in innerWidget && !!innerWidget.element) ||
    ('component' in innerWidget && !!innerWidget.component)
  )
}

export function getControlWidget(
  widget: IBaseWidget
): SafeControlWidget | undefined {
  const cagWidget = widget.linkedWidgets?.find(
    (w) => w.name == 'control_after_generate'
  )
  if (!cagWidget) return
  return {
    value: normalizeControlOption(cagWidget.value),
    update: (value) => (cagWidget.value = normalizeControlOption(value))
  }
}

interface SharedWidgetEnhancements {
  controlWidget?: SafeControlWidget
  spec?: InputSpec
}

function getSharedWidgetEnhancements(
  node: LGraphNode,
  widget: IBaseWidget
): SharedWidgetEnhancements {
  const nodeDefStore = useNodeDefStore()

  return {
    controlWidget: getControlWidget(widget),
    spec: nodeDefStore.getInputSpecForWidget(node, widget.name)
  }
}

/**
 * Validates that a value is a valid WidgetValue type
 */
const normalizeWidgetValue = (value: unknown): WidgetValue => {
  if (value === null || value === undefined || value === void 0) {
    return undefined
  }
  if (
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    return value
  }
  if (typeof value === 'object') {
    // Check if it's a File array
    if (
      Array.isArray(value) &&
      value.length > 0 &&
      value.every((item): item is File => item instanceof File)
    ) {
      return value
    }
    // Otherwise it's a generic object
    return value
  }
  // If none of the above, return undefined
  console.warn(`Invalid widget value type: ${typeof value}`, value)
  return undefined
}

function safeWidgetMapper(
  node: LGraphNode,
  slotMetadata: Map<string, WidgetSlotMetadata>
): (widget: IBaseWidget) => SafeWidgetData {
  return function (widget) {
    try {
      // Get shared enhancements (controlWidget, spec, nodeType)
      const sharedEnhancements = getSharedWidgetEnhancements(node, widget)
      const slotInfo = slotMetadata.get(widget.name)

      // Wrapper callback specific to Nodes 2.0 rendering
      const callback = (v: unknown) => {
        const value = normalizeWidgetValue(v)
        widget.value = value ?? undefined
        // Match litegraph callback signature: (value, canvas, node, pos, event)
        // Some extensions (e.g., Impact Pack) expect node as the 3rd parameter
        widget.callback?.(value, app.canvas, node)
        // Trigger redraw for all legacy widgets on this node (e.g., mask preview)
        // This ensures widgets that depend on other widget values get updated
        node.widgets?.forEach((w) => w.triggerDraw?.())
      }

      const isPromotedPseudoWidget =
        isPromotedWidgetView(widget) && widget.sourceWidgetName.startsWith('$$')

      // Extract only render-critical options (canvasOnly, advanced, read_only)
      const options = widget.options
        ? {
            canvasOnly: widget.options.canvasOnly,
            advanced: widget.advanced,
            hidden: widget.options.hidden,
            read_only: widget.options.read_only
          }
        : undefined
      const subgraphId = node.isSubgraphNode() && node.subgraph.id

      const localId = isPromotedWidgetView(widget)
        ? widget.sourceNodeId
        : undefined
      const nodeId =
        subgraphId && localId ? `${subgraphId}:${localId}` : undefined
      const name = isPromotedWidgetView(widget)
        ? widget.sourceWidgetName
        : widget.name

      return {
        nodeId,
        name,
        type: widget.type,
        ...sharedEnhancements,
        callback,
        hasLayoutSize: typeof widget.computeLayoutSize === 'function',
        isDOMWidget: isDOMWidget(widget) || isPromotedDOMWidget(widget),
        options: isPromotedPseudoWidget
          ? { ...options, canvasOnly: true }
          : options,
        slotMetadata: slotInfo,
        slotName: name !== widget.name ? widget.name : undefined
      }
    } catch (error) {
      return {
        name: widget.name || 'unknown',
        type: widget.type || 'text'
      }
    }
  }
}

// Extract safe data from LiteGraph node for Vue consumption
export function extractVueNodeData(node: LGraphNode): VueNodeData {
  // Determine subgraph ID - null for root graph, string for subgraphs
  const subgraphId =
    node.graph && 'id' in node.graph && node.graph !== node.graph.rootGraph
      ? String(node.graph.id)
      : null
  // Extract safe widget data
  const slotMetadata = new Map<string, WidgetSlotMetadata>()

  const existingWidgetsDescriptor = Object.getOwnPropertyDescriptor(
    node,
    'widgets'
  )
  const reactiveWidgets = shallowReactive<IBaseWidget[]>(node.widgets ?? [])
  if (existingWidgetsDescriptor?.get) {
    // Node has a custom widgets getter (e.g. SubgraphNode's synthetic getter).
    // Preserve it but sync results into a reactive array for Vue.
    const originalGetter = existingWidgetsDescriptor.get
    Object.defineProperty(node, 'widgets', {
      get() {
        const current: IBaseWidget[] = originalGetter.call(node) ?? []
        if (
          current.length !== reactiveWidgets.length ||
          current.some((w, i) => w !== reactiveWidgets[i])
        ) {
          reactiveWidgets.splice(0, reactiveWidgets.length, ...current)
        }
        return reactiveWidgets
      },
      set: existingWidgetsDescriptor.set ?? (() => {}),
      configurable: true,
      enumerable: true
    })
  } else {
    Object.defineProperty(node, 'widgets', {
      get() {
        return reactiveWidgets
      },
      set(v) {
        reactiveWidgets.splice(0, reactiveWidgets.length, ...v)
      }
    })
  }
  const reactiveInputs = shallowReactive<INodeInputSlot[]>(node.inputs ?? [])
  Object.defineProperty(node, 'inputs', {
    get() {
      return reactiveInputs
    },
    set(v) {
      reactiveInputs.splice(0, reactiveInputs.length, ...v)
    }
  })

  const safeWidgets = reactiveComputed<SafeWidgetData[]>(() => {
    node.inputs?.forEach((input, index) => {
      if (!input?.widget?.name) return
      slotMetadata.set(input.widget.name, {
        index,
        linked: input.link != null
      })
    })
    return node.widgets?.map(safeWidgetMapper(node, slotMetadata)) ?? []
  })

  const nodeType =
    node.type ||
    node.constructor?.comfyClass ||
    node.constructor?.title ||
    node.constructor?.name ||
    'Unknown'

  const apiNode = node.constructor?.nodeData?.api_node ?? false
  const badges = node.badges

  return {
    id: String(node.id),
    title: typeof node.title === 'string' ? node.title : '',
    type: nodeType,
    mode: node.mode || 0,
    titleMode: node.title_mode,
    selected: node.selected || false,
    executing: false, // Will be updated separately based on execution state
    subgraphId,
    apiNode,
    badges,
    hasErrors: !!node.has_errors,
    widgets: safeWidgets,
    inputs: reactiveInputs,
    outputs: node.outputs ? [...node.outputs] : undefined,
    flags: node.flags ? { ...node.flags } : undefined,
    color: node.color || undefined,
    bgcolor: node.bgcolor || undefined,
    resizable: node.resizable,
    shape: node.shape,
    showAdvanced: node.showAdvanced
  }
}

export function useGraphNodeManager(graph: LGraph): GraphNodeManager {
  // Get layout mutations composable
  const { createNode, deleteNode, setSource } = useLayoutMutations()
  // Safe reactive data extracted from LiteGraph nodes
  const vueNodeData = reactive(new Map<string, VueNodeData>())

  // Non-reactive storage for original LiteGraph nodes
  const nodeRefs = new Map<string, LGraphNode>()

  const refreshNodeSlots = (nodeId: string) => {
    const nodeRef = nodeRefs.get(nodeId)
    const currentData = vueNodeData.get(nodeId)

    if (!nodeRef || !currentData) return

    // Only extract slot-related data instead of full node re-extraction
    const slotMetadata = new Map<string, WidgetSlotMetadata>()

    nodeRef.inputs?.forEach((input, index) => {
      if (!input?.widget?.name) return
      slotMetadata.set(input.widget.name, {
        index,
        linked: input.link != null
      })
    })

    // Update only widgets with new slot metadata, keeping other widget data intact
    for (const widget of currentData.widgets ?? []) {
      const slotInfo = slotMetadata.get(widget.slotName ?? widget.name)
      if (slotInfo) widget.slotMetadata = slotInfo
    }
  }

  // Get access to original LiteGraph node (non-reactive)
  const getNode = (id: string): LGraphNode | undefined => {
    return nodeRefs.get(id)
  }

  const syncWithGraph = () => {
    if (!graph?._nodes) return

    const currentNodes = new Set(graph._nodes.map((n) => String(n.id)))

    // Remove deleted nodes
    for (const id of Array.from(vueNodeData.keys())) {
      if (!currentNodes.has(id)) {
        nodeRefs.delete(id)
        vueNodeData.delete(id)
      }
    }

    // Add/update existing nodes
    graph._nodes.forEach((node) => {
      const id = String(node.id)

      // Store non-reactive reference
      nodeRefs.set(id, node)

      // Extract and store safe data for Vue
      vueNodeData.set(id, extractVueNodeData(node))
    })
  }

  /**
   * Handles node addition to the graph - sets up Vue state and spatial indexing
   * Defers position extraction until after potential configure() calls
   */
  const handleNodeAdded = (
    node: LGraphNode,
    originalCallback?: (node: LGraphNode) => void
  ) => {
    const id = String(node.id)

    // Store non-reactive reference to original node
    nodeRefs.set(id, node)

    // Extract initial data for Vue (may be incomplete during graph configure)
    vueNodeData.set(id, extractVueNodeData(node))

    const initializeVueNodeLayout = () => {
      // Check if the node was removed mid-sequence
      if (!nodeRefs.has(id)) return

      // Extract actual positions after configure() has potentially updated them
      const nodePosition = { x: node.pos[0], y: node.pos[1] }
      const nodeSize = { width: node.size[0], height: node.size[1] }

      // Skip layout creation if it already exists
      // (e.g. in-place node replacement where the old node's layout is reused for the new node with the same ID).
      const existingLayout = layoutStore.getNodeLayoutRef(id).value
      if (existingLayout) return

      // Add node to layout store with final positions
      setSource(LayoutSource.Canvas)
      void createNode(id, {
        position: nodePosition,
        size: nodeSize,
        zIndex: node.order || 0,
        visible: true
      })
    }

    // Check if we're in the middle of configuring the graph (workflow loading)
    if (window.app?.configuringGraph) {
      // During workflow loading - defer layout initialization until configure completes
      // Chain our callback with any existing onAfterGraphConfigured callback
      node.onAfterGraphConfigured = useChainCallback(
        node.onAfterGraphConfigured,
        () => {
          // Re-extract data now that configure() has populated title/slots/widgets/etc.
          vueNodeData.set(id, extractVueNodeData(node))
          initializeVueNodeLayout()
        }
      )
    } else {
      // Not during workflow loading - initialize layout immediately
      // This handles individual node additions during normal operation
      initializeVueNodeLayout()
    }

    // Call original callback if provided
    if (originalCallback) {
      void originalCallback(node)
    }
  }

  /**
   * Handles node removal from the graph - cleans up all references
   */
  const handleNodeRemoved = (
    node: LGraphNode,
    originalCallback?: (node: LGraphNode) => void
  ) => {
    const id = String(node.id)

    // Remove node from layout store
    setSource(LayoutSource.Canvas)
    void deleteNode(id)

    // Clean up all tracking references
    nodeRefs.delete(id)
    vueNodeData.delete(id)

    // Call original callback if provided
    if (originalCallback) {
      originalCallback(node)
    }
  }

  /**
   * Creates cleanup function for event listeners and state
   */
  const createCleanupFunction = (
    originalOnNodeAdded: ((node: LGraphNode) => void) | undefined,
    originalOnNodeRemoved: ((node: LGraphNode) => void) | undefined,
    originalOnTrigger: ((event: LGraphTriggerEvent) => void) | undefined
  ) => {
    return () => {
      // Restore original callbacks
      graph.onNodeAdded = originalOnNodeAdded || undefined
      graph.onNodeRemoved = originalOnNodeRemoved || undefined
      graph.onTrigger = originalOnTrigger || undefined

      // Clear all state maps
      nodeRefs.clear()
      vueNodeData.clear()
    }
  }

  /**
   * Sets up event listeners - now simplified with extracted handlers
   */
  const setupEventListeners = (): (() => void) => {
    // Store original callbacks
    const originalOnNodeAdded = graph.onNodeAdded
    const originalOnNodeRemoved = graph.onNodeRemoved
    const originalOnTrigger = graph.onTrigger

    // Set up graph event handlers
    graph.onNodeAdded = (node: LGraphNode) => {
      handleNodeAdded(node, originalOnNodeAdded)
    }

    graph.onNodeRemoved = (node: LGraphNode) => {
      handleNodeRemoved(node, originalOnNodeRemoved)
    }

    const triggerHandlers: {
      [K in LGraphTriggerAction]: (event: LGraphTriggerParam<K>) => void
    } = {
      'node:property:changed': (propertyEvent) => {
        const nodeId = String(propertyEvent.nodeId)
        const currentData = vueNodeData.get(nodeId)

        if (currentData) {
          switch (propertyEvent.property) {
            case 'title':
              vueNodeData.set(nodeId, {
                ...currentData,
                title: String(propertyEvent.newValue)
              })
              break
            case 'flags.collapsed':
              vueNodeData.set(nodeId, {
                ...currentData,
                flags: {
                  ...currentData.flags,
                  collapsed: Boolean(propertyEvent.newValue)
                }
              })
              break
            case 'flags.ghost':
              vueNodeData.set(nodeId, {
                ...currentData,
                flags: {
                  ...currentData.flags,
                  ghost: Boolean(propertyEvent.newValue)
                }
              })
              break
            case 'flags.pinned':
              vueNodeData.set(nodeId, {
                ...currentData,
                flags: {
                  ...currentData.flags,
                  pinned: Boolean(propertyEvent.newValue)
                }
              })
              break
            case 'mode':
              vueNodeData.set(nodeId, {
                ...currentData,
                mode:
                  typeof propertyEvent.newValue === 'number'
                    ? propertyEvent.newValue
                    : 0
              })
              break
            case 'color':
              vueNodeData.set(nodeId, {
                ...currentData,
                color:
                  typeof propertyEvent.newValue === 'string'
                    ? propertyEvent.newValue
                    : undefined
              })
              break
            case 'bgcolor':
              vueNodeData.set(nodeId, {
                ...currentData,
                bgcolor:
                  typeof propertyEvent.newValue === 'string'
                    ? propertyEvent.newValue
                    : undefined
              })
              break
            case 'shape':
              vueNodeData.set(nodeId, {
                ...currentData,
                shape:
                  typeof propertyEvent.newValue === 'number'
                    ? propertyEvent.newValue
                    : undefined
              })
              break
            case 'showAdvanced':
              vueNodeData.set(nodeId, {
                ...currentData,
                showAdvanced: Boolean(propertyEvent.newValue)
              })
              break
          }
        }
      },
      'node:slot-errors:changed': (slotErrorsEvent) => {
        refreshNodeSlots(String(slotErrorsEvent.nodeId))
      },
      'node:slot-links:changed': (slotLinksEvent) => {
        if (slotLinksEvent.slotType === NodeSlotType.INPUT) {
          refreshNodeSlots(String(slotLinksEvent.nodeId))
        }
      }
    }

    graph.onTrigger = (event: LGraphTriggerEvent) => {
      switch (event.type) {
        case 'node:property:changed':
          triggerHandlers['node:property:changed'](event)
          break
        case 'node:slot-errors:changed':
          triggerHandlers['node:slot-errors:changed'](event)
          break
        case 'node:slot-links:changed':
          triggerHandlers['node:slot-links:changed'](event)
          break
      }

      // Chain to original handler
      originalOnTrigger?.(event)
    }

    // Initialize state
    syncWithGraph()

    // Return cleanup function
    return createCleanupFunction(
      originalOnNodeAdded || undefined,
      originalOnNodeRemoved || undefined,
      originalOnTrigger || undefined
    )
  }

  // Set up event listeners immediately
  const cleanup = setupEventListeners()

  // Process any existing nodes after event listeners are set up
  if (graph._nodes && graph._nodes.length > 0) {
    graph._nodes.forEach((node: LGraphNode) => {
      if (graph.onNodeAdded) {
        graph.onNodeAdded(node)
      }
    })
  }

  return {
    vueNodeData,
    getNode,
    cleanup
  }
}
