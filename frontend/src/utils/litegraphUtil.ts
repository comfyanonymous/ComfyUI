import _ from 'es-toolkit/compat'

import type { ColorOption, LGraph } from '@/lib/litegraph/src/litegraph'
import type { ExecutedWsMessage } from '@/schemas/apiSchema'
import {
  LGraphCanvas,
  LGraphGroup,
  LGraphNode,
  LiteGraph,
  Reroute,
  isColorable
} from '@/lib/litegraph/src/litegraph'
import type {
  ExportedSubgraph,
  ISerialisableNodeInput,
  ISerialisedGraph
} from '@/lib/litegraph/src/types/serialisation'
import type {
  IBaseWidget,
  IComboWidget,
  WidgetCallbackOptions
} from '@/lib/litegraph/src/types/widgets'
import type { InputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { t } from '@/i18n'

type ImageNode = LGraphNode & { imgs: HTMLImageElement[] | undefined }
type VideoNode = LGraphNode & {
  videoContainer: HTMLElement | undefined
  imgs: HTMLVideoElement[] | undefined
}

/**
 * Extract & Promisify Litegraph.createNode to allow for positioning
 * @param canvas
 * @param name
 */
export async function createNode(
  canvas: LGraphCanvas,
  name: string
): Promise<LGraphNode | null> {
  if (!name) {
    return null
  }

  const {
    graph,
    graph_mouse: [posX, posY]
  } = canvas
  const newNode = LiteGraph.createNode(name)
  await new Promise((r) => setTimeout(r, 0))

  if (newNode && graph) {
    newNode.pos = [posX, posY]
    const addedNode = graph.add(newNode) ?? null

    if (addedNode) graph.change()
    return addedNode
  } else {
    useToastStore().addAlert(t('assetBrowser.failedToCreateNode'))
    return null
  }
}

export function isImageNode(node: LGraphNode | undefined): node is ImageNode {
  if (!node) return false
  return (
    node.previewMediaType === 'image' ||
    (node.previewMediaType !== 'video' && !!node.imgs?.length)
  )
}

export function isVideoNode(node: LGraphNode | undefined): node is VideoNode {
  if (!node) return false
  return node.previewMediaType === 'video' || !!node.videoContainer
}

/**
 * Check if output data indicates animated content (animated webp/png or video).
 */
export function isAnimatedOutput(
  output: ExecutedWsMessage['output'] | undefined
): boolean {
  return !!output?.animated?.find(Boolean)
}

/**
 * Check if output data indicates video content (animated but not webp/png).
 */
export function isVideoOutput(
  output: ExecutedWsMessage['output'] | undefined
): boolean {
  if (!isAnimatedOutput(output)) return false

  const isAnimatedWebp = output?.images?.some((img) =>
    img.filename?.endsWith('.webp')
  )
  const isAnimatedPng = output?.images?.some((img) =>
    img.filename?.endsWith('.png')
  )
  return !isAnimatedWebp && !isAnimatedPng
}

export function isAudioNode(node: LGraphNode | undefined): boolean {
  return !!node && node.previewMediaType === 'audio'
}

export function addToComboValues(widget: IComboWidget, value: string) {
  if (!widget.options) widget.options = { values: [] }
  if (!widget.options.values) widget.options.values = []
  // @ts-expect-error Combo widget values may be a dictionary or legacy function type
  if (!widget.options.values.includes(value)) {
    // @ts-expect-error Combo widget values may be a dictionary or legacy function type
    widget.options.values.push(value)
  }
}

export const isLGraphNode = (item: unknown): item is LGraphNode => {
  return item instanceof LGraphNode
}

export const isLGraphGroup = (item: unknown): item is LGraphGroup => {
  return item instanceof LGraphGroup
}

export const isReroute = (item: unknown): item is Reroute => {
  return item instanceof Reroute
}

/**
 * Get the color option of all canvas items if they are all the same.
 * @param items - The items to get the color option of.
 * @returns The color option of the item.
 */
export const getItemsColorOption = (items: unknown[]): ColorOption | null => {
  const validItems = _.filter(items, isColorable)
  if (_.isEmpty(validItems)) return null

  const colorOptions = _.map(validItems, (item) => item.getColorOption())

  return _.every(colorOptions, (option) =>
    _.isEqual(option, _.head(colorOptions))
  )
    ? _.head(colorOptions)!
    : null
}

export function executeWidgetsCallback(
  nodes: LGraphNode[],
  callbackName: 'onRemove' | 'beforeQueued' | 'afterQueued',
  options?: WidgetCallbackOptions
) {
  for (const node of nodes) {
    for (const widget of node.widgets ?? []) {
      widget[callbackName]?.(options)
    }
  }
}

/**
 * Since frontend version 1.16, forceInput input is no longer treated
 * as widget. So we need to remove the dummy widget value serialized
 * from workflows prior to v1.16.
 * Ref: https://github.com/Comfy-Org/ComfyUI_frontend/pull/3326
 *
 * @param nodeDef the node definition
 * @param widgets the widgets on the node instance (from node definition)
 * @param widgetsValues the widgets values to populate the node during configuration
 * @returns the widgets values without the dummy widget values
 */
export function migrateWidgetsValues<TWidgetValue>(
  inputDefs: Record<string, InputSpec>,
  widgets: IBaseWidget[],
  widgetsValues: TWidgetValue[]
): TWidgetValue[] {
  const widgetNames = new Set(widgets.map((w) => w.name))
  const originalWidgetsInputs = Object.values(inputDefs).filter(
    (input) => widgetNames.has(input.name) || input.forceInput
  )

  const widgetIndexHasForceInput = originalWidgetsInputs.flatMap((input) =>
    input.control_after_generate
      ? [!!input.forceInput, false]
      : [!!input.forceInput]
  )

  if (widgetIndexHasForceInput.length !== widgetsValues?.length)
    return widgetsValues

  return widgetsValues.filter((_, index) => !widgetIndexHasForceInput[index])
}

/**
 * Fix link input slots after loading a graph. Because the node inputs follows
 * the node definition after 1.16, the node inputs array from previous versions,
 * might get added items in the middle, which can cause shift to link's slot index.
 * For example, the node inputs definition is:
 * "required": {
 *   "input1": ["INT", { forceInput: true }],
 *   "input2": ["MODEL", { forceInput: false }],
 *   "input3": ["MODEL", { forceInput: false }]
 * }
 *
 * previously node inputs array was:
 * [{name: 'input2'}, {name: 'input3'}, {name: 'input1'}]
 * because input1 is created as widget first, then convert to input socket after
 * input 2 and 3.
 *
 * Now, the node inputs array just follows the definition order:
 * [{name: 'input1'}, {name: 'input2'}, {name: 'input3'}]
 *
 * We need to update the slot index of corresponding links to match the new
 * node inputs array order.
 *
 * Ref: https://github.com/Comfy-Org/ComfyUI_frontend/issues/3348
 *
 * @param graph - The graph to fix links for.
 */
export function fixLinkInputSlots(graph: LGraph) {
  // Note: We can't use forEachNode here because we need access to the graph's
  // links map at each level. Links are stored in their respective graph/subgraph.
  for (const node of graph.nodes) {
    // Fix links for the current node
    for (const [inputIndex, input] of node.inputs.entries()) {
      const linkId = input.link
      if (!linkId) continue

      const link = graph.links.get(linkId)
      if (!link) continue

      link.target_slot = inputIndex
    }

    // Recursively fix links in subgraphs
    if (node.isSubgraphNode?.() && node.subgraph) {
      fixLinkInputSlots(node.subgraph)
    }
  }
}

/**
 * Compress widget input slots by removing all unconnected widget input slots.
 * This should match the serialization format of legacy widget conversion.
 *
 * @param graph - The graph to compress widget input slots for.
 * @throws If an infinite loop is detected.
 */
export function compressWidgetInputSlots(graph: ISerialisedGraph) {
  for (const node of graph.nodes) {
    node.inputs = node.inputs?.filter(matchesLegacyApi)

    for (const [inputIndex, input] of node.inputs?.entries() ?? []) {
      if (input.link) {
        const link = graph.links.find((link) => link[0] === input.link)
        if (link) {
          link[4] = inputIndex
        }
      }
    }
  }

  compressSubgraphWidgetInputSlots(graph.definitions?.subgraphs)
}

function matchesLegacyApi(input: ISerialisableNodeInput) {
  return !(input.widget && input.link === null && !input.label)
}

/**
 * Duplication to handle the legacy link arrays in the root workflow.
 * @see compressWidgetInputSlots
 * @param subgraph The subgraph to compress widget input slots for.
 */
function compressSubgraphWidgetInputSlots(
  subgraphs: ExportedSubgraph[] | undefined,
  visited = new WeakSet<ExportedSubgraph>()
) {
  if (!subgraphs) return

  for (const subgraph of subgraphs) {
    if (visited.has(subgraph)) throw new Error('Infinite loop detected')
    visited.add(subgraph)

    if (subgraph.nodes) {
      for (const node of subgraph.nodes) {
        node.inputs = node.inputs?.filter(matchesLegacyApi)

        if (!subgraph.links) continue

        for (const [inputIndex, input] of node.inputs?.entries() ?? []) {
          if (input.link) {
            const link = subgraph.links.find((link) => link.id === input.link)
            if (link) link.target_slot = inputIndex
          }
        }
      }
    }

    compressSubgraphWidgetInputSlots(subgraph.definitions?.subgraphs, visited)
  }
}

export function getLinkTypeColor(typeName: string): string {
  return LGraphCanvas.link_type_colors[typeName] ?? LiteGraph.LINK_COLOR
}

export function isLoad3dNode(node: LGraphNode) {
  return (
    node &&
    node.type &&
    (node.type === 'Load3D' || node.type === 'Load3DAnimation')
  )
}
