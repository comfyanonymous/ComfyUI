/**
 * Slot Position Calculations
 *
 * Centralized utility for calculating input/output slot positions on nodes.
 * This allows both litegraph nodes and the layout system to use the same
 * calculation logic while providing their own position data.
 */
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type {
  INodeInputSlot,
  INodeOutputSlot,
  Point
} from '@/lib/litegraph/src/interfaces'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import { isWidgetInputSlot } from '@/lib/litegraph/src/node/slotUtils'
import { getSlotKey } from '@/renderer/core/layout/slots/slotIdentifier'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'

export interface SlotPositionContext {
  /** Node's X position in graph coordinates */
  nodeX: number
  /** Node's Y position in graph coordinates */
  nodeY: number
  /** Node's width */
  nodeWidth: number
  /** Node's height */
  nodeHeight: number
  /** Whether the node is collapsed */
  collapsed: boolean
  /** Collapsed width (if applicable) */
  collapsedWidth?: number
  /** Node constructor's slot_start_y offset */
  slotStartY?: number
  /** Node's input slots */
  inputs: INodeInputSlot[]
  /** Node's output slots */
  outputs: INodeOutputSlot[]
  /** Node's widgets (for widget slot detection) */
  widgets?: Array<{ name?: string }>
}

/**
 * Calculate the position of an input slot in graph coordinates
 * @param context Node context containing position and slot data
 * @param slot The input slot index
 * @returns Position of the input slot center in graph coordinates
 */
function calculateInputSlotPos(
  context: SlotPositionContext,
  slot: number
): Point {
  const input = context.inputs[slot]
  if (!input) return [context.nodeX, context.nodeY]

  return calculateInputSlotPosFromSlot(context, input)
}

/**
 * Calculate the position of an input slot in graph coordinates
 * @param context Node context containing position and slot data
 * @param input The input slot object
 * @returns Position of the input slot center in graph coordinates
 */
export function calculateInputSlotPosFromSlot(
  context: SlotPositionContext,
  input: INodeInputSlot
): Point {
  const { nodeX, nodeY, collapsed } = context

  // Handle collapsed nodes
  if (collapsed) {
    const halfTitle = LiteGraph.NODE_TITLE_HEIGHT * 0.5
    return [nodeX, nodeY - halfTitle]
  }

  // Handle hard-coded positions
  const { pos } = input
  if (pos) return [nodeX + pos[0], nodeY + pos[1]]

  // Default vertical slots
  const offsetX = LiteGraph.NODE_SLOT_HEIGHT * 0.5
  const nodeOffsetY = context.slotStartY || 0
  const defaultVerticalInputs = getDefaultVerticalInputs(context)
  const slotIndex = defaultVerticalInputs.indexOf(input)
  const slotY = (slotIndex + 0.7) * LiteGraph.NODE_SLOT_HEIGHT

  return [nodeX + offsetX, nodeY + slotY + nodeOffsetY]
}

/**
 * Calculate the position of an output slot in graph coordinates
 * @param context Node context containing position and slot data
 * @param slot The output slot index
 * @returns Position of the output slot center in graph coordinates
 */
function calculateOutputSlotPos(
  context: SlotPositionContext,
  slot: number
): Point {
  const { nodeX, nodeY, nodeWidth, collapsed, collapsedWidth, outputs } =
    context

  // Handle collapsed nodes
  if (collapsed) {
    const width = collapsedWidth || LiteGraph.NODE_COLLAPSED_WIDTH
    const halfTitle = LiteGraph.NODE_TITLE_HEIGHT * 0.5
    return [nodeX + width, nodeY - halfTitle]
  }

  const outputSlot = outputs[slot]
  if (!outputSlot) return [nodeX + nodeWidth, nodeY]

  // Handle hard-coded positions
  const outputPos = outputSlot.pos
  if (outputPos) return [nodeX + outputPos[0], nodeY + outputPos[1]]

  // Default vertical slots
  const offsetX = LiteGraph.NODE_SLOT_HEIGHT * 0.5
  const nodeOffsetY = context.slotStartY || 0
  const defaultVerticalOutputs = getDefaultVerticalOutputs(context)
  const slotIndex = defaultVerticalOutputs.indexOf(outputSlot)
  const slotY = (slotIndex + 0.7) * LiteGraph.NODE_SLOT_HEIGHT

  // TODO: Why +1?
  return [nodeX + nodeWidth + 1 - offsetX, nodeY + slotY + nodeOffsetY]
}

/**
 * Get slot position using layout tree if available, fallback to node's position
 * Unified implementation used by both LitegraphLinkAdapter and useLinkLayoutSync
 * @param node The LGraphNode
 * @param slotIndex The slot index
 * @param isInput Whether this is an input slot
 * @returns Position of the slot center in graph coordinates
 */
export function getSlotPosition(
  node: LGraphNode,
  slotIndex: number,
  isInput: boolean
): Point {
  // Only use DOM-registered slot positions when Vue nodes mode is enabled
  if (LiteGraph.vueNodesMode) {
    // Try to get precise position from slot layout (DOM-registered)
    const slotKey = getSlotKey(String(node.id), slotIndex, isInput)
    const slotLayout = layoutStore.getSlotLayout(slotKey)
    if (slotLayout) {
      return [slotLayout.position.x, slotLayout.position.y]
    }

    // Fallback: derive position from node layout tree and slot model
    const nodeLayout = layoutStore.getNodeLayoutRef(String(node.id)).value

    if (nodeLayout) {
      // Create context from layout tree data
      const context: SlotPositionContext = {
        nodeX: nodeLayout.position.x,
        nodeY: nodeLayout.position.y,
        nodeWidth: nodeLayout.size.width,
        nodeHeight: nodeLayout.size.height,
        collapsed: node.flags.collapsed || false,
        collapsedWidth: node._collapsed_width,
        slotStartY: node.constructor.slot_start_y,
        inputs: node.inputs,
        outputs: node.outputs,
        widgets: node.widgets
      }

      // Use helper to calculate position
      return isInput
        ? calculateInputSlotPos(context, slotIndex)
        : calculateOutputSlotPos(context, slotIndex)
    }
  }

  // Fallback: calculate directly from node properties (legacy litegraph behavior)
  const context: SlotPositionContext = {
    nodeX: node.pos[0],
    nodeY: node.pos[1],
    nodeWidth: node.size[0],
    nodeHeight: node.size[1],
    collapsed: node.flags.collapsed || false,
    collapsedWidth: node._collapsed_width,
    slotStartY: node.constructor.slot_start_y,
    inputs: node.inputs,
    outputs: node.outputs,
    widgets: node.widgets
  }

  return isInput
    ? calculateInputSlotPos(context, slotIndex)
    : calculateOutputSlotPos(context, slotIndex)
}

/**
 * Get the inputs that are not positioned with absolute coordinates
 */
function getDefaultVerticalInputs(
  context: SlotPositionContext
): INodeInputSlot[] {
  return context.inputs.filter(
    (slot) => !slot.pos && !(context.widgets?.length && isWidgetInputSlot(slot))
  )
}

/**
 * Get the outputs that are not positioned with absolute coordinates
 */
function getDefaultVerticalOutputs(
  context: SlotPositionContext
): INodeOutputSlot[] {
  return context.outputs.filter((slot) => !slot.pos)
}
