import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type {
  INodeInputSlot,
  INodeOutputSlot,
  Point
} from '@/lib/litegraph/src/interfaces'
import { isInRectangle } from '@/lib/litegraph/src/measure'

export function getNodeInputOnPos(
  node: LGraphNode,
  x: number,
  y: number
): { index: number; input: INodeInputSlot; pos: Point } | undefined {
  const { inputs } = node
  if (!inputs) return

  for (const [index, input] of inputs.entries()) {
    const pos = node.getInputPos(index)

    // TODO: Find a cheap way to measure text, and do it on node label change instead of here
    // Input icon width + text approximation
    const nameLength =
      input.label?.length ?? input.localized_name?.length ?? input.name?.length
    const width = 20 + (nameLength || 3) * 7

    if (isInRectangle(x, y, pos[0] - 10, pos[1] - 10, width, 20)) {
      return { index, input, pos }
    }
  }
}

export function getNodeOutputOnPos(
  node: LGraphNode,
  x: number,
  y: number
): { index: number; output: INodeOutputSlot; pos: Point } | undefined {
  const { outputs } = node
  if (!outputs) return

  for (const [index, output] of outputs.entries()) {
    const pos = node.getOutputPos(index)

    if (isInRectangle(x, y, pos[0] - 10, pos[1] - 10, 40, 20)) {
      return { index, output, pos }
    }
  }
}

/**
 * Returns the input slot index if the given position (in graph space) is on top of a node input slot.
 * A helper function - originally on the prototype of LGraphCanvas.
 */
export function isOverNodeInput(
  node: LGraphNode,
  canvasx: number,
  canvasy: number,
  slot_pos?: Point
): number {
  const result = getNodeInputOnPos(node, canvasx, canvasy)
  if (!result) return -1

  if (slot_pos) {
    slot_pos[0] = result.pos[0]
    slot_pos[1] = result.pos[1]
  }
  return result.index
}

/**
 * Returns the output slot index if the given position (in graph space) is on top of a node output slot.
 * A helper function - originally on the prototype of LGraphCanvas.
 */
export function isOverNodeOutput(
  node: LGraphNode,
  canvasx: number,
  canvasy: number,
  slot_pos?: Point
): number {
  const result = getNodeOutputOnPos(node, canvasx, canvasy)
  if (!result) return -1

  if (slot_pos) {
    slot_pos[0] = result.pos[0]
    slot_pos[1] = result.pos[1]
  }
  return result.index
}
