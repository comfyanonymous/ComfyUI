import type { LGraphCanvas } from '@/lib/litegraph/src/LGraphCanvas'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'

export function augmentToCanvasPointerEvent(
  e: PointerEvent,
  node: LGraphNode,
  canvas: LGraphCanvas
): asserts e is CanvasPointerEvent {
  canvas.adjustMouseEvent(e)
  canvas.graph_mouse[0] = e.offsetX + node.pos[0]
  canvas.graph_mouse[1] = e.offsetY + node.pos[1]
}
