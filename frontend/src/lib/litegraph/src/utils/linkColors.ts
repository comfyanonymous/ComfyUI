import type { CanvasColour, ISlotType } from '../interfaces'
import { LiteGraph } from '../litegraph'

/**
 * Resolve the colour used while rendering or previewing a connection of a given slot type.
 */
export function resolveConnectingLinkColor(
  type: ISlotType | undefined
): CanvasColour {
  return type === LiteGraph.EVENT
    ? LiteGraph.EVENT_LINK_COLOR
    : LiteGraph.CONNECTING_LINK_COLOR
}
