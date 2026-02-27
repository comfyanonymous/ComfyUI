import type { NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { SlotLayout } from '@/renderer/core/layout/types'

/**
 * Slot link drag context
 *
 * Non-reactive, per-drag ephemeral caches and RAF batching used during
 * link drag interactions. Keeps high-churn data out of the reactive UI state.
 */

interface PendingPointerMoveData {
  clientX: number
  clientY: number
  target: EventTarget | null
}

export interface SlotLinkDragContext {
  preferredSlotForNode: Map<
    NodeId,
    { index: number; key: string; layout: SlotLayout } | null
  >
  lastHoverSlotKey: string | null
  lastHoverNodeId: NodeId | null
  lastCandidateKey: string | null
  pendingPointerMove: PendingPointerMoveData | null
  lastPointerEventTarget: EventTarget | null
  lastPointerTargetSlotKey: string | null
  lastPointerTargetNodeId: NodeId | null
  reset: () => void
  dispose: () => void
}

export function createSlotLinkDragContext(): SlotLinkDragContext {
  const state: SlotLinkDragContext = {
    preferredSlotForNode: new Map(),
    lastHoverSlotKey: null,
    lastHoverNodeId: null,
    lastCandidateKey: null,
    pendingPointerMove: null,
    lastPointerEventTarget: null,
    lastPointerTargetSlotKey: null,
    lastPointerTargetNodeId: null,
    reset: () => {
      state.preferredSlotForNode = new Map()
      state.lastHoverSlotKey = null
      state.lastHoverNodeId = null
      state.lastCandidateKey = null
      state.pendingPointerMove = null
      state.lastPointerEventTarget = null
      state.lastPointerTargetSlotKey = null
      state.lastPointerTargetNodeId = null
    },
    dispose: () => {
      state.reset()
    }
  }

  return state
}
