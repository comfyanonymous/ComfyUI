import { reactive, readonly } from 'vue'

import type { LinkDirection } from '@/lib/litegraph/src/types/globalEnums'
import { getSlotKey } from '@/renderer/core/layout/slots/slotIdentifier'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import type { Point, SlotLayout } from '@/renderer/core/layout/types'

/**
 * Slot link drag UI state
 *
 * Reactive, shared state for a single drag interaction that UI components subscribe to.
 * Tracks pointer position, source slot, and resolved drop candidate. Also exposes
 * a compatibility map used to dim incompatible slots during drag.
 */

type SlotDragType = 'input' | 'output'

interface SlotDragSource {
  nodeId: string
  slotIndex: number
  type: SlotDragType
  direction: LinkDirection
  position: Readonly<Point>
  linkId?: number
  movingExistingOutput?: boolean
}

export interface SlotDropCandidate {
  layout: SlotLayout
  compatible: boolean
}

interface PointerPosition {
  client: Point
  canvas: Point
}

interface SlotDragState {
  active: boolean
  pointerId: number | null
  source: SlotDragSource | null
  pointer: PointerPosition
  candidate: SlotDropCandidate | null
  compatible: Map<string, boolean>
}

const state = reactive<SlotDragState>({
  active: false,
  pointerId: null,
  source: null,
  pointer: {
    client: { x: 0, y: 0 },
    canvas: { x: 0, y: 0 }
  },
  candidate: null,
  compatible: new Map<string, boolean>()
})

function updatePointerPosition(
  clientX: number,
  clientY: number,
  canvasX: number,
  canvasY: number
) {
  state.pointer.client.x = clientX
  state.pointer.client.y = clientY
  state.pointer.canvas.x = canvasX
  state.pointer.canvas.y = canvasY
}

function setCandidate(candidate: SlotDropCandidate | null) {
  state.candidate = candidate
}

function beginDrag(source: SlotDragSource, pointerId: number) {
  state.active = true
  state.source = source
  state.pointerId = pointerId
  state.candidate = null
  state.compatible.clear()
}

function endDrag() {
  state.active = false
  state.pointerId = null
  state.source = null
  state.pointer.client.x = 0
  state.pointer.client.y = 0
  state.pointer.canvas.x = 0
  state.pointer.canvas.y = 0
  state.candidate = null
  state.compatible.clear()
}

function getSlotLayout(nodeId: string, slotIndex: number, isInput: boolean) {
  const slotKey = getSlotKey(nodeId, slotIndex, isInput)
  return layoutStore.getSlotLayout(slotKey)
}

export function useSlotLinkDragUIState() {
  return {
    state: readonly(state),
    beginDrag,
    endDrag,
    updatePointerPosition,
    setCandidate,
    getSlotLayout,
    setCompatibleMap: (entries: Iterable<[string, boolean]>) => {
      state.compatible.clear()
      for (const [key, value] of entries) state.compatible.set(key, value)
    },
    setCompatibleForKey: (key: string, value: boolean) => {
      state.compatible.set(key, value)
    },
    clearCompatible: () => state.compatible.clear()
  }
}
