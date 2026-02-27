import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, nextTick, ref } from 'vue'

import { getSlotKey } from '@/renderer/core/layout/slots/slotIdentifier'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import { LayoutSource } from '@/renderer/core/layout/types'
import type { SlotLayout } from '@/renderer/core/layout/types'
import { useNodeSlotRegistryStore } from '@/renderer/extensions/vueNodes/stores/nodeSlotRegistryStore'

import {
  flushScheduledSlotLayoutSync,
  useSlotElementTracking
} from './useSlotElementTracking'

const mockGraph = vi.hoisted(() => ({ _nodes: [] as unknown[] }))

vi.mock('@/scripts/app', () => ({
  app: { canvas: { graph: mockGraph, setDirty: vi.fn() } }
}))

vi.mock('@/composables/element/useCanvasPositionConversion', () => ({
  useSharedCanvasPositionConversion: () => ({
    clientPosToCanvasPos: ([x, y]: [number, number]) => [x, y]
  })
}))

const NODE_ID = 'test-node'
const SLOT_INDEX = 0

function createWrapperComponent(type: 'input' | 'output') {
  return defineComponent({
    setup() {
      const el = ref<HTMLElement | null>(null)
      useSlotElementTracking({
        nodeId: NODE_ID,
        index: SLOT_INDEX,
        type,
        element: el
      })
      return { el }
    },
    // No template ref — el starts null so the immediate watcher doesn't fire
    // before the stop handle is assigned
    template: '<div />'
  })
}

/**
 * Mount the wrapper, set the element ref, and wait for slot registration.
 */
async function mountAndRegisterSlot(type: 'input' | 'output') {
  const wrapper = mount(createWrapperComponent(type))
  wrapper.vm.el = document.createElement('div')
  await nextTick()
  flushScheduledSlotLayoutSync()
  return wrapper
}

describe('useSlotElementTracking', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    layoutStore.initializeFromLiteGraph([])
    layoutStore.applyOperation({
      type: 'createNode',
      entity: 'node',
      nodeId: NODE_ID,
      layout: {
        id: NODE_ID,
        position: { x: 0, y: 0 },
        size: { width: 200, height: 100 },
        zIndex: 0,
        visible: true,
        bounds: { x: 0, y: 0, width: 200, height: 100 }
      },
      timestamp: Date.now(),
      source: LayoutSource.External,
      actor: 'test'
    })
    mockGraph._nodes = [{ id: 1 }]
  })

  it.each([
    { type: 'input' as const, isInput: true },
    { type: 'output' as const, isInput: false }
  ])('cleans up $type slot layout on unmount', async ({ type, isInput }) => {
    const wrapper = await mountAndRegisterSlot(type)

    const slotKey = getSlotKey(NODE_ID, SLOT_INDEX, isInput)
    const registryStore = useNodeSlotRegistryStore()
    expect(registryStore.getNode(NODE_ID)?.slots.has(slotKey)).toBe(true)
    expect(layoutStore.getSlotLayout(slotKey)).not.toBeNull()

    wrapper.unmount()

    expect(layoutStore.getSlotLayout(slotKey)).toBeNull()
    expect(registryStore.getNode(NODE_ID)).toBeUndefined()
  })

  it('clears pendingSlotSync when slot layouts already exist', () => {
    // Seed a slot layout (simulates slot layouts persisting through undo/redo)
    const slotKey = getSlotKey(NODE_ID, SLOT_INDEX, true)
    const slotLayout: SlotLayout = {
      nodeId: NODE_ID,
      index: 0,
      type: 'input',
      position: { x: 0, y: 0 },
      bounds: { x: 0, y: 0, width: 10, height: 10 }
    }
    layoutStore.batchUpdateSlotLayouts([{ key: slotKey, layout: slotLayout }])

    // Simulate what app.ts onConfigure does: set pending, then flush
    layoutStore.setPendingSlotSync(true)
    expect(layoutStore.pendingSlotSync).toBe(true)

    // No slots were scheduled (undo/redo — onMounted didn't fire),
    // but slot layouts already exist. Flush should clear the flag.
    flushScheduledSlotLayoutSync()

    expect(layoutStore.pendingSlotSync).toBe(false)
  })

  it('keeps pendingSlotSync when graph has nodes but no slot layouts', () => {
    // No slot layouts exist (simulates initial mount before Vue registers slots)
    layoutStore.setPendingSlotSync(true)

    flushScheduledSlotLayoutSync()

    // Should remain pending — waiting for Vue components to mount
    expect(layoutStore.pendingSlotSync).toBe(true)
  })
})
