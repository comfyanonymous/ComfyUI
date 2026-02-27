import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, ref } from 'vue'

import { useNodePointerInteractions } from '@/renderer/extensions/vueNodes/composables/useNodePointerInteractions'
import { useNodeEventHandlers } from '@/renderer/extensions/vueNodes/composables/useNodeEventHandlers'
import { createTestingPinia } from '@pinia/testing'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import type { NodeLayout } from '@/renderer/core/layout/types'
import { useNodeDrag } from '@/renderer/extensions/vueNodes/layout/useNodeDrag'

const forwardEventToCanvasMock = vi.fn()
const selectedItemsState: { items: Array<{ id?: string }> } = { items: [] }

// Mock the dependencies
vi.mock('@/renderer/core/canvas/useCanvasInteractions', () => ({
  useCanvasInteractions: () => ({
    forwardEventToCanvas: forwardEventToCanvasMock,
    shouldHandleNodePointerEvents: ref(true)
  })
}))

vi.mock('@/renderer/extensions/vueNodes/layout/useNodeDrag', () => {
  const startDrag = vi.fn()
  const handleDrag = vi.fn()
  const endDrag = vi.fn()
  return {
    useNodeDrag: () => ({
      startDrag,
      handleDrag,
      endDrag
    })
  }
})

vi.mock('@/renderer/core/canvas/canvasStore', () => ({
  useCanvasStore: () => ({
    get selectedItems() {
      return selectedItemsState.items
    }
  })
}))

vi.mock(
  '@/renderer/extensions/vueNodes/composables/useNodeEventHandlers',
  () => {
    const handleNodeSelect = vi.fn()
    const deselectNode = vi.fn()
    const selectNodes = vi.fn()
    const toggleNodeSelectionAfterPointerUp = vi.fn()
    const ensureNodeSelectedForShiftDrag = vi.fn()

    return {
      useNodeEventHandlers: () => ({
        handleNodeSelect,
        deselectNode,
        selectNodes,
        toggleNodeSelectionAfterPointerUp,
        ensureNodeSelectedForShiftDrag
      })
    }
  }
)

vi.mock('@/composables/graph/useVueNodeLifecycle', () => ({
  useVueNodeLifecycle: () => ({
    nodeManager: ref({
      getNode: vi.fn((id: string) => ({
        id,
        selected: false // Default to not selected
      }))
    })
  })
}))

const mockData = vi.hoisted(() => {
  const fakeNodeLayout: NodeLayout = {
    id: '',
    position: { x: 0, y: 0 },
    size: { width: 100, height: 100 },
    zIndex: 1,
    visible: true,
    bounds: {
      x: 0,
      y: 0,
      width: 100,
      height: 100
    }
  }
  return { fakeNodeLayout }
})

vi.mock('@/renderer/core/layout/store/layoutStore', () => {
  const isDraggingVueNodes = ref(false)
  const isResizingVueNodes = ref(false)
  const fakeNodeLayoutRef = ref(mockData.fakeNodeLayout)
  const getNodeLayoutRef = vi.fn(() => fakeNodeLayoutRef)
  const setSource = vi.fn()
  return {
    layoutStore: {
      isDraggingVueNodes,
      isResizingVueNodes,
      getNodeLayoutRef,
      setSource
    }
  }
})

const createPointerEvent = (
  eventType: string,
  overrides: Partial<PointerEventInit> = {}
): PointerEvent => {
  return new PointerEvent(eventType, {
    pointerId: 1,
    button: 0,
    clientX: 100,
    clientY: 100,
    ...overrides
  })
}

const createMouseEvent = (
  eventType: string,
  overrides: Partial<MouseEventInit> = {}
): MouseEvent => {
  return new MouseEvent(eventType, {
    button: 2, // Right click
    clientX: 100,
    clientY: 100,
    ...overrides
  })
}

describe('useNodePointerInteractions', () => {
  beforeEach(async () => {
    vi.resetAllMocks()
    selectedItemsState.items = []
    setActivePinia(createTestingPinia())
  })

  it('should only start drag on left-click', async () => {
    const { handleNodeSelect } = useNodeEventHandlers()
    const { startDrag } = useNodeDrag()

    const { pointerHandlers } = useNodePointerInteractions('test-node-123')

    // Right-click should not trigger selection
    const rightClickEvent = createPointerEvent('pointerdown', { button: 2 })
    pointerHandlers.onPointerdown(rightClickEvent)

    expect(handleNodeSelect).not.toHaveBeenCalled()

    // Left-click should trigger selection on pointer down
    const leftClickEvent = createPointerEvent('pointerdown', { button: 0 })
    pointerHandlers.onPointerdown(leftClickEvent)

    expect(startDrag).toHaveBeenCalledWith(leftClickEvent, 'test-node-123')
  })

  it.skip('should call onNodeSelect on pointer down', async () => {
    const { handleNodeSelect } = useNodeEventHandlers()

    const { pointerHandlers } = useNodePointerInteractions('test-node-123')

    // Selection should happen on pointer down
    const downEvent = createPointerEvent('pointerdown', {
      clientX: 100,
      clientY: 100
    })
    pointerHandlers.onPointerdown(downEvent)

    expect(handleNodeSelect).toHaveBeenCalledWith(downEvent, 'test-node-123')

    vi.mocked(handleNodeSelect).mockClear()

    // Even if we drag, selection already happened on pointer down
    pointerHandlers.onPointerup(
      createPointerEvent('pointerup', { clientX: 200, clientY: 200 })
    )

    // onNodeSelect should not be called again on pointer up
    expect(handleNodeSelect).not.toHaveBeenCalled()
  })

  it('should handle drag termination via cancel and context menu', async () => {
    const { handleNodeSelect } = useNodeEventHandlers()

    const { pointerHandlers } = useNodePointerInteractions('test-node-123')

    // Test pointer cancel - selection happens on pointer down
    pointerHandlers.onPointerdown(
      createPointerEvent('pointerdown', { clientX: 100, clientY: 100 })
    )

    // Simulate drag by moving pointer beyond threshold
    pointerHandlers.onPointermove(
      createPointerEvent('pointermove', {
        clientX: 110,
        clientY: 110,
        buttons: 1
      })
    )

    expect(handleNodeSelect).toHaveBeenCalledTimes(1)

    pointerHandlers.onPointercancel(createPointerEvent('pointercancel'))

    // Selection should have been called on pointer down only
    expect(handleNodeSelect).toHaveBeenCalledTimes(1)

    vi.mocked(handleNodeSelect).mockClear()

    // Test context menu during drag prevents default
    pointerHandlers.onPointerdown(
      createPointerEvent('pointerdown', { clientX: 100, clientY: 100 })
    )
    // Simulate drag by moving pointer beyond threshold
    pointerHandlers.onPointermove(
      createPointerEvent('pointermove', {
        clientX: 110,
        clientY: 110,
        buttons: 1
      })
    )

    const contextMenuEvent = createMouseEvent('contextmenu')
    const preventDefaultSpy = vi.spyOn(contextMenuEvent, 'preventDefault')

    pointerHandlers.onContextmenu(contextMenuEvent)

    expect(preventDefaultSpy).toHaveBeenCalled()
  })

  it('should integrate with layout store dragging state', async () => {
    const { pointerHandlers } = useNodePointerInteractions('test-node-123')

    // Pointer down alone shouldn't set dragging state
    pointerHandlers.onPointerdown(
      createPointerEvent('pointerdown', { clientX: 100, clientY: 100 })
    )
    expect(layoutStore.isDraggingVueNodes.value).toBe(false)

    // Move pointer beyond threshold to start drag
    pointerHandlers.onPointermove(
      createPointerEvent('pointermove', {
        clientX: 110,
        clientY: 110,
        buttons: 1
      })
    )
    await nextTick()
    expect(layoutStore.isDraggingVueNodes.value).toBe(true)

    // End drag
    pointerHandlers.onPointercancel(createPointerEvent('pointercancel'))
    await nextTick()
    expect(layoutStore.isDraggingVueNodes.value).toBe(false)
  })

  it('should select node immediately when drag starts', async () => {
    const { pointerHandlers } = useNodePointerInteractions('test-node-123')

    // Pointer down should select node immediately
    const downEvent = createPointerEvent('pointerdown', {
      clientX: 100,
      clientY: 100
    })
    pointerHandlers.onPointerdown(downEvent)
    const { handleNodeSelect } = useNodeEventHandlers()

    // Dragging state should NOT be active yet
    expect(layoutStore.isDraggingVueNodes.value).toBe(false)

    const pointerMove = createPointerEvent('pointermove', {
      clientX: 150,
      clientY: 150,
      buttons: 1
    })
    // Move the pointer beyond threshold (start dragging)
    pointerHandlers.onPointermove(pointerMove)

    // Now dragging state should be active
    expect(layoutStore.isDraggingVueNodes.value).toBe(true)

    // Selection should happen on pointer down (before move)
    expect(handleNodeSelect).toHaveBeenCalledWith(pointerMove, 'test-node-123')
    expect(handleNodeSelect).toHaveBeenCalledTimes(1)

    // End drag
    pointerHandlers.onPointerup(
      createPointerEvent('pointerup', { clientX: 150, clientY: 150 })
    )

    // Selection should still only have been called once
    expect(handleNodeSelect).toHaveBeenCalledTimes(1)
  })

  it('on ctrl+click: calls toggleNodeSelectionAfterPointerUp on pointer up (not pointer down)', async () => {
    const { pointerHandlers } = useNodePointerInteractions('test-node-123')
    const { toggleNodeSelectionAfterPointerUp } = useNodeEventHandlers()

    // Pointer down with ctrl
    const downEvent = createPointerEvent('pointerdown', {
      ctrlKey: true,
      clientX: 100,
      clientY: 100
    })
    pointerHandlers.onPointerdown(downEvent)

    // On pointer down: toggle handler should NOT be called yet
    expect(toggleNodeSelectionAfterPointerUp).not.toHaveBeenCalled()

    // Pointer up with ctrl (no drag - same position)
    const upEvent = createPointerEvent('pointerup', {
      ctrlKey: true,
      clientX: 100,
      clientY: 100
    })
    pointerHandlers.onPointerup(upEvent)

    // On pointer up: toggle handler IS called with correct params
    expect(toggleNodeSelectionAfterPointerUp).toHaveBeenCalledWith(
      'test-node-123',
      true
    )
  })
})
