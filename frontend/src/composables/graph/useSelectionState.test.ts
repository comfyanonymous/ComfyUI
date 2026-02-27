import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, test, vi } from 'vitest'

import { useSelectionState } from '@/composables/graph/useSelectionState'
import { useNodeLibrarySidebarTab } from '@/composables/sidebarTabs/useNodeLibrarySidebarTab'
import { LGraphEventMode } from '@/lib/litegraph/src/litegraph'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { isImageNode, isLGraphNode } from '@/utils/litegraphUtil'
import { filterOutputNodes } from '@/utils/nodeFilterUtil'
import {
  createMockLGraphNode,
  createMockPositionable
} from '@/utils/__tests__/litegraphTestUtils'

// Mock composables
vi.mock('@/composables/sidebarTabs/useNodeLibrarySidebarTab', () => ({
  useNodeLibrarySidebarTab: vi.fn()
}))

vi.mock('@/utils/litegraphUtil', () => ({
  isLGraphNode: vi.fn(),
  isImageNode: vi.fn()
}))

vi.mock('@/utils/nodeFilterUtil', () => ({
  filterOutputNodes: vi.fn()
}))

// Mock comment/connection objects with additional properties
const mockComment = {
  ...createMockPositionable({ id: 999 }),
  type: 'comment',
  isNode: false
}
const mockConnection = {
  ...createMockPositionable({ id: 1000 }),
  type: 'connection',
  isNode: false
}

describe('useSelectionState', () => {
  beforeEach(() => {
    vi.clearAllMocks()

    // Create testing Pinia instance
    setActivePinia(
      createTestingPinia({
        createSpy: vi.fn
      })
    )

    // Setup mock composables
    vi.mocked(useNodeLibrarySidebarTab).mockReturnValue({
      id: 'node-library-tab',
      title: 'Node Library',
      type: 'custom',
      render: () => null
    } as ReturnType<typeof useNodeLibrarySidebarTab>)

    // Setup mock utility functions
    vi.mocked(isLGraphNode).mockImplementation((item: unknown) => {
      const typedItem = item as { isNode?: boolean }
      return typedItem?.isNode !== false
    })
    vi.mocked(isImageNode).mockImplementation((node: unknown) => {
      const typedNode = node as { type?: string }
      return typedNode?.type === 'ImageNode'
    })
    vi.mocked(filterOutputNodes).mockImplementation((nodes) =>
      nodes.filter((n) => n.type === 'OutputNode')
    )
  })

  describe('Selection Detection', () => {
    test('should return false when nothing selected', () => {
      const { hasAnySelection } = useSelectionState()
      expect(hasAnySelection.value).toBe(false)
    })

    test('should return true when items selected', () => {
      const canvasStore = useCanvasStore()
      const node1 = createMockLGraphNode({ id: 1 })
      const node2 = createMockLGraphNode({ id: 2 })
      canvasStore.$state.selectedItems = [node1, node2]

      const { hasAnySelection } = useSelectionState()
      expect(hasAnySelection.value).toBe(true)
    })

    test('hasMultipleSelection should be true when 2+ items selected', () => {
      const canvasStore = useCanvasStore()
      const node1 = createMockLGraphNode({ id: 1 })
      const node2 = createMockLGraphNode({ id: 2 })
      canvasStore.$state.selectedItems = [node1, node2]

      const { hasMultipleSelection } = useSelectionState()
      expect(hasMultipleSelection.value).toBe(true)
    })

    test('hasMultipleSelection should be false when only 1 item selected', () => {
      const canvasStore = useCanvasStore()
      const node1 = createMockLGraphNode({ id: 1 })
      canvasStore.$state.selectedItems = [node1]

      const { hasMultipleSelection } = useSelectionState()
      expect(hasMultipleSelection.value).toBe(false)
    })
  })

  describe('Node Type Filtering', () => {
    test('should pick only LGraphNodes from mixed selections', () => {
      const canvasStore = useCanvasStore()
      const graphNode = createMockLGraphNode({ id: 3 })
      canvasStore.$state.selectedItems = [
        graphNode,
        mockComment,
        mockConnection
      ]

      const { selectedNodes } = useSelectionState()
      expect(selectedNodes.value).toHaveLength(1)
      expect(selectedNodes.value[0]).toEqual(graphNode)
    })
  })

  describe('Node State Computation', () => {
    test('should detect bypassed nodes', () => {
      const canvasStore = useCanvasStore()
      const bypassedNode = createMockLGraphNode({
        id: 4,
        mode: LGraphEventMode.BYPASS
      })
      canvasStore.$state.selectedItems = [bypassedNode]

      const { selectedNodes } = useSelectionState()
      const isBypassed = selectedNodes.value.some(
        (n) => n.mode === LGraphEventMode.BYPASS
      )
      expect(isBypassed).toBe(true)
    })

    test('should detect pinned/collapsed states', () => {
      const canvasStore = useCanvasStore()
      const pinnedNode = createMockLGraphNode({ id: 5, pinned: true })
      const collapsedNode = createMockLGraphNode({
        id: 6,
        flags: { collapsed: true }
      })
      canvasStore.$state.selectedItems = [pinnedNode, collapsedNode]

      const { selectedNodes } = useSelectionState()
      const isPinned = selectedNodes.value.some((n) => n.pinned === true)
      const isCollapsed = selectedNodes.value.some(
        (n) => n.flags?.collapsed === true
      )
      const isBypassed = selectedNodes.value.some(
        (n) => n.mode === LGraphEventMode.BYPASS
      )
      expect(isPinned).toBe(true)
      expect(isCollapsed).toBe(true)
      expect(isBypassed).toBe(false)
    })

    test('should provide non-reactive state computation', () => {
      const canvasStore = useCanvasStore()
      const node = createMockLGraphNode({ id: 7, pinned: true })
      canvasStore.$state.selectedItems = [node]

      const { selectedNodes } = useSelectionState()
      const isPinned = selectedNodes.value.some((n) => n.pinned === true)
      const isCollapsed = selectedNodes.value.some(
        (n) => n.flags?.collapsed === true
      )
      const isBypassed = selectedNodes.value.some(
        (n) => n.mode === LGraphEventMode.BYPASS
      )

      expect(isPinned).toBe(true)
      expect(isCollapsed).toBe(false)
      expect(isBypassed).toBe(false)

      // Test with empty selection using new composable instance
      canvasStore.$state.selectedItems = []
      const { selectedNodes: newSelectedNodes } = useSelectionState()
      const newIsPinned = newSelectedNodes.value.some((n) => n.pinned === true)
      expect(newIsPinned).toBe(false)
    })
  })
})
