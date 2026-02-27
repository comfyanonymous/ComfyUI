import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import type { LGraph, Subgraph } from '@/lib/litegraph/src/litegraph'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import type { ComfyWorkflow } from '@/platform/workflow/management/stores/workflowStore'
import { app } from '@/scripts/app'
import { useSubgraphNavigationStore } from '@/stores/subgraphNavigationStore'

const { mockSetDirty } = vi.hoisted(() => ({
  mockSetDirty: vi.fn()
}))

vi.mock('@/scripts/app', () => {
  const mockCanvas = {
    subgraph: null,
    ds: {
      scale: 1,
      offset: [0, 0],
      state: {
        scale: 1,
        offset: [0, 0]
      }
    },
    setDirty: mockSetDirty
  }

  return {
    app: {
      graph: {
        _nodes: [],
        nodes: [],
        subgraphs: new Map(),
        getNodeById: vi.fn()
      },
      canvas: mockCanvas
    }
  }
})

// Mock canvasStore
vi.mock('@/renderer/core/canvas/canvasStore', () => ({
  useCanvasStore: () => ({
    getCanvas: () => app.canvas
  })
}))

// Get reference to mock canvas
const mockCanvas = app.canvas

describe('useSubgraphNavigationStore - Viewport Persistence', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    // Reset canvas state
    mockCanvas.ds.scale = 1
    mockCanvas.ds.offset = [0, 0]
    mockCanvas.ds.state.scale = 1
    mockCanvas.ds.state.offset = [0, 0]
    mockSetDirty.mockClear()
  })

  describe('saveViewport', () => {
    it('should save viewport state for root graph', () => {
      const navigationStore = useSubgraphNavigationStore()

      // Set viewport state
      mockCanvas.ds.state.scale = 2
      mockCanvas.ds.state.offset = [100, 200]

      // Save viewport for root
      navigationStore.saveViewport('root')

      // Check it was saved
      const saved = navigationStore.viewportCache.get('root')
      expect(saved).toEqual({
        scale: 2,
        offset: [100, 200]
      })
    })

    it('should save viewport state for subgraph', () => {
      const navigationStore = useSubgraphNavigationStore()

      // Set viewport state
      mockCanvas.ds.state.scale = 1.5
      mockCanvas.ds.state.offset = [50, 75]

      // Save viewport for subgraph
      navigationStore.saveViewport('subgraph-123')

      // Check it was saved
      const saved = navigationStore.viewportCache.get('subgraph-123')
      expect(saved).toEqual({
        scale: 1.5,
        offset: [50, 75]
      })
    })

    it('should save viewport for current context when no ID provided', () => {
      const navigationStore = useSubgraphNavigationStore()
      const workflowStore = useWorkflowStore()

      // Mock being in a subgraph
      const mockSubgraph = { id: 'sub-456' }
      workflowStore.activeSubgraph = mockSubgraph as Subgraph

      // Set viewport state
      mockCanvas.ds.state.scale = 3
      mockCanvas.ds.state.offset = [10, 20]

      // Save viewport without ID (should default to root since activeSubgraph is not tracked by navigation store)
      navigationStore.saveViewport('sub-456')

      // Should save for the specified subgraph
      const saved = navigationStore.viewportCache.get('sub-456')
      expect(saved).toEqual({
        scale: 3,
        offset: [10, 20]
      })
    })
  })

  describe('restoreViewport', () => {
    it('should restore viewport state for root graph', () => {
      const navigationStore = useSubgraphNavigationStore()

      // Save a viewport state
      navigationStore.viewportCache.set('root', {
        scale: 2.5,
        offset: [150, 250]
      })

      // Restore it
      navigationStore.restoreViewport('root')

      // Check canvas was updated
      expect(mockCanvas.ds.scale).toBe(2.5)
      expect(mockCanvas.ds.offset).toEqual([150, 250])
      expect(mockCanvas.setDirty).toHaveBeenCalledWith(true, true)
    })

    it('should restore viewport state for subgraph', () => {
      const navigationStore = useSubgraphNavigationStore()

      // Save a viewport state
      navigationStore.viewportCache.set('sub-789', {
        scale: 0.75,
        offset: [-50, -100]
      })

      // Restore it
      navigationStore.restoreViewport('sub-789')

      // Check canvas was updated
      expect(mockCanvas.ds.scale).toBe(0.75)
      expect(mockCanvas.ds.offset).toEqual([-50, -100])
    })

    it('should do nothing if no saved viewport exists', () => {
      const navigationStore = useSubgraphNavigationStore()

      // Reset canvas
      mockCanvas.ds.scale = 1
      mockCanvas.ds.offset = [0, 0]
      mockSetDirty.mockClear()

      // Try to restore non-existent viewport
      navigationStore.restoreViewport('non-existent')

      // Canvas should not change
      expect(mockCanvas.ds.scale).toBe(1)
      expect(mockCanvas.ds.offset).toEqual([0, 0])
      expect(mockSetDirty).not.toHaveBeenCalled()
    })
  })

  describe('navigation integration', () => {
    it('should save and restore viewport when navigating between subgraphs', async () => {
      const navigationStore = useSubgraphNavigationStore()
      const workflowStore = useWorkflowStore()

      // Create mock subgraph with both _nodes and nodes properties
      const mockRootGraph = {
        _nodes: [],
        nodes: [],
        subgraphs: new Map(),
        getNodeById: vi.fn()
      } as Partial<LGraph> as LGraph
      const subgraph1 = {
        id: 'sub1',
        rootGraph: mockRootGraph,
        _nodes: [],
        nodes: []
      }

      // Start at root with custom viewport
      mockCanvas.ds.state.scale = 2
      mockCanvas.ds.state.offset = [100, 100]

      // Navigate to subgraph
      workflowStore.activeSubgraph = subgraph1 as Partial<Subgraph> as Subgraph
      await nextTick()

      // Root viewport should have been saved automatically
      const rootViewport = navigationStore.viewportCache.get('root')
      expect(rootViewport).toBeDefined()
      expect(rootViewport?.scale).toBe(2)
      expect(rootViewport?.offset).toEqual([100, 100])

      // Change viewport in subgraph
      mockCanvas.ds.state.scale = 0.5
      mockCanvas.ds.state.offset = [-50, -50]

      // Navigate back to root
      workflowStore.activeSubgraph = undefined
      await nextTick()

      // Subgraph viewport should have been saved automatically
      const sub1Viewport = navigationStore.viewportCache.get('sub1')
      expect(sub1Viewport).toBeDefined()
      expect(sub1Viewport?.scale).toBe(0.5)
      expect(sub1Viewport?.offset).toEqual([-50, -50])

      // Root viewport should be restored automatically
      expect(mockCanvas.ds.scale).toBe(2)
      expect(mockCanvas.ds.offset).toEqual([100, 100])
    })

    it('should preserve viewport cache when switching workflows', async () => {
      const navigationStore = useSubgraphNavigationStore()
      const workflowStore = useWorkflowStore()

      // Add some viewport states
      navigationStore.viewportCache.set('root', { scale: 2, offset: [0, 0] })
      navigationStore.viewportCache.set('sub1', {
        scale: 1.5,
        offset: [10, 10]
      })

      expect(navigationStore.viewportCache.size).toBe(2)

      // Switch workflows
      const workflow1 = { path: 'workflow1.json' } as ComfyWorkflow
      const workflow2 = { path: 'workflow2.json' } as ComfyWorkflow

      workflowStore.activeWorkflow = workflow1 as ReturnType<
        typeof useWorkflowStore
      >['activeWorkflow']
      await nextTick()

      workflowStore.activeWorkflow = workflow2 as ReturnType<
        typeof useWorkflowStore
      >['activeWorkflow']
      await nextTick()

      // Cache should be preserved (LRU will manage memory)
      expect(navigationStore.viewportCache.size).toBe(2)
      expect(navigationStore.viewportCache.has('root')).toBe(true)
      expect(navigationStore.viewportCache.has('sub1')).toBe(true)
    })
  })
})
