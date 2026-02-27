import { describe, expect, it, vi } from 'vitest'
import { computed } from 'vue'
import type { ComputedRef } from 'vue'

import type { NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { LGraph, LGraphNode, LLink } from '@/lib/litegraph/src/litegraph'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import type { NodeLayout } from '@/renderer/core/layout/types'
import { MinimapDataSourceFactory } from '@/renderer/extensions/minimap/data/MinimapDataSourceFactory'

// Mock layoutStore
vi.mock('@/renderer/core/layout/store/layoutStore', () => ({
  layoutStore: {
    getAllNodes: vi.fn()
  }
}))

// Helper to create mock links that satisfy LGraph['links'] type
function createMockLinks(): LGraph['links'] {
  const map = new Map<number, LLink>()
  return Object.assign(map, {}) as LGraph['links']
}

describe('MinimapDataSource', () => {
  describe('MinimapDataSourceFactory', () => {
    it('should create LayoutStoreDataSource when LayoutStore has data', () => {
      // Arrange
      const mockNodes = new Map<string, NodeLayout>([
        [
          'node1',
          {
            id: 'node1',
            position: { x: 0, y: 0 },
            size: { width: 100, height: 50 },
            zIndex: 0,
            visible: true,
            bounds: { x: 0, y: 0, width: 100, height: 50 }
          }
        ]
      ])

      // Create a computed ref that returns the map
      const computedNodes: ComputedRef<ReadonlyMap<string, NodeLayout>> =
        computed(() => mockNodes)
      vi.mocked(layoutStore.getAllNodes).mockReturnValue(computedNodes)

      const mockGraph: Pick<LGraph, '_nodes' | '_groups' | 'links'> = {
        _nodes: [],
        _groups: [],
        links: createMockLinks()
      }

      // Act
      const dataSource = MinimapDataSourceFactory.create(mockGraph as LGraph)

      // Assert
      expect(dataSource).toBeDefined()
      expect(dataSource.hasData()).toBe(true)
      expect(dataSource.getNodeCount()).toBe(1)
    })

    it('should create LiteGraphDataSource when LayoutStore is empty', () => {
      // Arrange
      const emptyMap = new Map<string, NodeLayout>()
      const computedEmpty: ComputedRef<ReadonlyMap<string, NodeLayout>> =
        computed(() => emptyMap)
      vi.mocked(layoutStore.getAllNodes).mockReturnValue(computedEmpty)

      const mockNode: Pick<
        LGraphNode,
        'id' | 'pos' | 'size' | 'bgcolor' | 'mode' | 'has_errors' | 'outputs'
      > = {
        id: 'node1' as NodeId,
        pos: [0, 0],
        size: [100, 50],
        bgcolor: '#fff',
        mode: 0,
        has_errors: false,
        outputs: []
      }

      const mockGraph: Pick<LGraph, '_nodes' | '_groups' | 'links'> = {
        _nodes: [mockNode as LGraphNode],
        _groups: [],
        links: createMockLinks()
      }

      // Act
      const dataSource = MinimapDataSourceFactory.create(mockGraph as LGraph)

      // Assert
      expect(dataSource).toBeDefined()
      expect(dataSource.hasData()).toBe(true)
      expect(dataSource.getNodeCount()).toBe(1)

      const nodes = dataSource.getNodes()
      expect(nodes).toHaveLength(1)
      expect(nodes[0]).toMatchObject({
        id: 'node1',
        x: 0,
        y: 0,
        width: 100,
        height: 50
      })
    })

    it('should handle empty graph correctly', () => {
      // Arrange
      const emptyMap = new Map<string, NodeLayout>()
      const computedEmpty: ComputedRef<ReadonlyMap<string, NodeLayout>> =
        computed(() => emptyMap)
      vi.mocked(layoutStore.getAllNodes).mockReturnValue(computedEmpty)

      const mockGraph: Pick<LGraph, '_nodes' | '_groups' | 'links'> = {
        _nodes: [],
        _groups: [],
        links: createMockLinks()
      }

      // Act
      const dataSource = MinimapDataSourceFactory.create(mockGraph as LGraph)

      // Assert
      expect(dataSource.hasData()).toBe(false)
      expect(dataSource.getNodeCount()).toBe(0)
      expect(dataSource.getNodes()).toEqual([])
      expect(dataSource.getLinks()).toEqual([])
      expect(dataSource.getGroups()).toEqual([])
    })
  })

  describe('Bounds calculation', () => {
    it('should calculate correct bounds from nodes', () => {
      // Arrange
      const emptyMap = new Map<string, NodeLayout>()
      const computedEmpty: ComputedRef<ReadonlyMap<string, NodeLayout>> =
        computed(() => emptyMap)
      vi.mocked(layoutStore.getAllNodes).mockReturnValue(computedEmpty)

      const mockNode1: Pick<LGraphNode, 'id' | 'pos' | 'size' | 'outputs'> = {
        id: 'node1' as NodeId,
        pos: [0, 0],
        size: [100, 50],
        outputs: []
      }

      const mockNode2: Pick<LGraphNode, 'id' | 'pos' | 'size' | 'outputs'> = {
        id: 'node2' as NodeId,
        pos: [200, 100],
        size: [150, 75],
        outputs: []
      }

      const mockGraph: Pick<LGraph, '_nodes' | '_groups' | 'links'> = {
        _nodes: [mockNode1 as LGraphNode, mockNode2 as LGraphNode],
        _groups: [],
        links: createMockLinks()
      }

      // Act
      const dataSource = MinimapDataSourceFactory.create(mockGraph as LGraph)
      const bounds = dataSource.getBounds()

      // Assert
      expect(bounds).toEqual({
        minX: 0,
        minY: 0,
        maxX: 350,
        maxY: 175,
        width: 350,
        height: 175
      })
    })
  })
})
