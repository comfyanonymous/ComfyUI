import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { ComfyNodeDef } from '@/schemas/nodeDefSchema'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import type { NodeDefFilter } from '@/stores/nodeDefStore'

describe('useNodeDefStore', () => {
  let store: ReturnType<typeof useNodeDefStore>

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useNodeDefStore()
  })

  const createMockNodeDef = (
    overrides: Partial<ComfyNodeDef> = {}
  ): ComfyNodeDef => ({
    name: 'TestNode',
    display_name: 'Test Node',
    category: 'test',
    python_module: 'test_module',
    description: 'Test node',
    input: {},
    output: [],
    output_is_list: [],
    output_name: [],
    output_node: false,
    deprecated: false,
    experimental: false,
    ...overrides
  })

  describe('filter registry', () => {
    it('should register a new filter', () => {
      const filter: NodeDefFilter = {
        id: 'test.filter',
        name: 'Test Filter',
        predicate: () => true
      }

      store.registerNodeDefFilter(filter)
      expect(store.nodeDefFilters).toContainEqual(filter)
    })

    it('should unregister a filter by id', () => {
      const filter: NodeDefFilter = {
        id: 'test.filter',
        name: 'Test Filter',
        predicate: () => true
      }

      store.registerNodeDefFilter(filter)
      store.unregisterNodeDefFilter('test.filter')
      expect(store.nodeDefFilters).not.toContainEqual(filter)
    })

    it('should register core filters on initialization', () => {
      const deprecatedFilter = store.nodeDefFilters.find(
        (f) => f.id === 'core.deprecated'
      )
      const experimentalFilter = store.nodeDefFilters.find(
        (f) => f.id === 'core.experimental'
      )

      expect(deprecatedFilter).toBeDefined()
      expect(experimentalFilter).toBeDefined()
    })
  })

  describe('filter application', () => {
    beforeEach(() => {
      // Clear existing filters for isolated tests
      store.nodeDefFilters.splice(0)
    })

    it('should apply single filter to visible nodes', () => {
      const normalNode = createMockNodeDef({
        name: 'normal',
        deprecated: false
      })
      const deprecatedNode = createMockNodeDef({
        name: 'deprecated',
        deprecated: true
      })

      store.updateNodeDefs([normalNode, deprecatedNode])

      // Register filter that hides deprecated nodes
      store.registerNodeDefFilter({
        id: 'test.no-deprecated',
        name: 'Hide Deprecated',
        predicate: (node) => !node.deprecated
      })

      expect(store.visibleNodeDefs).toHaveLength(1)
      expect(store.visibleNodeDefs[0].name).toBe('normal')
    })

    it('should apply multiple filters with AND logic', () => {
      const node1 = createMockNodeDef({
        name: 'node1',
        deprecated: false,
        experimental: false
      })
      const node2 = createMockNodeDef({
        name: 'node2',
        deprecated: true,
        experimental: false
      })
      const node3 = createMockNodeDef({
        name: 'node3',
        deprecated: false,
        experimental: true
      })
      const node4 = createMockNodeDef({
        name: 'node4',
        deprecated: true,
        experimental: true
      })

      store.updateNodeDefs([node1, node2, node3, node4])

      // Register filters
      store.registerNodeDefFilter({
        id: 'test.no-deprecated',
        name: 'Hide Deprecated',
        predicate: (node) => !node.deprecated
      })

      store.registerNodeDefFilter({
        id: 'test.no-experimental',
        name: 'Hide Experimental',
        predicate: (node) => !node.experimental
      })

      // Only node1 should be visible (not deprecated AND not experimental)
      expect(store.visibleNodeDefs).toHaveLength(1)
      expect(store.visibleNodeDefs[0].name).toBe('node1')
    })

    it('should show all nodes when no filters are registered', () => {
      const nodes = [
        createMockNodeDef({ name: 'node1' }),
        createMockNodeDef({ name: 'node2' }),
        createMockNodeDef({ name: 'node3' })
      ]

      store.updateNodeDefs(nodes)
      expect(store.visibleNodeDefs).toHaveLength(3)
    })

    it('should update visibility when filter is removed', () => {
      const deprecatedNode = createMockNodeDef({
        name: 'deprecated',
        deprecated: true
      })
      store.updateNodeDefs([deprecatedNode])

      const filter: NodeDefFilter = {
        id: 'test.no-deprecated',
        name: 'Hide Deprecated',
        predicate: (node) => !node.deprecated
      }

      // Add filter - node should be hidden
      store.registerNodeDefFilter(filter)
      expect(store.visibleNodeDefs).toHaveLength(0)

      // Remove filter - node should be visible
      store.unregisterNodeDefFilter('test.no-deprecated')
      expect(store.visibleNodeDefs).toHaveLength(1)
    })
  })

  describe('core filters behavior', () => {
    it('should hide deprecated nodes by default', () => {
      const normalNode = createMockNodeDef({
        name: 'normal',
        deprecated: false
      })
      const deprecatedNode = createMockNodeDef({
        name: 'deprecated',
        deprecated: true
      })

      store.updateNodeDefs([normalNode, deprecatedNode])

      expect(store.visibleNodeDefs).toHaveLength(1)
      expect(store.visibleNodeDefs[0].name).toBe('normal')
    })

    it('should show deprecated nodes when showDeprecated is true', () => {
      const normalNode = createMockNodeDef({
        name: 'normal',
        deprecated: false
      })
      const deprecatedNode = createMockNodeDef({
        name: 'deprecated',
        deprecated: true
      })

      store.updateNodeDefs([normalNode, deprecatedNode])
      store.showDeprecated = true

      expect(store.visibleNodeDefs).toHaveLength(2)
    })

    it('should hide experimental nodes by default', () => {
      const normalNode = createMockNodeDef({
        name: 'normal',
        experimental: false
      })
      const experimentalNode = createMockNodeDef({
        name: 'experimental',
        experimental: true
      })

      store.updateNodeDefs([normalNode, experimentalNode])

      expect(store.visibleNodeDefs).toHaveLength(1)
      expect(store.visibleNodeDefs[0].name).toBe('normal')
    })

    it('should show experimental nodes when showExperimental is true', () => {
      const normalNode = createMockNodeDef({
        name: 'normal',
        experimental: false
      })
      const experimentalNode = createMockNodeDef({
        name: 'experimental',
        experimental: true
      })

      store.updateNodeDefs([normalNode, experimentalNode])
      store.showExperimental = true

      expect(store.visibleNodeDefs).toHaveLength(2)
    })

    it('should hide subgraph nodes by default', () => {
      const normalNode = createMockNodeDef({
        name: 'normal',
        category: 'conditioning',
        python_module: 'nodes'
      })
      const subgraphNode = createMockNodeDef({
        name: 'MySubgraph',
        category: 'subgraph',
        python_module: 'nodes'
      })

      store.updateNodeDefs([normalNode, subgraphNode])

      expect(store.visibleNodeDefs).toHaveLength(1)
      expect(store.visibleNodeDefs[0].name).toBe('normal')
    })

    it('should show non-subgraph nodes with subgraph category', () => {
      const normalNode = createMockNodeDef({
        name: 'normal',
        category: 'conditioning',
        python_module: 'custom_extension'
      })
      const fakeSubgraphNode = createMockNodeDef({
        name: 'FakeSubgraph',
        category: 'subgraph',
        python_module: 'custom_extension' // Different python_module
      })

      store.updateNodeDefs([normalNode, fakeSubgraphNode])

      expect(store.visibleNodeDefs).toHaveLength(2)
      expect(store.visibleNodeDefs.map((n) => n.name)).toEqual([
        'normal',
        'FakeSubgraph'
      ])
    })
  })

  describe('allNodeDefsByName', () => {
    it('should include all node defs by name', () => {
      const node1 = createMockNodeDef({ name: 'Node1' })
      const node2 = createMockNodeDef({ name: 'Node2' })

      store.updateNodeDefs([node1, node2])

      expect(store.allNodeDefsByName).toHaveProperty('Node1')
      expect(store.allNodeDefsByName).toHaveProperty('Node2')
      expect(store.allNodeDefsByName['Node1'].name).toBe('Node1')
      expect(store.allNodeDefsByName['Node2'].name).toBe('Node2')
    })

    it('should include deprecated and experimental nodes', () => {
      const normal = createMockNodeDef({ name: 'Normal' })
      const deprecated = createMockNodeDef({
        name: 'Deprecated',
        deprecated: true
      })
      const experimental = createMockNodeDef({
        name: 'Experimental',
        experimental: true
      })

      store.updateNodeDefs([normal, deprecated, experimental])

      expect(store.allNodeDefsByName).toHaveProperty('Normal')
      expect(store.allNodeDefsByName).toHaveProperty('Deprecated')
      expect(store.allNodeDefsByName).toHaveProperty('Experimental')
    })

    it('should include nodes filtered out of visibleNodeDefs', () => {
      const normal = createMockNodeDef({
        name: 'Normal',
        deprecated: false
      })
      const deprecated = createMockNodeDef({
        name: 'Deprecated',
        deprecated: true
      })

      store.updateNodeDefs([normal, deprecated])

      // visibleNodeDefs filters out deprecated by default
      expect(store.visibleNodeDefs).toHaveLength(1)

      // allNodeDefsByName includes all
      expect(store.allNodeDefsByName).toHaveProperty('Normal')
      expect(store.allNodeDefsByName).toHaveProperty('Deprecated')
    })
  })

  describe('performance', () => {
    it('should perform single traversal for multiple filters', () => {
      let filterCallCount = 0

      // Register multiple filters that count their calls
      for (let i = 0; i < 5; i++) {
        store.registerNodeDefFilter({
          id: `test.counter-${i}`,
          name: `Counter ${i}`,
          predicate: () => {
            filterCallCount++
            return true
          }
        })
      }

      const nodes = Array.from({ length: 10 }, (_, i) =>
        createMockNodeDef({ name: `node${i}` })
      )
      store.updateNodeDefs(nodes)

      // Force recomputation by accessing visibleNodeDefs
      expect(store.visibleNodeDefs).toBeDefined()

      // Each node (10) should be checked by each filter (5 test + 2 core = 7 total)
      expect(filterCallCount).toBe(10 * 5)
    })
  })
})
