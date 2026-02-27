import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, ref } from 'vue'

import type { LGraphNode, LGraph } from '@/lib/litegraph/src/litegraph'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { collectAllNodes } from '@/utils/graphTraversalUtil'
import { useMissingNodes } from '@/workbench/extensions/manager/composables/nodePack/useMissingNodes'
import { useWorkflowPacks } from '@/workbench/extensions/manager/composables/nodePack/useWorkflowPacks'
import type { WorkflowPack } from '@/workbench/extensions/manager/composables/nodePack/useWorkflowPacks'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import { createMockLGraphNode } from '@/utils/__tests__/litegraphTestUtils'

vi.mock('@vueuse/core', async () => {
  const actual = await vi.importActual('@vueuse/core')
  return {
    ...actual,
    createSharedComposable: <Fn extends (...args: unknown[]) => unknown>(
      fn: Fn
    ) => fn
  }
})

// Mock the dependencies
vi.mock(
  '@/workbench/extensions/manager/composables/nodePack/useWorkflowPacks',
  () => ({
    useWorkflowPacks: vi.fn()
  })
)

vi.mock('@/workbench/extensions/manager/stores/comfyManagerStore', () => ({
  useComfyManagerStore: vi.fn()
}))

vi.mock('@/stores/nodeDefStore', () => ({
  useNodeDefStore: vi.fn()
}))

vi.mock('@/platform/workflow/management/stores/workflowStore', () => ({
  useWorkflowStore: vi.fn(() => ({
    activeWorkflow: null
  }))
}))

const mockApp: { rootGraph?: Partial<LGraph> } = vi.hoisted(() => ({}))

vi.mock('@/scripts/app', () => ({
  app: mockApp
}))

vi.mock('@/utils/graphTraversalUtil', () => ({
  collectAllNodes: vi.fn()
}))

const mockUseWorkflowPacks = vi.mocked(useWorkflowPacks)
const mockUseComfyManagerStore = vi.mocked(useComfyManagerStore)
const mockUseNodeDefStore = vi.mocked(useNodeDefStore)
const mockCollectAllNodes = vi.mocked(collectAllNodes)

describe('useMissingNodes', () => {
  const mockWorkflowPacks = [
    {
      id: 'pack-1',
      name: 'Test Pack 1',
      latest_version: { version: '1.0.0' }
    },
    {
      id: 'pack-2',
      name: 'Test Pack 2',
      latest_version: { version: '2.0.0' }
    },
    {
      id: 'pack-3',
      name: 'Installed Pack',
      latest_version: { version: '1.5.0' }
    }
  ]

  const mockStartFetchWorkflowPacks = vi.fn()
  const mockIsPackInstalled = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()

    // Default setup: pack-3 is installed, others are not
    mockIsPackInstalled.mockImplementation((id: string) => id === 'pack-3')

    mockUseComfyManagerStore.mockReturnValue({
      isPackInstalled: mockIsPackInstalled
    } as Partial<ReturnType<typeof useComfyManagerStore>> as ReturnType<
      typeof useComfyManagerStore
    >)

    mockUseWorkflowPacks.mockReturnValue({
      workflowPacks: ref([]),
      isLoading: ref(false),
      error: ref(null),
      startFetchWorkflowPacks: mockStartFetchWorkflowPacks,
      isReady: ref(false),
      filterWorkflowPack: vi.fn()
    })

    // Reset node def store mock
    mockUseNodeDefStore.mockReturnValue({
      nodeDefsByName: {}
    } as Partial<ReturnType<typeof useNodeDefStore>> as ReturnType<
      typeof useNodeDefStore
    >)

    // Reset app.rootGraph.nodes
    mockApp.rootGraph = { nodes: [] }

    // Default mock for collectAllNodes - returns empty array
    mockCollectAllNodes.mockReturnValue([])
  })

  describe('core filtering logic', () => {
    it('filters out installed packs correctly', () => {
      mockUseWorkflowPacks.mockReturnValue({
        workflowPacks: ref(mockWorkflowPacks),
        isLoading: ref(false),
        error: ref(null),
        startFetchWorkflowPacks: mockStartFetchWorkflowPacks,
        isReady: ref(true),
        filterWorkflowPack: vi.fn()
      })

      const { missingNodePacks } = useMissingNodes()

      // Should only include packs that are not installed (pack-1, pack-2)
      expect(missingNodePacks.value).toHaveLength(2)
      expect(missingNodePacks.value[0].id).toBe('pack-1')
      expect(missingNodePacks.value[1].id).toBe('pack-2')
      expect(
        missingNodePacks.value.find((pack) => pack.id === 'pack-3')
      ).toBeUndefined()
    })

    it('returns empty array when all packs are installed', () => {
      mockUseWorkflowPacks.mockReturnValue({
        workflowPacks: ref(mockWorkflowPacks),
        isLoading: ref(false),
        error: ref(null),
        startFetchWorkflowPacks: mockStartFetchWorkflowPacks,
        isReady: ref(true),
        filterWorkflowPack: vi.fn()
      })

      // Mock all packs as installed
      mockIsPackInstalled.mockReturnValue(true)

      const { missingNodePacks } = useMissingNodes()

      expect(missingNodePacks.value).toEqual([])
    })

    it('returns all packs when none are installed', () => {
      mockUseWorkflowPacks.mockReturnValue({
        workflowPacks: ref(mockWorkflowPacks),
        isLoading: ref(false),
        error: ref(null),
        startFetchWorkflowPacks: mockStartFetchWorkflowPacks,
        isReady: ref(true),
        filterWorkflowPack: vi.fn()
      })

      // Mock no packs as installed
      mockIsPackInstalled.mockReturnValue(false)

      const { missingNodePacks } = useMissingNodes()

      expect(missingNodePacks.value).toHaveLength(3)
      expect(missingNodePacks.value).toEqual(mockWorkflowPacks)
    })

    it('returns empty array when no workflow packs exist', () => {
      const { missingNodePacks } = useMissingNodes()

      expect(missingNodePacks.value).toEqual([])
    })
  })

  describe('automatic data fetching', () => {
    it('fetches workflow packs automatically on initialization via watch with immediate:true', async () => {
      useMissingNodes()

      expect(mockStartFetchWorkflowPacks).toHaveBeenCalledOnce()
    })

    it('fetches even when packs already exist (watch always fires with immediate:true)', async () => {
      mockUseWorkflowPacks.mockReturnValue({
        workflowPacks: ref(mockWorkflowPacks),
        isLoading: ref(false),
        error: ref(null),
        startFetchWorkflowPacks: mockStartFetchWorkflowPacks,
        isReady: ref(true),
        filterWorkflowPack: vi.fn()
      })

      useMissingNodes()

      expect(mockStartFetchWorkflowPacks).toHaveBeenCalledOnce()
    })

    it('fetches even when already loading (watch fires regardless of loading state)', async () => {
      mockUseWorkflowPacks.mockReturnValue({
        workflowPacks: ref([]),
        isLoading: ref(true),
        error: ref(null),
        startFetchWorkflowPacks: mockStartFetchWorkflowPacks,
        isReady: ref(false),
        filterWorkflowPack: vi.fn()
      })

      useMissingNodes()

      expect(mockStartFetchWorkflowPacks).toHaveBeenCalledOnce()
    })
  })

  describe('state management', () => {
    it('exposes loading state from useWorkflowPacks', () => {
      mockUseWorkflowPacks.mockReturnValue({
        workflowPacks: ref([]),
        isLoading: ref(true),
        error: ref(null),
        startFetchWorkflowPacks: mockStartFetchWorkflowPacks,
        isReady: ref(false),
        filterWorkflowPack: vi.fn()
      })

      const { isLoading } = useMissingNodes()

      expect(isLoading.value).toBe(true)
    })

    it('exposes error state from useWorkflowPacks', () => {
      const testError = 'Failed to fetch workflow packs'
      mockUseWorkflowPacks.mockReturnValue({
        workflowPacks: ref([]),
        isLoading: ref(false),
        error: ref(testError),
        startFetchWorkflowPacks: mockStartFetchWorkflowPacks,
        isReady: ref(false),
        filterWorkflowPack: vi.fn()
      })

      const { error } = useMissingNodes()

      expect(error.value).toBe(testError)
    })
  })

  describe('reactivity', () => {
    it('updates when workflow packs change', async () => {
      const workflowPacksRef = ref<WorkflowPack[]>([])
      mockUseWorkflowPacks.mockReturnValue({
        workflowPacks: workflowPacksRef,
        isLoading: ref(false),
        error: ref(null),
        startFetchWorkflowPacks: mockStartFetchWorkflowPacks,
        isReady: ref(true),
        filterWorkflowPack: vi.fn()
      })

      const { missingNodePacks } = useMissingNodes()

      // Initially empty
      expect(missingNodePacks.value).toEqual([])

      // Update workflow packs

      workflowPacksRef.value =
        mockWorkflowPacks as Partial<WorkflowPack>[] as WorkflowPack[]
      await nextTick()

      // Should update missing packs (2 missing since pack-3 is installed)
      expect(missingNodePacks.value).toHaveLength(2)
    })

    it('clears missing nodes when switching to empty workflow', async () => {
      const workflowPacksRef = ref(mockWorkflowPacks)
      mockUseWorkflowPacks.mockReturnValue({
        workflowPacks: workflowPacksRef,
        isLoading: ref(false),
        error: ref(null),
        startFetchWorkflowPacks: mockStartFetchWorkflowPacks,
        isReady: ref(true),
        filterWorkflowPack: vi.fn()
      })

      const { hasMissingNodes, missingNodePacks } = useMissingNodes()

      // Should have missing nodes initially (2 missing since pack-3 is installed)
      expect(missingNodePacks.value).toHaveLength(2)
      expect(hasMissingNodes.value).toBe(true)

      // Switch to empty workflow (simulates creating a new empty workflow)
      workflowPacksRef.value = []
      await nextTick()

      // Should clear missing nodes
      expect(missingNodePacks.value).toHaveLength(0)
      expect(hasMissingNodes.value).toBe(false)
    })
  })

  describe('missing core nodes detection', () => {
    const createMockNode = (type: string, packId?: string, version?: string) =>
      createMockLGraphNode({
        type,
        properties: { cnr_id: packId, ver: version },
        id: 1,
        title: type,
        pos: [0, 0],
        size: [100, 100],
        flags: {},
        graph: null,
        mode: 0,
        inputs: [],
        outputs: []
      })

    it('identifies missing core nodes not in nodeDefStore', () => {
      const coreNode1 = createMockNode('CoreNode1', 'comfy-core', '1.2.0')
      const coreNode2 = createMockNode('CoreNode2', 'comfy-core', '1.2.0')

      // Mock collectAllNodes to return only the filtered nodes (missing core nodes)
      mockCollectAllNodes.mockReturnValue([coreNode1, coreNode2])

      const namedNode = {
        name: 'RegisteredNode'
      } as Partial<ComfyNodeDefImpl> as ComfyNodeDefImpl
      mockUseNodeDefStore.mockReturnValue({
        nodeDefsByName: {
          RegisteredNode: namedNode
        }
      } as Partial<ReturnType<typeof useNodeDefStore>> as ReturnType<
        typeof useNodeDefStore
      >)

      const { missingCoreNodes } = useMissingNodes()

      expect(Object.keys(missingCoreNodes.value)).toHaveLength(1)
      expect(missingCoreNodes.value['1.2.0']).toHaveLength(2)
      expect(missingCoreNodes.value['1.2.0'][0].type).toBe('CoreNode1')
      expect(missingCoreNodes.value['1.2.0'][1].type).toBe('CoreNode2')
    })

    it('groups missing core nodes by version', () => {
      const node120 = createMockNode('Node120', 'comfy-core', '1.2.0')
      const node130 = createMockNode('Node130', 'comfy-core', '1.3.0')
      const nodeNoVer = createMockNode('NodeNoVer', 'comfy-core')

      // Mock collectAllNodes to return these nodes
      mockCollectAllNodes.mockReturnValue([node120, node130, nodeNoVer])

      mockUseNodeDefStore.mockReturnValue({
        nodeDefsByName: {}
      } as Partial<ReturnType<typeof useNodeDefStore>> as ReturnType<
        typeof useNodeDefStore
      >)

      const { missingCoreNodes } = useMissingNodes()

      expect(Object.keys(missingCoreNodes.value)).toHaveLength(3)
      expect(missingCoreNodes.value['1.2.0']).toHaveLength(1)
      expect(missingCoreNodes.value['1.3.0']).toHaveLength(1)
      expect(missingCoreNodes.value['']).toHaveLength(1)
    })

    it('ignores non-core nodes', () => {
      const coreNode = createMockNode('CoreNode', 'comfy-core', '1.2.0')

      // Mock collectAllNodes to return only the filtered nodes (core nodes only)
      mockCollectAllNodes.mockReturnValue([coreNode])

      mockUseNodeDefStore.mockReturnValue({
        nodeDefsByName: {}
      } as Partial<ReturnType<typeof useNodeDefStore>> as ReturnType<
        typeof useNodeDefStore
      >)

      const { missingCoreNodes } = useMissingNodes()

      expect(Object.keys(missingCoreNodes.value)).toHaveLength(1)
      expect(missingCoreNodes.value['1.2.0']).toHaveLength(1)
      expect(missingCoreNodes.value['1.2.0'][0].type).toBe('CoreNode')
    })

    it('returns empty object when no core nodes are missing', () => {
      // Mock collectAllNodes to return empty array (no missing nodes after filtering)
      mockCollectAllNodes.mockReturnValue([])

      mockUseNodeDefStore.mockReturnValue({
        nodeDefsByName: {
          RegisteredNode1: {
            name: 'RegisteredNode1'
          } as Partial<ComfyNodeDefImpl> as ComfyNodeDefImpl,
          RegisteredNode2: {
            name: 'RegisteredNode2'
          } as Partial<ComfyNodeDefImpl> as ComfyNodeDefImpl
        }
      } as Partial<ReturnType<typeof useNodeDefStore>> as ReturnType<
        typeof useNodeDefStore
      >)

      const { missingCoreNodes } = useMissingNodes()

      expect(Object.keys(missingCoreNodes.value)).toHaveLength(0)
    })
  })

  describe('subgraph support', () => {
    const createMockNode = (
      type: string,
      packId?: string,
      version?: string
    ): LGraphNode =>
      createMockLGraphNode({
        type,
        properties: { cnr_id: packId, ver: version },
        id: 1,
        title: type,
        pos: [0, 0],
        size: [100, 100],
        flags: {},
        graph: null,
        mode: 0,
        inputs: [],
        outputs: []
      })

    it('detects missing core nodes from subgraphs via collectAllNodes', () => {
      const mainNode = createMockNode('MainNode', 'comfy-core', '1.0.0')
      const subgraphNode1 = createMockNode(
        'SubgraphNode1',
        'comfy-core',
        '1.0.0'
      )
      const subgraphNode2 = createMockNode(
        'SubgraphNode2',
        'comfy-core',
        '1.1.0'
      )

      // Mock collectAllNodes to return all nodes including subgraph nodes
      mockCollectAllNodes.mockReturnValue([
        mainNode,
        subgraphNode1,
        subgraphNode2
      ])

      // Mock none of the nodes as registered
      mockUseNodeDefStore.mockReturnValue({
        nodeDefsByName: {}
      } as Partial<ReturnType<typeof useNodeDefStore>> as ReturnType<
        typeof useNodeDefStore
      >)

      const { missingCoreNodes } = useMissingNodes()

      // Should detect all 3 nodes as missing
      expect(Object.keys(missingCoreNodes.value)).toHaveLength(2) // 2 versions: 1.0.0, 1.1.0
      expect(missingCoreNodes.value['1.0.0']).toHaveLength(2) // MainNode + SubgraphNode1
      expect(missingCoreNodes.value['1.1.0']).toHaveLength(1) // SubgraphNode2
    })

    it('calls collectAllNodes with the app graph and filter function', () => {
      const mockGraph = { nodes: [], subgraphs: new Map() }
      mockApp.rootGraph = mockGraph

      const { missingCoreNodes } = useMissingNodes()
      // Access the computed to trigger the function
      void missingCoreNodes.value

      expect(mockCollectAllNodes).toHaveBeenCalledWith(
        mockGraph,
        expect.any(Function)
      )
    })

    it('handles collectAllNodes returning empty array', () => {
      mockCollectAllNodes.mockReturnValue([])

      const { missingCoreNodes } = useMissingNodes()

      expect(Object.keys(missingCoreNodes.value)).toHaveLength(0)
    })

    it('filter function correctly identifies missing core nodes', () => {
      const mockGraph = { nodes: [], subgraphs: new Map() }
      mockApp.rootGraph = mockGraph

      mockUseNodeDefStore.mockReturnValue({
        nodeDefsByName: {
          RegisteredCore: {
            name: 'RegisteredCore'
          } as Partial<ComfyNodeDefImpl> as ComfyNodeDefImpl
        }
      } as Partial<ReturnType<typeof useNodeDefStore>> as ReturnType<
        typeof useNodeDefStore
      >)

      let capturedFilterFunction: ((node: LGraphNode) => boolean) | undefined

      mockCollectAllNodes.mockImplementation((_graph, filter) => {
        capturedFilterFunction = filter
        return []
      })

      const { missingCoreNodes } = useMissingNodes()
      void missingCoreNodes.value

      expect(capturedFilterFunction).toBeDefined()

      if (capturedFilterFunction) {
        const missingCoreNode = createMockNode(
          'MissingCore',
          'comfy-core',
          '1.0.0'
        )
        const registeredCoreNode = createMockNode(
          'RegisteredCore',
          'comfy-core',
          '1.0.0'
        )
        const customNode = createMockNode('CustomNode', 'custom-pack', '1.0.0')
        const nodeWithoutPack = createMockNode('NodeWithoutPack')

        expect(capturedFilterFunction(missingCoreNode)).toBe(true)
        expect(capturedFilterFunction(registeredCoreNode)).toBe(false)
        expect(capturedFilterFunction(customNode)).toBe(false)
        expect(capturedFilterFunction(nodeWithoutPack)).toBe(false)
      }
    })

    it('integrates with collectAllNodes to find nodes from subgraphs', () => {
      mockCollectAllNodes.mockImplementation((graph, filter) => {
        const allNodes: LGraphNode[] = []

        for (const node of graph.nodes) {
          if (node.isSubgraphNode?.() && node.subgraph) {
            for (const subNode of node.subgraph.nodes) {
              if (!filter || filter(subNode)) {
                allNodes.push(subNode)
              }
            }
          }

          if (!filter || filter(node)) {
            allNodes.push(node)
          }
        }

        return allNodes
      })

      const mainMissingNode = createMockNode(
        'MainMissing',
        'comfy-core',
        '1.0.0'
      )
      const subgraphMissingNode = createMockNode(
        'SubgraphMissing',
        'comfy-core',
        '1.1.0'
      )
      const subgraphRegisteredNode = createMockNode(
        'SubgraphRegistered',
        'comfy-core',
        '1.0.0'
      )

      const mockSubgraph = {
        nodes: [subgraphMissingNode, subgraphRegisteredNode]
      }

      const mockSubgraphNode = createMockLGraphNode({
        isSubgraphNode: () => true,
        subgraph: mockSubgraph,
        type: 'SubgraphContainer',
        properties: { cnr_id: 'custom-pack' }
      })

      const mockMainGraph = {
        nodes: [mainMissingNode, mockSubgraphNode]
      } as Partial<LGraph> as LGraph

      mockApp.rootGraph = mockMainGraph

      mockUseNodeDefStore.mockReturnValue({
        nodeDefsByName: {
          SubgraphRegistered: {
            name: 'SubgraphRegistered'
          } as Partial<ComfyNodeDefImpl> as ComfyNodeDefImpl
        }
      } as Partial<ReturnType<typeof useNodeDefStore>> as ReturnType<
        typeof useNodeDefStore
      >)

      const { missingCoreNodes } = useMissingNodes()

      expect(Object.keys(missingCoreNodes.value)).toHaveLength(2)
      expect(missingCoreNodes.value['1.0.0']).toHaveLength(1)
      expect(missingCoreNodes.value['1.1.0']).toHaveLength(1)
      expect(missingCoreNodes.value['1.0.0'][0].type).toBe('MainMissing')
      expect(missingCoreNodes.value['1.1.0'][0].type).toBe('SubgraphMissing')
    })
  })
})
