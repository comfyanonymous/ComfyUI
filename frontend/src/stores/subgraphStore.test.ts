import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ComfyNodeDef as ComfyNodeDefV1 } from '@/schemas/nodeDefSchema'
import type { GlobalSubgraphData } from '@/scripts/api'
import type { ExportedSubgraph } from '@/lib/litegraph/src/types/serialisation'
import { TemplateIncludeOnDistributionEnum } from '@/platform/workflow/templates/types/template'
import { api } from '@/scripts/api'
import { app as comfyApp } from '@/scripts/app'
import { useLitegraphService } from '@/services/litegraphService'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { useSubgraphStore } from '@/stores/subgraphStore'

import {
  createTestSubgraph,
  createTestSubgraphNode
} from '@/lib/litegraph/src/subgraph/__fixtures__/subgraphHelpers'
import { createTestingPinia } from '@pinia/testing'

const mockDistributionTypes = vi.hoisted(() => ({
  isCloud: false,
  isDesktop: false
}))
vi.mock('@/platform/distribution/types', () => mockDistributionTypes)

// Mock telemetry to break circular dependency (telemetry → workflowStore → app → telemetry)
vi.mock('@/platform/telemetry', () => ({
  useTelemetry: () => null
}))

// Add mock for api at the top of the file
vi.mock('@/scripts/api', () => ({
  api: {
    getUserData: vi.fn(),
    storeUserData: vi.fn(),
    listUserDataFullInfo: vi.fn(),
    getGlobalSubgraphs: vi.fn(),
    apiURL: vi.fn(),
    addEventListener: vi.fn()
  }
}))
vi.mock('@/services/dialogService', () => ({
  useDialogService: vi.fn(() => ({
    prompt: () => 'testname',
    confirm: () => true
  }))
}))
vi.mock('@/renderer/core/canvas/canvasStore', () => ({
  useCanvasStore: vi.fn(() => ({
    getCanvas: () => comfyApp.canvas
  }))
}))

// Mock comfyApp globally for the store setup
vi.mock('@/scripts/app', () => ({
  app: {
    canvas: {
      _deserializeItems: vi.fn((i) => i),
      ds: { visible_area: [0, 0, 0, 0] },
      selected_nodes: null
    },
    loadGraphData: vi.fn()
  }
}))

const mockGraph = {
  nodes: [{ type: '123' }],
  definitions: { subgraphs: [{ id: '123' }] }
}

describe('useSubgraphStore', () => {
  let store: ReturnType<typeof useSubgraphStore>
  async function mockFetch(
    filenames: Record<string, unknown>,
    globalSubgraphs: Record<string, GlobalSubgraphData> = {}
  ) {
    vi.mocked(api.listUserDataFullInfo).mockResolvedValue(
      Object.keys(filenames).map((filename) => ({
        path: filename,
        modified: new Date().getTime(),
        size: 1 // size !== -1 for remote workflows
      }))
    )
    vi.mocked(api).getUserData = vi.fn((f) =>
      Promise.resolve({
        status: 200,
        text: () => Promise.resolve(JSON.stringify(filenames[f.slice(10)]))
      } as Response)
    )
    vi.mocked(api.getGlobalSubgraphs).mockResolvedValue(globalSubgraphs)
    return await store.fetchSubgraphs()
  }

  beforeEach(() => {
    mockDistributionTypes.isCloud = false
    mockDistributionTypes.isDesktop = false
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useSubgraphStore()
    vi.clearAllMocks()
  })

  it('should allow publishing of a subgraph', async () => {
    //mock canvas to provide a minimal subgraphNode
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)
    const graph = subgraphNode.graph!
    graph.add(subgraphNode)
    vi.mocked(comfyApp.canvas).selectedItems = new Set([subgraphNode])
    vi.mocked(comfyApp.canvas)._serializeItems = vi.fn(() => {
      const serializedSubgraph = {
        ...subgraph.serialize(),
        links: [],
        groups: [],
        version: 1
      } as Partial<ExportedSubgraph> as ExportedSubgraph
      return {
        nodes: [subgraphNode.serialize()],
        subgraphs: [serializedSubgraph]
      }
    })
    //mock saving of file
    vi.mocked(api.storeUserData).mockResolvedValue({
      status: 200,
      json: () =>
        Promise.resolve({
          path: 'subgraphs/testname.json',
          modified: Date.now(),
          size: 2
        })
    } as Response)
    await mockFetch({ 'testname.json': mockGraph })
    //Dialogue service already mocked
    await store.publishSubgraph()
    expect(api.storeUserData).toHaveBeenCalled()
  })
  it('should display published nodes in the node library', async () => {
    await mockFetch({ 'test.json': mockGraph })
    expect(
      useNodeDefStore().nodeDefs.filter(
        (d) => d.category === 'Subgraph Blueprints/User'
      )
    ).toHaveLength(1)
  })
  it('should allow subgraphs to be edited', async () => {
    await mockFetch({ 'test.json': mockGraph })
    await store.editBlueprint(store.typePrefix + 'test')
    //check active graph
    expect(comfyApp.loadGraphData).toHaveBeenCalled()
  })
  it('should allow subgraphs to be added to graph', async () => {
    //mock
    await mockFetch({ 'test.json': mockGraph })
    const res = useLitegraphService().addNodeOnGraph({
      name: 'SubgraphBlueprint.test'
    } as ComfyNodeDefV1)
    expect(res).toBeTruthy()
  })
  it('should identify user blueprints as non-global', async () => {
    await mockFetch({ 'test.json': mockGraph })
    expect(store.isGlobalBlueprint('test')).toBe(false)
  })
  it('should identify global blueprints loaded from getGlobalSubgraphs', async () => {
    await mockFetch(
      {},
      {
        global_test: {
          name: 'Global Test Blueprint',
          info: { node_pack: 'comfy_essentials' },
          data: JSON.stringify(mockGraph)
        }
      }
    )
    expect(store.isGlobalBlueprint('global_test')).toBe(true)
  })
  it('should return false for non-existent blueprints', async () => {
    await mockFetch({ 'test.json': mockGraph })
    expect(store.isGlobalBlueprint('nonexistent')).toBe(false)
  })

  describe('blueprint badge display', () => {
    it('should set isGlobal flag on global blueprints', async () => {
      await mockFetch(
        {},
        {
          global_bp: {
            name: 'Global Blueprint',
            info: { node_pack: 'some-uuid-string' },
            data: JSON.stringify(mockGraph)
          }
        }
      )
      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.global_bp'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.isGlobal).toBe(true)
    })

    it('should not set isGlobal flag on user blueprints', async () => {
      await mockFetch({ 'user-blueprint.json': mockGraph })
      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.user-blueprint'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.isGlobal).toBeUndefined()
    })

    it('should use blueprint python_module for global blueprints to show Blueprint badge', async () => {
      await mockFetch(
        {},
        {
          global_bp: {
            name: 'Global Blueprint',
            info: { node_pack: 'comfyui-ltx-video-0fbc55c6-long-uuid' },
            data: JSON.stringify(mockGraph)
          }
        }
      )
      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.global_bp'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.python_module).toBe('blueprint')
      expect(nodeDef?.nodeSource.displayText).toBe('Blueprint')
    })
  })

  it('should handle global blueprint with empty data gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    await mockFetch(
      {},
      {
        broken_blueprint: {
          name: 'Broken Blueprint',
          info: { node_pack: 'test_pack' },
          data: ''
        }
      }
    )
    expect(consoleSpy).toHaveBeenCalledWith(
      'Failed to load subgraph blueprint',
      expect.any(Error)
    )
    expect(store.subgraphBlueprints).toHaveLength(0)
    consoleSpy.mockRestore()
  })

  it('should handle global blueprint with rejected data promise gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    await mockFetch(
      {},
      {
        failing_blueprint: {
          name: 'Failing Blueprint',
          info: { node_pack: 'test_pack' },
          data: Promise.reject(new Error('Network error')) as unknown as string
        }
      }
    )
    expect(consoleSpy).toHaveBeenCalledWith(
      'Failed to load subgraph blueprint',
      expect.any(Error)
    )
    expect(store.subgraphBlueprints).toHaveLength(0)
    consoleSpy.mockRestore()
  })

  it('should load valid global blueprints even when others fail', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    await mockFetch(
      {},
      {
        broken: {
          name: 'Broken',
          info: { node_pack: 'test_pack' },
          data: ''
        },
        valid: {
          name: 'Valid Blueprint',
          info: { node_pack: 'test_pack' },
          data: JSON.stringify(mockGraph)
        }
      }
    )
    expect(consoleSpy).toHaveBeenCalled()
    expect(store.subgraphBlueprints).toHaveLength(1)
    consoleSpy.mockRestore()
  })

  describe('search_aliases support', () => {
    it('should include search_aliases from workflow extra', async () => {
      const mockGraphWithAliases = {
        nodes: [{ type: '123' }],
        definitions: {
          subgraphs: [{ id: '123' }]
        },
        extra: {
          BlueprintSearchAliases: ['alias1', 'alias2', 'my workflow']
        }
      }
      await mockFetch({ 'test-with-aliases.json': mockGraphWithAliases })

      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.test-with-aliases'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.search_aliases).toEqual([
        'alias1',
        'alias2',
        'my workflow'
      ])
    })

    it('should include search_aliases from global blueprint info', async () => {
      await mockFetch(
        {},
        {
          global_with_aliases: {
            name: 'Global With Aliases',
            info: {
              node_pack: 'comfy_essentials',
              search_aliases: ['global alias', 'test alias']
            },
            data: JSON.stringify(mockGraph)
          }
        }
      )

      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.global_with_aliases'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.search_aliases).toEqual(['global alias', 'test alias'])
    })

    it('should not have search_aliases if not provided', async () => {
      await mockFetch({ 'test.json': mockGraph })

      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.test'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.search_aliases).toBeUndefined()
    })

    it('should include description from workflow extra', async () => {
      const mockGraphWithDescription = {
        nodes: [{ type: '123' }],
        definitions: {
          subgraphs: [{ id: '123' }]
        },
        extra: {
          BlueprintDescription: 'This is a test blueprint'
        }
      }
      await mockFetch({
        'test-with-description.json': mockGraphWithDescription
      })

      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.test-with-description'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.description).toBe('This is a test blueprint')
    })

    it('should not duplicate metadata in both workflow extra and subgraph extra when publishing', async () => {
      const subgraph = createTestSubgraph()
      const subgraphNode = createTestSubgraphNode(subgraph)
      const graph = subgraphNode.graph!
      graph.add(subgraphNode)

      // Set metadata on the subgraph's extra (as the commands do)
      subgraph.extra = {
        BlueprintDescription: 'Test description',
        BlueprintSearchAliases: ['alias1', 'alias2']
      }

      vi.mocked(comfyApp.canvas).selectedItems = new Set([subgraphNode])
      vi.mocked(comfyApp.canvas)._serializeItems = vi.fn(() => {
        const serializedSubgraph = {
          ...subgraph.serialize(),
          links: [],
          groups: [],
          version: 1
        } as Partial<ExportedSubgraph> as ExportedSubgraph
        return {
          nodes: [subgraphNode.serialize()],
          subgraphs: [serializedSubgraph]
        }
      })

      let savedWorkflowData: Record<string, unknown> | null = null
      vi.mocked(api.storeUserData).mockImplementation(async (_path, data) => {
        savedWorkflowData = JSON.parse(data as string)
        return {
          status: 200,
          json: () =>
            Promise.resolve({
              path: 'subgraphs/testname.json',
              modified: Date.now(),
              size: 2
            })
        } as Response
      })

      await mockFetch({ 'testname.json': mockGraph })
      await store.publishSubgraph()

      expect(savedWorkflowData).not.toBeNull()

      // Metadata should be in top-level extra
      expect(savedWorkflowData!.extra).toEqual({
        BlueprintDescription: 'Test description',
        BlueprintSearchAliases: ['alias1', 'alias2']
      })

      // Metadata should NOT be in subgraph's extra
      const definitions = savedWorkflowData!.definitions as {
        subgraphs: Array<{ extra?: Record<string, unknown> }>
      }
      const subgraphExtra = definitions.subgraphs[0]?.extra
      expect(subgraphExtra?.BlueprintDescription).toBeUndefined()
      expect(subgraphExtra?.BlueprintSearchAliases).toBeUndefined()
    })
  })

  describe('subgraph definition category', () => {
    it('should use category from subgraph definition as default', async () => {
      const mockGraphWithCategory = {
        nodes: [{ type: '123' }],
        definitions: {
          subgraphs: [{ id: '123', category: 'Image Processing' }]
        }
      }
      await mockFetch(
        {},
        {
          categorized: {
            name: 'Categorized Blueprint',
            info: { node_pack: 'test_pack' },
            data: JSON.stringify(mockGraphWithCategory)
          }
        }
      )

      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.categorized'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.category).toBe('Subgraph Blueprints/Image Processing')
    })

    it('should use User override for user blueprints even with definition category', async () => {
      const mockGraphWithCategory = {
        nodes: [{ type: '123' }],
        definitions: {
          subgraphs: [{ id: '123', category: 'Image Processing' }]
        }
      }
      await mockFetch({ 'user-bp.json': mockGraphWithCategory })

      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.user-bp'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.category).toBe('Subgraph Blueprints/User')
    })

    it('should fallback to bare Subgraph Blueprints when no category anywhere', async () => {
      await mockFetch(
        {},
        {
          no_cat_global: {
            name: 'No Category Global',
            info: { node_pack: 'test_pack' },
            data: JSON.stringify(mockGraph)
          }
        }
      )

      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.no_cat_global'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.category).toBe('Subgraph Blueprints')
    })

    it('should let overrides take precedence over definition category', async () => {
      const mockGraphWithCategory = {
        nodes: [{ type: '123' }],
        definitions: {
          subgraphs: [{ id: '123', category: 'Image Processing' }]
        }
      }
      await mockFetch(
        {},
        {
          bp_override: {
            name: 'Override Blueprint',
            info: {
              node_pack: 'test_pack',
              category: 'Custom Category'
            },
            data: JSON.stringify(mockGraphWithCategory)
          }
        }
      )

      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.bp_override'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.category).toBe('Subgraph Blueprints/Custom Category')
    })
  })

  describe('essentials_category passthrough', () => {
    it('should prefer GlobalSubgraphData essentials_category over definition fallback', async () => {
      const graphWithEssentials = {
        ...mockGraph,
        definitions: {
          subgraphs: [
            {
              ...mockGraph.definitions?.subgraphs?.[0],
              essentials_category: 'Image Tools'
            }
          ]
        }
      }
      await mockFetch(
        {},
        {
          bp_precedence: {
            name: 'Precedence Blueprint',
            info: { node_pack: 'test_pack' },
            data: JSON.stringify(graphWithEssentials),
            essentials_category: 'Video Generation'
          }
        }
      )
      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.bp_precedence'
      )
      expect(nodeDef?.essentials_category).toBe('video generation')
    })

    it('should pass essentials_category from GlobalSubgraphData to node def', async () => {
      await mockFetch(
        {},
        {
          bp_essentials: {
            name: 'Test Essentials Blueprint',
            info: { node_pack: 'test_pack', category: 'Test Category' },
            data: JSON.stringify(mockGraph),
            essentials_category: 'Image Generation'
          }
        }
      )
      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.bp_essentials'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.essentials_category).toBe('image generation')
    })

    it('should extract essentials_category from subgraph definition as fallback', async () => {
      const graphWithEssentials = {
        ...mockGraph,
        definitions: {
          subgraphs: [
            {
              ...mockGraph.definitions?.subgraphs?.[0],
              essentials_category: 'Image Tools'
            }
          ]
        }
      }
      await mockFetch(
        {},
        {
          bp_fallback: {
            name: 'Fallback Blueprint',
            info: { node_pack: 'test_pack' },
            data: JSON.stringify(graphWithEssentials)
          }
        }
      )
      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.bp_fallback'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.essentials_category).toBe('image tools')
    })

    it('should normalize title-cased essentials_category to canonical form', async () => {
      await mockFetch(
        {},
        {
          bp_3d: {
            name: 'Test 3D Blueprint',
            info: { node_pack: 'test_pack', category: 'Test Category' },
            data: JSON.stringify(mockGraph),
            essentials_category: '3d'
          }
        }
      )
      const nodeDef = useNodeDefStore().nodeDefs.find(
        (d) => d.name === 'SubgraphBlueprint.bp_3d'
      )
      expect(nodeDef).toBeDefined()
      expect(nodeDef?.essentials_category).toBe('3D')
    })
  })

  describe('global blueprint filtering', () => {
    function globalBlueprint(
      overrides: Partial<GlobalSubgraphData['info']> = {}
    ): GlobalSubgraphData {
      return {
        name: 'Filtered Blueprint',
        info: { node_pack: 'test_pack', ...overrides },
        data: JSON.stringify(mockGraph)
      }
    }

    it('should exclude blueprints with requiresCustomNodes on non-cloud', async () => {
      await mockFetch(
        {},
        {
          bp: globalBlueprint({ requiresCustomNodes: ['custom-node-pack'] })
        }
      )
      expect(store.isGlobalBlueprint('bp')).toBe(false)
    })

    it('should include blueprints with requiresCustomNodes on cloud', async () => {
      mockDistributionTypes.isCloud = true
      await mockFetch(
        {},
        {
          bp: globalBlueprint({ requiresCustomNodes: ['custom-node-pack'] })
        }
      )
      expect(store.isGlobalBlueprint('bp')).toBe(true)
    })

    it('should include blueprints with empty requiresCustomNodes everywhere', async () => {
      await mockFetch({}, { bp: globalBlueprint({ requiresCustomNodes: [] }) })
      expect(store.isGlobalBlueprint('bp')).toBe(true)
    })

    it('should exclude blueprints whose includeOnDistributions does not match', async () => {
      await mockFetch(
        {},
        {
          bp: globalBlueprint({
            includeOnDistributions: [TemplateIncludeOnDistributionEnum.Cloud]
          })
        }
      )
      expect(store.isGlobalBlueprint('bp')).toBe(false)
    })

    it('should include blueprints whose includeOnDistributions matches current distribution', async () => {
      await mockFetch(
        {},
        {
          bp: globalBlueprint({
            includeOnDistributions: [TemplateIncludeOnDistributionEnum.Local]
          })
        }
      )
      expect(store.isGlobalBlueprint('bp')).toBe(true)
    })

    it('should include blueprints on desktop when includeOnDistributions has desktop', async () => {
      mockDistributionTypes.isDesktop = true
      await mockFetch(
        {},
        {
          bp: globalBlueprint({
            includeOnDistributions: [TemplateIncludeOnDistributionEnum.Desktop]
          })
        }
      )
      expect(store.isGlobalBlueprint('bp')).toBe(true)
    })

    it('should include blueprints with no filtering fields', async () => {
      await mockFetch({}, { bp: globalBlueprint() })
      expect(store.isGlobalBlueprint('bp')).toBe(true)
    })
  })
})
