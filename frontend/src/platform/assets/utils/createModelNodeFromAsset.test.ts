// oxlint-disable no-misused-spread
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { markRaw } from 'vue'
import type { Raw } from 'vue'

import type { LGraphNode, Subgraph } from '@/lib/litegraph/src/litegraph'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import type * as LitegraphModule from '@/lib/litegraph/src/litegraph'
import type * as ModelToNodeStoreModule from '@/stores/modelToNodeStore'
import type * as WorkflowStoreModule from '@/platform/workflow/management/stores/workflowStore'
import type * as LitegraphServiceModule from '@/services/litegraphService'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { createModelNodeFromAsset } from '@/platform/assets/utils/createModelNodeFromAsset'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { app } from '@/scripts/app'
import { useLitegraphService } from '@/services/litegraphService'
import { useModelToNodeStore } from '@/stores/modelToNodeStore'

// Mock dependencies
vi.mock('@/scripts/api', () => ({
  api: {
    apiURL: vi.fn((path: string) => `http://localhost:8188${path}`)
  }
}))
vi.mock('@/stores/modelToNodeStore', async (importOriginal) => {
  const actual = await importOriginal<typeof ModelToNodeStoreModule>()
  return {
    ...actual,
    useModelToNodeStore: vi.fn()
  }
})
vi.mock(
  '@/platform/workflow/management/stores/workflowStore',
  async (importOriginal) => {
    const actual = await importOriginal<typeof WorkflowStoreModule>()
    return {
      ...actual,
      useWorkflowStore: vi.fn()
    }
  }
)
vi.mock('@/services/litegraphService', async (importOriginal) => {
  const actual = await importOriginal<typeof LitegraphServiceModule>()
  return {
    ...actual,
    useLitegraphService: vi.fn()
  }
})
vi.mock('@/lib/litegraph/src/litegraph', async (importOriginal) => {
  const actual = await importOriginal<typeof LitegraphModule>()
  return {
    ...actual,
    LiteGraph: {
      ...actual.LiteGraph,
      createNode: vi.fn()
    }
  }
})
vi.mock('@/scripts/app', () => ({
  app: {
    canvas: {
      graph: {
        add: vi.fn()
      }
    }
  }
}))
function createMockAsset(overrides: Partial<AssetItem> = {}): AssetItem {
  return {
    id: 'asset-123',
    name: 'test-model.safetensors',
    size: 1024,
    created_at: '2025-10-01T00:00:00Z',
    tags: ['models', 'checkpoints'],
    user_metadata: {
      filename: 'models/checkpoints/test-model.safetensors'
    },
    ...overrides
  }
}
async function createMockNode(overrides?: {
  widgetName?: string
  widgetValue?: string
  hasWidgets?: boolean
}): Promise<LGraphNode> {
  const {
    widgetName = 'ckpt_name',
    widgetValue = '',
    hasWidgets = true
  } = overrides || {}

  const { LGraphNode: ActualLGraphNode } = await vi.importActual<
    typeof LitegraphModule
  >('@/lib/litegraph/src/litegraph')

  if (!hasWidgets) {
    return Object.create(ActualLGraphNode.prototype)
  }

  type Widget = NonNullable<LGraphNode['widgets']>[number]
  const widget: Pick<Widget, 'name' | 'value' | 'type' | 'options' | 'y'> = {
    name: widgetName,
    value: widgetValue,
    type: 'string',
    options: {},
    y: 0
  }

  return Object.create(ActualLGraphNode.prototype, {
    widgets: { value: [widget], writable: true }
  })
}
function createMockNodeProvider(
  overrides: {
    nodeDef?: { name: string; display_name: string }
    key?: string
  } = {}
) {
  return {
    nodeDef: {
      name: 'CheckpointLoaderSimple',
      display_name: 'Load Checkpoint',
      ...overrides.nodeDef
    },
    key: overrides.key ?? 'ckpt_name'
  }
}
/**
 * Configures all mocked dependencies with sensible defaults.
 * Uses semantic parameters for clearer test intent.
 * For error paths or edge cases, pass null values or specific overrides.
 */
async function setupMocks(
  overrides: {
    nodeProvider?: ReturnType<typeof createMockNodeProvider> | null
    canvasCenter?: [number, number]
    activeSubgraph?: Raw<Subgraph>
    createdNode?: Awaited<ReturnType<typeof createMockNode>> | null
  } = {}
) {
  const {
    nodeProvider = createMockNodeProvider(),
    canvasCenter = [100, 200],
    activeSubgraph = undefined,
    createdNode = await createMockNode()
  } = overrides

  vi.mocked(useModelToNodeStore).mockReturnValue({
    ...useModelToNodeStore(),
    getNodeProvider: vi.fn().mockReturnValue(nodeProvider)
  })

  vi.mocked(useLitegraphService).mockReturnValue({
    ...useLitegraphService(),
    getCanvasCenter: vi.fn().mockReturnValue(canvasCenter)
  })

  vi.mocked(useWorkflowStore).mockReturnValue({
    ...useWorkflowStore(),
    activeSubgraph,
    isSubgraphActive: !!activeSubgraph
  })
  vi.mocked(LiteGraph.createNode).mockReturnValue(createdNode)
}
describe('createModelNodeFromAsset', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.spyOn(console, 'warn').mockImplementation(() => {})
  })
  describe('when creating nodes from valid assets', () => {
    it('should create the appropriate loader node for the asset category', async () => {
      const asset = createMockAsset()
      await setupMocks()
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(true)
      if (result.success) {
        expect(
          vi.mocked(useModelToNodeStore)().getNodeProvider
        ).toHaveBeenCalledWith('checkpoints')
        expect(LiteGraph.createNode).toHaveBeenCalledWith(
          'CheckpointLoaderSimple',
          'Load Checkpoint',
          { pos: [100, 200] }
        )
      }
    })
    it('should place node at canvas center by default', async () => {
      const asset = createMockAsset()
      await setupMocks({
        canvasCenter: [150, 250]
      })
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(true)
      expect(
        vi.mocked(useLitegraphService)().getCanvasCenter
      ).toHaveBeenCalled()
      expect(LiteGraph.createNode).toHaveBeenCalledWith(
        'CheckpointLoaderSimple',
        'Load Checkpoint',
        { pos: [150, 250] }
      )
    })
    it('should place node at specified position when position is provided', async () => {
      const asset = createMockAsset()
      await setupMocks()
      const result = createModelNodeFromAsset(asset, { position: [300, 400] })
      expect(result.success).toBe(true)
      expect(
        vi.mocked(useLitegraphService)().getCanvasCenter
      ).not.toHaveBeenCalled()
      expect(LiteGraph.createNode).toHaveBeenCalledWith(
        'CheckpointLoaderSimple',
        'Load Checkpoint',
        { pos: [300, 400] }
      )
    })
    it('should populate the loader widget with the asset file path', async () => {
      const asset = createMockAsset()
      const mockNode = await createMockNode()
      await setupMocks({ createdNode: mockNode })
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(true)
      expect(mockNode.widgets?.[0].value).toBe(
        'models/checkpoints/test-model.safetensors'
      )
    })
    it('should add node to root graph when no subgraph is active', async () => {
      const asset = createMockAsset()
      const mockNode = await createMockNode()
      await setupMocks({ createdNode: mockNode })
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(true)
      expect(vi.mocked(app).canvas.graph!.add).toHaveBeenCalledWith(mockNode)
    })
    it('should fallback to asset.metadata.filename when user_metadata.filename missing', async () => {
      const asset = createMockAsset({
        user_metadata: {},
        metadata: { filename: 'models/checkpoints/from-metadata.safetensors' }
      })
      const mockNode = await createMockNode()
      await setupMocks({ createdNode: mockNode })
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(true)
      expect(mockNode.widgets?.[0].value).toBe(
        'models/checkpoints/from-metadata.safetensors'
      )
    })
    it('should fallback to asset.name when both filename sources missing', async () => {
      const asset = createMockAsset({
        user_metadata: {},
        metadata: undefined
      })
      const mockNode = await createMockNode()
      await setupMocks({ createdNode: mockNode })
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(true)
      expect(mockNode.widgets?.[0].value).toBe('test-model.safetensors')
    })
    it('should add node to active subgraph when present', async () => {
      const asset = createMockAsset()
      const mockNode = await createMockNode()
      const { Subgraph } = await vi.importActual<typeof LitegraphModule>(
        '@/lib/litegraph/src/litegraph'
      )
      const mockSubgraph = markRaw(
        Object.create(Subgraph.prototype, {
          add: { value: vi.fn() }
        })
      )
      await setupMocks({
        createdNode: mockNode,
        activeSubgraph: mockSubgraph
      })
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(true)
      expect(mockSubgraph.add).toHaveBeenCalledWith(mockNode)
      expect(vi.mocked(app).canvas.graph!.add).not.toHaveBeenCalled()
    })
    it('should succeed when provider has empty key (auto-load nodes)', async () => {
      const asset = createMockAsset({
        tags: ['models', 'chatterbox/chatterbox_vc'],
        user_metadata: { filename: 'chatterbox_vc_model.pt' }
      })
      const mockNode = await createMockNode({ hasWidgets: false })
      const nodeProvider = createMockNodeProvider({
        nodeDef: {
          name: 'FL_ChatterboxVC',
          display_name: 'FL Chatterbox VC'
        },
        key: ''
      })
      await setupMocks({ createdNode: mockNode, nodeProvider })
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(true)
      expect(vi.mocked(app).canvas.graph!.add).toHaveBeenCalledWith(mockNode)
    })
  })
  describe('when asset data is incomplete or invalid', () => {
    beforeEach(() => {
      vi.spyOn(console, 'error').mockImplementation(() => {})
    })
    it.each([
      {
        case: 'missing user_metadata with no fallback',
        overrides: { user_metadata: undefined, metadata: undefined, name: '' },
        expectedCode: 'INVALID_ASSET' as const,
        errorPattern: /Invalid filename.*expected non-empty string/
      },
      {
        case: 'empty filename with no fallback',
        overrides: {
          user_metadata: { filename: '' },
          metadata: undefined,
          name: ''
        },
        expectedCode: 'INVALID_ASSET' as const,
        errorPattern: /Invalid filename.*expected non-empty string/
      }
    ])(
      'should fail when asset has $case',
      ({ overrides, expectedCode, errorPattern }) => {
        const asset = createMockAsset(overrides)
        const result = createModelNodeFromAsset(asset)
        expect(result.success).toBe(false)
        if (!result.success) {
          expect(result.error.code).toBe(expectedCode)
          expect(result.error.message).toMatch(errorPattern)
          expect(result.error.assetId).toBe('asset-123')
        }
      }
    )
    it.each([
      {
        case: 'no tags',
        overrides: { tags: undefined },
        expectedCode: 'INVALID_ASSET' as const,
        errorMessage: 'Asset has no tags defined'
      },
      {
        case: 'only excluded tags',
        overrides: { tags: ['models', 'missing'] },
        expectedCode: 'INVALID_ASSET' as const,
        errorMessage: 'Asset has no valid category tag'
      },
      {
        case: 'only the models tag',
        overrides: { tags: ['models'] },
        expectedCode: 'INVALID_ASSET' as const,
        errorMessage: 'Asset has no valid category tag'
      }
    ])(
      'should fail when asset has $case',
      ({ overrides, expectedCode, errorMessage }) => {
        const asset = createMockAsset(overrides)
        const result = createModelNodeFromAsset(asset)
        expect(result.success).toBe(false)
        if (!result.success) {
          expect(result.error.code).toBe(expectedCode)
          expect(result.error.message).toBe(errorMessage)
        }
      }
    )
  })
  describe('when system resources are unavailable', () => {
    beforeEach(() => {
      vi.spyOn(console, 'error').mockImplementation(() => {})
    })
    it('should fail when no provider registered for category', async () => {
      const asset = createMockAsset()
      await setupMocks({ nodeProvider: null })
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(false)
      if (!result.success) {
        expect(result.error.code).toBe('NO_PROVIDER')
        expect(result.error.message).toContain('checkpoints')
        expect(result.error.details?.category).toBe('checkpoints')
      }
    })
    it('should fail when node creation fails', async () => {
      const asset = createMockAsset()
      await setupMocks()
      vi.mocked(LiteGraph.createNode).mockReturnValue(null)
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(false)
      if (!result.success) {
        expect(result.error.code).toBe('NODE_CREATION_FAILED')
        expect(result.error.message).toContain('CheckpointLoaderSimple')
      }
    })
    it('should fail when widget is missing from node', async () => {
      const asset = createMockAsset()
      const mockNode = await createMockNode({ widgetName: 'wrong_widget' })
      await setupMocks({ createdNode: mockNode })
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(false)
      if (!result.success) {
        expect(result.error.code).toBe('MISSING_WIDGET')
        expect(result.error.message).toContain('ckpt_name')
        expect(result.error.message).toContain('CheckpointLoaderSimple')
        expect(result.error.details?.widgetName).toBe('ckpt_name')
      }
    })
    it('should fail when node has no widgets array', async () => {
      const asset = createMockAsset()
      const mockNode = await createMockNode({ hasWidgets: false })
      await setupMocks({ createdNode: mockNode })
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(false)
      if (!result.success) {
        expect(result.error.code).toBe('MISSING_WIDGET')
        expect(result.error.message).toContain('ckpt_name not found')
      }
    })
    it('should not add node to graph when widget validation fails', async () => {
      const asset = createMockAsset()
      const mockNode = await createMockNode({ hasWidgets: false })
      await setupMocks({ createdNode: mockNode })
      createModelNodeFromAsset(asset)
      expect(vi.mocked(app).canvas.graph!.add).not.toHaveBeenCalled()
    })
  })
  describe('when graph is null', () => {
    beforeEach(() => {
      vi.spyOn(console, 'error').mockImplementation(() => {})
      vi.mocked(app).canvas.graph = null
    })
    it('should fail when no graph is available', async () => {
      const asset = createMockAsset()
      const mockNode = await createMockNode()
      await setupMocks({ createdNode: mockNode })
      const result = createModelNodeFromAsset(asset)
      expect(result.success).toBe(false)
      if (!result.success) {
        expect(result.error.code).toBe('NO_GRAPH')
        expect(result.error.message).toBe('No active graph available')
      }
    })
  })
})
