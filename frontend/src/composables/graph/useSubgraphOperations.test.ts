import { beforeEach, describe, expect, it, vi } from 'vitest'

import { SubgraphNode } from '@/lib/litegraph/src/litegraph'

const mocks = vi.hoisted(() => ({
  publishSubgraph: vi.fn(),
  selectedItems: [] as unknown[]
}))

vi.mock('@/composables/canvas/useSelectedLiteGraphItems', () => ({
  useSelectedLiteGraphItems: () => ({
    getSelectedNodes: vi.fn(() => [])
  })
}))

vi.mock('@/renderer/core/canvas/canvasStore', () => ({
  useCanvasStore: () => ({
    getCanvas: vi.fn(),
    get selectedItems() {
      return mocks.selectedItems
    },
    updateSelectedItems: vi.fn()
  })
}))

vi.mock('@/platform/workflow/management/stores/workflowStore', () => ({
  useWorkflowStore: () => ({
    activeWorkflow: null
  })
}))

vi.mock('@/stores/imagePreviewStore', () => ({
  useNodeOutputStore: () => ({
    revokeSubgraphPreviews: vi.fn()
  })
}))

vi.mock('@/stores/subgraphStore', () => ({
  useSubgraphStore: () => ({
    publishSubgraph: mocks.publishSubgraph
  })
}))

function createSubgraphNode(): SubgraphNode {
  const node = Object.create(SubgraphNode.prototype)
  return node
}

describe('useSubgraphOperations', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.selectedItems = []
  })

  it('addSubgraphToLibrary calls publishSubgraph when single SubgraphNode selected', async () => {
    mocks.selectedItems = [createSubgraphNode()]

    const { useSubgraphOperations } =
      await import('@/composables/graph/useSubgraphOperations')
    const { addSubgraphToLibrary } = useSubgraphOperations()

    await addSubgraphToLibrary()

    expect(mocks.publishSubgraph).toHaveBeenCalledOnce()
  })

  it('addSubgraphToLibrary does not call publishSubgraph when no items selected', async () => {
    mocks.selectedItems = []

    const { useSubgraphOperations } =
      await import('@/composables/graph/useSubgraphOperations')
    const { addSubgraphToLibrary } = useSubgraphOperations()

    await addSubgraphToLibrary()

    expect(mocks.publishSubgraph).not.toHaveBeenCalled()
  })

  it('addSubgraphToLibrary does not call publishSubgraph when multiple items selected', async () => {
    mocks.selectedItems = [createSubgraphNode(), createSubgraphNode()]

    const { useSubgraphOperations } =
      await import('@/composables/graph/useSubgraphOperations')
    const { addSubgraphToLibrary } = useSubgraphOperations()

    await addSubgraphToLibrary()

    expect(mocks.publishSubgraph).not.toHaveBeenCalled()
  })
})
