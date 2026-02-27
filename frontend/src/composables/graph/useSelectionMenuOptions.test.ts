import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useSelectionMenuOptions } from '@/composables/graph/useSelectionMenuOptions'

const mocks = vi.hoisted(() => ({
  convertToSubgraph: vi.fn(),
  unpackSubgraph: vi.fn(),
  addSubgraphToLibrary: vi.fn(),
  frameNodes: vi.fn(),
  createI18nMock: vi.fn(() => ({
    global: {
      t: vi.fn(),
      te: vi.fn(),
      d: vi.fn()
    }
  }))
}))

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: (key: string) => key
  }),
  createI18n: mocks.createI18nMock
}))

vi.mock('@/composables/graph/useSelectionOperations', () => ({
  useSelectionOperations: () => ({
    copySelection: vi.fn(),
    duplicateSelection: vi.fn(),
    deleteSelection: vi.fn(),
    renameSelection: vi.fn()
  })
}))

vi.mock('@/composables/graph/useNodeArrangement', () => ({
  useNodeArrangement: () => ({
    alignOptions: [{ localizedName: 'align-left', icon: 'align-left' }],
    distributeOptions: [{ localizedName: 'distribute', icon: 'distribute' }],
    applyAlign: vi.fn(),
    applyDistribute: vi.fn()
  })
}))

vi.mock('@/composables/graph/useSubgraphOperations', () => ({
  useSubgraphOperations: () => ({
    convertToSubgraph: mocks.convertToSubgraph,
    unpackSubgraph: mocks.unpackSubgraph,
    addSubgraphToLibrary: mocks.addSubgraphToLibrary
  })
}))

vi.mock('@/composables/graph/useFrameNodes', () => ({
  useFrameNodes: () => ({
    frameNodes: mocks.frameNodes
  })
}))

describe('useSelectionMenuOptions - multiple nodes options', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns Frame Nodes option that invokes frameNodes when called', () => {
    const { getMultipleNodesOptions } = useSelectionMenuOptions()
    const options = getMultipleNodesOptions()

    const frameOption = options.find((opt) => opt.label === 'g.frameNodes')
    expect(frameOption).toBeDefined()
    expect(frameOption?.action).toBeDefined()

    frameOption?.action?.()
    expect(mocks.frameNodes).toHaveBeenCalledOnce()
  })

  it('returns Convert to Group Node option from getMultipleNodesOptions', () => {
    const { getMultipleNodesOptions } = useSelectionMenuOptions()
    const options = getMultipleNodesOptions()

    const groupNodeOption = options.find(
      (opt) => opt.label === 'contextMenu.Convert to Group Node'
    )
    expect(groupNodeOption).toBeDefined()
  })
})

describe('useSelectionMenuOptions - subgraph options', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns only convert option when no subgraphs are selected', () => {
    const { getSubgraphOptions } = useSelectionMenuOptions()
    const options = getSubgraphOptions({
      hasSubgraphs: false,
      hasMultipleSelection: true
    })

    expect(options).toHaveLength(1)
    expect(options[0]?.label).toBe('contextMenu.Convert to Subgraph')
    expect(options[0]?.action).toBe(mocks.convertToSubgraph)
  })

  it('includes convert and unpack but hides add to library when multiple items with subgraphs are selected', () => {
    const { getSubgraphOptions } = useSelectionMenuOptions()
    const options = getSubgraphOptions({
      hasSubgraphs: true,
      hasMultipleSelection: true
    })
    const labels = options.map((option) => option.label)

    expect(labels).toContain('contextMenu.Convert to Subgraph')
    expect(labels).not.toContain('contextMenu.Add Subgraph to Library')
    expect(labels).toContain('contextMenu.Unpack Subgraph')
  })

  it('shows add to library and unpack when a single subgraph is selected', () => {
    const { getSubgraphOptions } = useSelectionMenuOptions()
    const options = getSubgraphOptions({
      hasSubgraphs: true,
      hasMultipleSelection: false
    })

    const labels = options.map((option) => option.label)
    expect(labels).not.toContain('contextMenu.Convert to Subgraph')
    expect(labels).toEqual([
      'contextMenu.Add Subgraph to Library',
      'contextMenu.Unpack Subgraph'
    ])

    const addToLibraryOption = options.find(
      (option) => option.label === 'contextMenu.Add Subgraph to Library'
    )
    expect(addToLibraryOption?.action).toBe(mocks.addSubgraphToLibrary)
  })
})
