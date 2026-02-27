import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useLayoutMutations } from '@/renderer/core/layout/operations/layoutMutations'
import { LayoutSource } from '@/renderer/core/layout/types'
import { useNodeZIndex } from '@/renderer/extensions/vueNodes/composables/useNodeZIndex'

// Mock the layout mutations module
vi.mock('@/renderer/core/layout/operations/layoutMutations', () => ({
  useLayoutMutations: vi.fn()
}))

const mockedUseLayoutMutations = vi.mocked(useLayoutMutations)

describe('useNodeZIndex', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('bringNodeToFront', () => {
    it('should bring node to front with default source', () => {
      const mockSetSource = vi.fn()
      const mockBringNodeToFront = vi.fn()

      mockedUseLayoutMutations.mockReturnValue({
        setSource: mockSetSource,
        bringNodeToFront: mockBringNodeToFront
      } as Partial<ReturnType<typeof useLayoutMutations>> as ReturnType<
        typeof useLayoutMutations
      >)

      const { bringNodeToFront } = useNodeZIndex()

      bringNodeToFront('node1')

      expect(mockSetSource).toHaveBeenCalledWith(LayoutSource.Vue)
      expect(mockBringNodeToFront).toHaveBeenCalledWith('node1')
    })

    it('should bring node to front with custom source', () => {
      const mockSetSource = vi.fn()
      const mockBringNodeToFront = vi.fn()

      mockedUseLayoutMutations.mockReturnValue({
        setSource: mockSetSource,
        bringNodeToFront: mockBringNodeToFront
      } as Partial<ReturnType<typeof useLayoutMutations>> as ReturnType<
        typeof useLayoutMutations
      >)

      const { bringNodeToFront } = useNodeZIndex()

      bringNodeToFront('node2', LayoutSource.Canvas)

      expect(mockSetSource).toHaveBeenCalledWith(LayoutSource.Canvas)
      expect(mockBringNodeToFront).toHaveBeenCalledWith('node2')
    })

    it('should use custom layout source from options', () => {
      const mockSetSource = vi.fn()
      const mockBringNodeToFront = vi.fn()

      mockedUseLayoutMutations.mockReturnValue({
        setSource: mockSetSource,
        bringNodeToFront: mockBringNodeToFront
      } as Partial<ReturnType<typeof useLayoutMutations>> as ReturnType<
        typeof useLayoutMutations
      >)

      const { bringNodeToFront } = useNodeZIndex({
        layoutSource: LayoutSource.External
      })

      bringNodeToFront('node3')

      expect(mockSetSource).toHaveBeenCalledWith(LayoutSource.External)
      expect(mockBringNodeToFront).toHaveBeenCalledWith('node3')
    })

    it('should override layout source with explicit source parameter', () => {
      const mockSetSource = vi.fn()
      const mockBringNodeToFront = vi.fn()

      mockedUseLayoutMutations.mockReturnValue({
        setSource: mockSetSource,
        bringNodeToFront: mockBringNodeToFront
      } as Partial<ReturnType<typeof useLayoutMutations>> as ReturnType<
        typeof useLayoutMutations
      >)

      const { bringNodeToFront } = useNodeZIndex({
        layoutSource: LayoutSource.External
      })

      bringNodeToFront('node4', LayoutSource.Canvas)

      expect(mockSetSource).toHaveBeenCalledWith(LayoutSource.Canvas)
      expect(mockBringNodeToFront).toHaveBeenCalledWith('node4')
    })
  })
})
