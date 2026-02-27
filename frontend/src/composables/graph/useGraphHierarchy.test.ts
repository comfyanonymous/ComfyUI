import { beforeEach, describe, expect, it, vi } from 'vitest'

import { Rectangle } from '@/lib/litegraph/src/infrastructure/Rectangle'
import type { LGraphGroup, LGraphNode } from '@/lib/litegraph/src/litegraph'
import * as measure from '@/lib/litegraph/src/measure'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import {
  createMockLGraphNode,
  createMockLGraphGroup
} from '@/utils/__tests__/litegraphTestUtils'

import { useGraphHierarchy } from './useGraphHierarchy'

vi.mock('@/renderer/core/canvas/canvasStore')

function createMockNode(overrides: Partial<LGraphNode> = {}): LGraphNode {
  return Object.assign(
    createMockLGraphNode(),
    {
      boundingRect: new Rectangle(100, 100, 50, 50)
    },
    overrides
  )
}

function createMockGroup(overrides: Partial<LGraphGroup> = {}): LGraphGroup {
  return createMockLGraphGroup(overrides)
}

describe('useGraphHierarchy', () => {
  let mockCanvasStore: Partial<ReturnType<typeof useCanvasStore>>
  let mockNode: LGraphNode
  let mockGroups: LGraphGroup[]

  beforeEach(() => {
    mockNode = createMockNode()
    mockGroups = []

    mockCanvasStore = {
      canvas: {
        graph: {
          groups: mockGroups
        }
      },
      $id: 'canvas',
      $state: {},
      $patch: vi.fn(),
      $reset: vi.fn(),
      $subscribe: vi.fn(),
      $onAction: vi.fn(),
      $dispose: vi.fn(),
      _customProperties: new Set(),
      _p: {}
    } as unknown as Partial<ReturnType<typeof useCanvasStore>>

    vi.mocked(useCanvasStore).mockReturnValue(
      mockCanvasStore as ReturnType<typeof useCanvasStore>
    )
  })

  describe('findParentGroup', () => {
    it('returns null when no groups exist', () => {
      const { findParentGroup } = useGraphHierarchy()

      const result = findParentGroup(mockNode)

      expect(result).toBeNull()
    })

    it('returns null when node is not in any group', () => {
      const group = createMockGroup({
        boundingRect: new Rectangle(0, 0, 50, 50)
      })
      mockGroups.push(group)

      vi.spyOn(measure, 'containsCentre').mockReturnValue(false)

      const { findParentGroup } = useGraphHierarchy()
      const result = findParentGroup(mockNode)

      expect(result).toBeNull()
    })

    it('returns the only group when node is in exactly one group', () => {
      const group = createMockGroup({
        boundingRect: new Rectangle(0, 0, 200, 200)
      })
      mockGroups.push(group)

      vi.spyOn(measure, 'containsCentre').mockReturnValue(true)

      const { findParentGroup } = useGraphHierarchy()
      const result = findParentGroup(mockNode)

      expect(result).toBe(group)
    })

    it('returns the smallest group when node is in multiple groups', () => {
      const largeGroup = createMockGroup({
        boundingRect: new Rectangle(0, 0, 300, 300)
      })
      const smallGroup = createMockGroup({
        boundingRect: new Rectangle(50, 50, 100, 100)
      })
      mockGroups.push(largeGroup, smallGroup)

      vi.spyOn(measure, 'containsCentre').mockReturnValue(true)
      vi.spyOn(measure, 'containsRect').mockReturnValue(false)

      const { findParentGroup } = useGraphHierarchy()
      const result = findParentGroup(mockNode)

      expect(result).toBe(smallGroup)
    })

    it('returns the inner group when one group contains another', () => {
      const outerGroup = createMockGroup({
        boundingRect: new Rectangle(0, 0, 300, 300)
      })
      const innerGroup = createMockGroup({
        boundingRect: new Rectangle(50, 50, 100, 100)
      })
      mockGroups.push(outerGroup, innerGroup)

      vi.spyOn(measure, 'containsCentre').mockReturnValue(true)
      vi.spyOn(measure, 'containsRect').mockImplementation(
        (container, contained) => {
          // outerGroup contains innerGroup
          if (container === outerGroup.boundingRect) {
            return contained === innerGroup.boundingRect
          }
          return false
        }
      )

      const { findParentGroup } = useGraphHierarchy()
      const result = findParentGroup(mockNode)

      expect(result).toBe(innerGroup)
    })

    it('handles null canvas gracefully', () => {
      mockCanvasStore.canvas = null

      const { findParentGroup } = useGraphHierarchy()
      const result = findParentGroup(mockNode)

      expect(result).toBeNull()
    })

    it('handles null graph gracefully', () => {
      mockCanvasStore.canvas!.graph = null

      const { findParentGroup } = useGraphHierarchy()
      const result = findParentGroup(mockNode)

      expect(result).toBeNull()
    })
  })
})
