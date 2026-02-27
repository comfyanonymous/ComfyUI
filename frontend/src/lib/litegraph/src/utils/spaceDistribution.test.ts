import { describe, expect, it } from 'vitest'

import { distributeSpace } from '@/lib/litegraph/src/litegraph'
import type { SpaceRequest } from '@/lib/litegraph/src/litegraph'

describe('distributeSpace', () => {
  it('should distribute space according to minimum sizes when space is limited', () => {
    const requests: SpaceRequest[] = [
      { minSize: 100 },
      { minSize: 100 },
      { minSize: 100 }
    ]
    expect(distributeSpace(300, requests)).toEqual([100, 100, 100])
  })

  it('should distribute extra space equally when no maxSize', () => {
    const requests: SpaceRequest[] = [{ minSize: 100 }, { minSize: 100 }]
    expect(distributeSpace(400, requests)).toEqual([200, 200])
  })

  it('should respect maximum sizes', () => {
    const requests: SpaceRequest[] = [
      { minSize: 100, maxSize: 150 },
      { minSize: 100 }
    ]
    expect(distributeSpace(400, requests)).toEqual([150, 250])
  })

  it('should handle empty requests array', () => {
    expect(distributeSpace(1000, [])).toEqual([])
  })

  it('should handle negative total space', () => {
    const requests: SpaceRequest[] = [{ minSize: 100 }, { minSize: 100 }]
    expect(distributeSpace(-100, requests)).toEqual([100, 100])
  })

  it('should handle total space smaller than minimum sizes', () => {
    const requests: SpaceRequest[] = [{ minSize: 100 }, { minSize: 100 }]
    expect(distributeSpace(100, requests)).toEqual([100, 100])
  })
})
