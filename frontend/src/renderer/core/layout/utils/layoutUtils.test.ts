import { describe, expect, it } from 'vitest'

import { makeLinkSegmentKey } from '@/renderer/core/layout/utils/layoutUtils'

describe('layoutUtils', () => {
  describe('makeLinkSegmentKey', () => {
    it('creates stable keys for null reroute', () => {
      expect(makeLinkSegmentKey(10, null)).toBe('10:final')
      expect(makeLinkSegmentKey(42, null)).toBe('42:final')
    })

    it('creates stable keys for numeric reroute ids', () => {
      expect(makeLinkSegmentKey(10, 3)).toBe('10:3')
      expect(makeLinkSegmentKey(42, 0)).toBe('42:0')
      expect(makeLinkSegmentKey(42, 7)).toBe('42:7')
    })
  })
})
