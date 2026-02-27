import { describe, it, expect } from 'vitest'
import { getEffectiveBrushSize, getEffectiveHardness } from './brushUtils'

describe('brushUtils', () => {
  describe('getEffectiveBrushSize', () => {
    it('should return original size when hardness is 1.0', () => {
      const size = 100
      const hardness = 1.0
      expect(getEffectiveBrushSize(size, hardness)).toBe(100)
    })

    it('should return 1.5x size when hardness is 0.0', () => {
      const size = 100
      const hardness = 0.0
      expect(getEffectiveBrushSize(size, hardness)).toBe(150)
    })

    it('should interpolate linearly', () => {
      const size = 100
      const hardness = 0.5
      // Scale should be 1.0 + 0.5 * 0.5 = 1.25
      expect(getEffectiveBrushSize(size, hardness)).toBe(125)
    })
  })

  describe('getEffectiveHardness', () => {
    it('should return same hardness if effective size matches size', () => {
      const size = 100
      const hardness = 0.8
      const effectiveSize = 100
      expect(getEffectiveHardness(size, hardness, effectiveSize)).toBe(0.8)
    })

    it('should scale hardness down as effective size increases', () => {
      const size = 100
      const hardness = 0.5
      // Effective size at 0.5 hardness is 125
      const effectiveSize = 125
      // Hard core radius = 50. New hardness = 50 / 125 = 0.4
      expect(getEffectiveHardness(size, hardness, effectiveSize)).toBe(0.4)
    })

    it('should return 0 if effective size is 0', () => {
      expect(getEffectiveHardness(100, 0.5, 0)).toBe(0)
    })
  })
})
