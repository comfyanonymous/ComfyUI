import { describe, expect, it } from 'vitest'

import type { ReadOnlyRect } from '@/lib/litegraph/src/interfaces'
import { computeUnionBounds, gcd, lcm } from '@/utils/mathUtil'

describe('mathUtil', () => {
  describe('gcd', () => {
    it('should compute greatest common divisor correctly', () => {
      expect(gcd(48, 18)).toBe(6)
      expect(gcd(100, 25)).toBe(25)
      expect(gcd(17, 13)).toBe(1)
      expect(gcd(0, 5)).toBe(5)
      expect(gcd(5, 0)).toBe(5)
    })

    it('should handle negative numbers', () => {
      expect(gcd(-48, 18)).toBe(6)
      expect(gcd(48, -18)).toBe(6)
      expect(gcd(-48, -18)).toBe(6)
    })

    it('should not cause stack overflow with small floating-point step values', () => {
      // This would cause Maximum call stack size exceeded with recursive impl
      // when used in lcm calculations with small step values
      expect(() => gcd(0.0001, 0.0003)).not.toThrow()
      expect(() => gcd(1e-10, 1e-9)).not.toThrow()
    })
  })

  describe('lcm', () => {
    it('should compute least common multiple correctly', () => {
      expect(lcm(4, 6)).toBe(12)
      expect(lcm(15, 20)).toBe(60)
      expect(lcm(7, 11)).toBe(77)
    })
  })

  describe('computeUnionBounds', () => {
    it('should return null for empty input', () => {
      expect(computeUnionBounds([])).toBe(null)
    })

    // Tests for tuple format (ReadOnlyRect)
    it('should work with ReadOnlyRect tuple format', () => {
      const tuples: ReadOnlyRect[] = [
        [10, 20, 30, 40] as const, // bounds: 10,20 to 40,60
        [50, 10, 20, 30] as const // bounds: 50,10 to 70,40
      ]

      const result = computeUnionBounds(tuples)

      expect(result).toEqual({
        x: 10, // min(10, 50)
        y: 10, // min(20, 10)
        width: 60, // max(40, 70) - min(10, 50) = 70 - 10
        height: 50 // max(60, 40) - min(20, 10) = 60 - 10
      })
    })

    it('should handle single ReadOnlyRect tuple', () => {
      const tuple: ReadOnlyRect = [10, 20, 30, 40] as const
      const result = computeUnionBounds([tuple])

      expect(result).toEqual({
        x: 10,
        y: 20,
        width: 30,
        height: 40
      })
    })

    it('should handle tuple format with negative dimensions', () => {
      const tuples: ReadOnlyRect[] = [
        [100, 50, -20, -10] as const, // x+width=80, y+height=40
        [90, 45, 15, 20] as const // x+width=105, y+height=65
      ]

      const result = computeUnionBounds(tuples)

      expect(result).toEqual({
        x: 90, // min(100, 90)
        y: 45, // min(50, 45)
        width: 15, // max(80, 105) - min(100, 90) = 105 - 90
        height: 20 // max(40, 65) - min(50, 45) = 65 - 45
      })
    })

    it('should maintain optimal performance with SoA tuples', () => {
      // Test that array access is as expected for typical selection sizes
      const tuples: ReadOnlyRect[] = Array.from(
        { length: 10 },
        (_, i) =>
          [
            i * 20, // x
            i * 15, // y
            100 + i * 5, // width
            80 + i * 3 // height
          ] as const
      )

      const result = computeUnionBounds(tuples)

      expect(result).toBeTruthy()
      expect(result!.x).toBe(0)
      expect(result!.y).toBe(0)
      expect(result!.width).toBe(325)
      expect(result!.height).toBe(242)
    })
  })
})
