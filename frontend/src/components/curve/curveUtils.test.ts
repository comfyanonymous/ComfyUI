import { describe, expect, it } from 'vitest'

import type { CurvePoint } from './types'

import {
  createMonotoneInterpolator,
  curvesToLUT,
  histogramToPath
} from './curveUtils'

describe('createMonotoneInterpolator', () => {
  it('returns 0 for empty points', () => {
    const interpolate = createMonotoneInterpolator([])
    expect(interpolate(0.5)).toBe(0)
  })

  it('returns constant for single point', () => {
    const interpolate = createMonotoneInterpolator([[0.5, 0.7]])
    expect(interpolate(0)).toBe(0.7)
    expect(interpolate(1)).toBe(0.7)
  })

  it('passes through control points exactly', () => {
    const points: CurvePoint[] = [
      [0, 0],
      [0.5, 0.8],
      [1, 1]
    ]
    const interpolate = createMonotoneInterpolator(points)
    expect(interpolate(0)).toBeCloseTo(0, 5)
    expect(interpolate(0.5)).toBeCloseTo(0.8, 5)
    expect(interpolate(1)).toBeCloseTo(1, 5)
  })

  it('clamps to endpoint values outside range', () => {
    const points: CurvePoint[] = [
      [0.2, 0.3],
      [0.8, 0.9]
    ]
    const interpolate = createMonotoneInterpolator(points)
    expect(interpolate(0)).toBe(0.3)
    expect(interpolate(1)).toBe(0.9)
  })

  it('produces monotone output for monotone input', () => {
    const points: CurvePoint[] = [
      [0, 0],
      [0.25, 0.2],
      [0.5, 0.5],
      [0.75, 0.8],
      [1, 1]
    ]
    const interpolate = createMonotoneInterpolator(points)

    let prev = -Infinity
    for (let x = 0; x <= 1; x += 0.01) {
      const y = interpolate(x)
      expect(y).toBeGreaterThanOrEqual(prev)
      prev = y
    }
  })

  it('handles unsorted input points', () => {
    const points: CurvePoint[] = [
      [1, 1],
      [0, 0],
      [0.5, 0.5]
    ]
    const interpolate = createMonotoneInterpolator(points)
    expect(interpolate(0)).toBeCloseTo(0, 5)
    expect(interpolate(0.5)).toBeCloseTo(0.5, 5)
    expect(interpolate(1)).toBeCloseTo(1, 5)
  })
})

describe('curvesToLUT', () => {
  it('returns a 256-entry Uint8Array', () => {
    const lut = curvesToLUT([
      [0, 0],
      [1, 1]
    ])
    expect(lut).toBeInstanceOf(Uint8Array)
    expect(lut.length).toBe(256)
  })

  it('produces identity LUT for diagonal curve', () => {
    const lut = curvesToLUT([
      [0, 0],
      [1, 1]
    ])
    for (let i = 0; i < 256; i++) {
      expect(lut[i]).toBeCloseTo(i, 0)
    }
  })

  it('clamps output to [0, 255]', () => {
    const lut = curvesToLUT([
      [0, 0],
      [0.5, 1.5],
      [1, 1]
    ])
    for (let i = 0; i < 256; i++) {
      expect(lut[i]).toBeGreaterThanOrEqual(0)
      expect(lut[i]).toBeLessThanOrEqual(255)
    }
  })
})

describe('histogramToPath', () => {
  it('returns empty string for empty histogram', () => {
    expect(histogramToPath(new Uint32Array(0))).toBe('')
  })

  it('returns empty string when all bins are zero', () => {
    expect(histogramToPath(new Uint32Array(256))).toBe('')
  })

  it('returns a closed SVG path for valid histogram', () => {
    const histogram = new Uint32Array(256)
    for (let i = 0; i < 256; i++) histogram[i] = i + 1
    const path = histogramToPath(histogram)
    expect(path).toMatch(/^M0,1/)
    expect(path).toMatch(/L1,1 Z$/)
  })

  it('normalizes using 99.5th percentile to suppress outliers', () => {
    const histogram = new Uint32Array(256)
    for (let i = 0; i < 256; i++) histogram[i] = 100
    histogram[255] = 100000
    const path = histogramToPath(histogram)
    // Most bins should map to y=0 (1 - 100/100 = 0) since
    // the 99.5th percentile is 100, not the outlier 100000
    const yValues = path
      .split(/[ML]/)
      .filter(Boolean)
      .map((s) => parseFloat(s.split(',')[1]))
      .filter((y) => !isNaN(y))
    const nearZero = yValues.filter((y) => Math.abs(y) < 0.01)
    expect(nearZero.length).toBeGreaterThan(200)
  })
})
