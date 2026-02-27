import type { CurvePoint } from './types'

/**
 * Monotone cubic Hermite interpolation.
 * Produces a smooth curve that passes through all control points
 * without overshooting (monotone property).
 *
 * Returns a function that evaluates y for any x in [0, 1].
 */
export function createMonotoneInterpolator(
  points: CurvePoint[]
): (x: number) => number {
  if (points.length === 0) return () => 0
  if (points.length === 1) return () => points[0][1]

  const sorted = [...points].sort((a, b) => a[0] - b[0])
  const n = sorted.length
  const xs = sorted.map((p) => p[0])
  const ys = sorted.map((p) => p[1])

  const deltas: number[] = []
  const slopes: number[] = []
  for (let i = 0; i < n - 1; i++) {
    const dx = xs[i + 1] - xs[i]
    deltas.push(dx === 0 ? 0 : (ys[i + 1] - ys[i]) / dx)
  }

  slopes.push(deltas[0] ?? 0)
  for (let i = 1; i < n - 1; i++) {
    if (deltas[i - 1] * deltas[i] <= 0) {
      slopes.push(0)
    } else {
      slopes.push((deltas[i - 1] + deltas[i]) / 2)
    }
  }
  slopes.push(deltas[n - 2] ?? 0)

  for (let i = 0; i < n - 1; i++) {
    if (deltas[i] === 0) {
      slopes[i] = 0
      slopes[i + 1] = 0
    } else {
      const alpha = slopes[i] / deltas[i]
      const beta = slopes[i + 1] / deltas[i]
      const s = alpha * alpha + beta * beta
      if (s > 9) {
        const t = 3 / Math.sqrt(s)
        slopes[i] = t * alpha * deltas[i]
        slopes[i + 1] = t * beta * deltas[i]
      }
    }
  }

  return (x: number): number => {
    if (x <= xs[0]) return ys[0]
    if (x >= xs[n - 1]) return ys[n - 1]

    let lo = 0
    let hi = n - 1
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1
      if (xs[mid] <= x) lo = mid
      else hi = mid
    }

    const dx = xs[hi] - xs[lo]
    if (dx === 0) return ys[lo]

    const t = (x - xs[lo]) / dx
    const t2 = t * t
    const t3 = t2 * t

    const h00 = 2 * t3 - 3 * t2 + 1
    const h10 = t3 - 2 * t2 + t
    const h01 = -2 * t3 + 3 * t2
    const h11 = t3 - t2

    return (
      h00 * ys[lo] +
      h10 * dx * slopes[lo] +
      h01 * ys[hi] +
      h11 * dx * slopes[hi]
    )
  }
}

/**
 * Convert a 256-bin histogram into an SVG path string.
 * Normalizes using the 99.5th percentile to avoid outlier spikes.
 */
export function histogramToPath(histogram: Uint32Array): string {
  if (!histogram.length) return ''

  const sorted = Array.from(histogram).sort((a, b) => a - b)
  const max = sorted[Math.floor(255 * 0.995)]
  if (max === 0) return ''

  const invMax = 1 / max
  const parts: string[] = ['M0,1']
  for (let i = 0; i < 256; i++) {
    const x = i / 255
    const y = 1 - Math.min(1, histogram[i] * invMax)
    parts.push(`L${x},${y}`)
  }
  parts.push('L1,1 Z')
  return parts.join(' ')
}

export function curvesToLUT(points: CurvePoint[]): Uint8Array {
  const lut = new Uint8Array(256)
  const interpolate = createMonotoneInterpolator(points)

  for (let i = 0; i < 256; i++) {
    const x = i / 255
    const y = interpolate(x)
    lut[i] = Math.max(0, Math.min(255, Math.round(y * 255)))
  }

  return lut
}
