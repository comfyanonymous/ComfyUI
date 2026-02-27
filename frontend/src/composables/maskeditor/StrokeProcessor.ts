import type { Point } from '@/extensions/core/maskeditor/types'
import { catmullRomSpline, resampleSegment } from './splineUtils'

export class StrokeProcessor {
  private controlPoints: Point[] = []
  private remainder: number = 0
  private spacing: number
  private isFirstPoint: boolean = true
  private hasProcessedSegment: boolean = false

  constructor(spacing: number) {
    this.spacing = spacing
  }

  /**
   * Adds a point to the stroke and returns any new equidistant points generated.
   * Maintain a sliding window of 4 control points for spline generation
   */
  public addPoint(point: Point): Point[] {
    // Initialize buffer with the first point
    if (this.isFirstPoint) {
      this.controlPoints.push(point) // p0: phantom start point
      this.controlPoints.push(point) // p1: actual start point
      this.isFirstPoint = false
      return [] // Wait for more points to form a segment
    }

    this.controlPoints.push(point)

    // Require 4 points for a spline segment
    if (this.controlPoints.length < 4) {
      return []
    }

    // Generate segment p1->p2
    const p0 = this.controlPoints[0]
    const p1 = this.controlPoints[1]
    const p2 = this.controlPoints[2]
    const p3 = this.controlPoints[3]

    const newPoints = this.processSegment(p0, p1, p2, p3)

    // Slide window
    this.controlPoints.shift()

    return newPoints
  }

  /**
   * End stroke and flush remaining segments
   */
  public endStroke(): Point[] {
    if (this.controlPoints.length < 2) {
      // Insufficient points for a segment
      return []
    }

    // Process remaining segments by duplicating the last point

    const newPoints: Point[] = []

    // Flush the buffer by processing the final segment

    while (this.controlPoints.length >= 3) {
      const p0 = this.controlPoints[0]
      const p1 = this.controlPoints[1]
      const p2 = this.controlPoints[2]
      const p3 = p2 // Duplicate last point as phantom end

      const points = this.processSegment(p0, p1, p2, p3)
      newPoints.push(...points)

      this.controlPoints.shift()
    }

    // Handle single point click
    if (!this.hasProcessedSegment && this.controlPoints.length >= 2) {
      // Process zero-length segment for single point
      const p = this.controlPoints[1]
      const points = this.processSegment(p, p, p, p)
      newPoints.push(...points)
    }

    return newPoints
  }

  private processSegment(p0: Point, p1: Point, p2: Point, p3: Point): Point[] {
    this.hasProcessedSegment = true
    // Generate dense points for the segment
    const densePoints: Point[] = []

    // Adaptive sampling based on segment length
    const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y)
    // Use 1 sample per pixel, but at least 5 samples to ensure smoothness for short segments
    // and cap at a reasonable maximum if needed (though not strictly necessary with density)
    const samples = Math.max(5, Math.ceil(dist))

    for (let i = 0; i < samples; i++) {
      const t = i / samples
      densePoints.push(catmullRomSpline(p0, p1, p2, p3, t))
    }
    // Add segment end point
    densePoints.push(p2)

    // Resample points with carried-over remainder
    const { points, remainder } = resampleSegment(
      densePoints,
      this.spacing,
      this.remainder
    )

    this.remainder = remainder
    return points
  }
}
