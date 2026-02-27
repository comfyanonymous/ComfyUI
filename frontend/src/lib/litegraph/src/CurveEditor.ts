import { clamp } from 'es-toolkit/compat'

import type { Point, Rect } from './interfaces'
import type { LGraphCanvas } from './litegraph'
import { distance } from './measure'

// used by some widgets to render a curve editor

export class CurveEditor {
  points: Point[]
  selected: number
  nearest: number
  size: Rect | null
  must_update: boolean
  margin: number
  _nearest?: number

  constructor(points: Point[]) {
    this.points = points
    this.selected = -1
    this.nearest = -1
    // stores last size used
    this.size = null
    this.must_update = true
    this.margin = 5
  }

  static sampleCurve(f: number, points: Point[]): number | undefined {
    if (!points) return

    for (let i = 0; i < points.length - 1; ++i) {
      const p = points[i]
      const pn = points[i + 1]
      if (pn[0] < f) continue

      const r = pn[0] - p[0]
      if (Math.abs(r) < 0.000_01) return p[1]

      const local_f = (f - p[0]) / r
      return p[1] * (1.0 - local_f) + pn[1] * local_f
    }
    return 0
  }

  draw(
    ctx: CanvasRenderingContext2D,
    size: Rect,
    // @ts-expect-error - LGraphCanvas parameter type needs fixing
    graphcanvas?: LGraphCanvas,
    background_color?: string,
    line_color?: string,
    inactive = false
  ): void {
    const points = this.points
    if (!points) return

    this.size = size
    const w = size[0] - this.margin * 2
    const h = size[1] - this.margin * 2

    line_color = line_color || '#666'

    ctx.save()
    ctx.translate(this.margin, this.margin)

    if (background_color) {
      ctx.fillStyle = '#111'
      ctx.fillRect(0, 0, w, h)
      ctx.fillStyle = '#222'
      ctx.fillRect(w * 0.5, 0, 1, h)
      ctx.strokeStyle = '#333'
      ctx.strokeRect(0, 0, w, h)
    }
    ctx.strokeStyle = line_color
    if (inactive) ctx.globalAlpha = 0.5
    ctx.beginPath()
    for (const p of points) {
      ctx.lineTo(p[0] * w, (1.0 - p[1]) * h)
    }
    ctx.stroke()
    ctx.globalAlpha = 1
    if (!inactive) {
      for (const [i, p] of points.entries()) {
        ctx.fillStyle =
          this.selected == i ? '#FFF' : this.nearest == i ? '#DDD' : '#AAA'
        ctx.beginPath()
        ctx.arc(p[0] * w, (1.0 - p[1]) * h, 2, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    ctx.restore()
  }

  // localpos is mouse in curve editor space
  onMouseDown(localpos: Point, graphcanvas: LGraphCanvas): boolean | undefined {
    const points = this.points
    if (!points) return
    if (localpos[1] < 0) return

    // this.captureInput(true);
    if (this.size == null)
      throw new Error('CurveEditor.size was null or undefined.')
    const w = this.size[0] - this.margin * 2
    const h = this.size[1] - this.margin * 2
    const x = localpos[0] - this.margin
    const y = localpos[1] - this.margin
    const pos: Point = [x, y]
    const max_dist = 30 / graphcanvas.ds.scale
    // search closer one
    this.selected = this.getCloserPoint(pos, max_dist)
    // create one
    if (this.selected == -1) {
      const point: Point = [x / w, 1 - y / h]
      points.push(point)
      points.sort(function (a, b) {
        return a[0] - b[0]
      })
      this.selected = points.indexOf(point)
      this.must_update = true
    }
    if (this.selected != -1) return true
  }

  onMouseMove(localpos: Point, graphcanvas: LGraphCanvas): void {
    const points = this.points
    if (!points) return

    const s = this.selected
    if (s < 0) return

    if (this.size == null)
      throw new Error('CurveEditor.size was null or undefined.')
    const x = (localpos[0] - this.margin) / (this.size[0] - this.margin * 2)
    const y = (localpos[1] - this.margin) / (this.size[1] - this.margin * 2)
    const curvepos: Point = [
      localpos[0] - this.margin,
      localpos[1] - this.margin
    ]
    const max_dist = 30 / graphcanvas.ds.scale
    this._nearest = this.getCloserPoint(curvepos, max_dist)
    const point = points[s]
    if (point) {
      const is_edge_point = s == 0 || s == points.length - 1
      if (
        !is_edge_point &&
        (localpos[0] < -10 ||
          localpos[0] > this.size[0] + 10 ||
          localpos[1] < -10 ||
          localpos[1] > this.size[1] + 10)
      ) {
        points.splice(s, 1)
        this.selected = -1
        return
      }
      // not edges
      if (!is_edge_point) point[0] = clamp(x, 0, 1)
      else point[0] = s == 0 ? 0 : 1
      point[1] = 1.0 - clamp(y, 0, 1)
      points.sort(function (a, b) {
        return a[0] - b[0]
      })
      this.selected = points.indexOf(point)
      this.must_update = true
    }
  }

  // Former params: localpos, graphcanvas
  onMouseUp(): boolean {
    this.selected = -1
    return false
  }

  getCloserPoint(pos: Point, max_dist: number): number {
    const points = this.points
    if (!points) return -1

    max_dist = max_dist || 30
    if (this.size == null)
      throw new Error('CurveEditor.size was null or undefined.')
    const w = this.size[0] - this.margin * 2
    const h = this.size[1] - this.margin * 2
    const num = points.length
    const p2: Point = [0, 0]
    let min_dist = 1_000_000
    let closest = -1

    for (let i = 0; i < num; ++i) {
      const p = points[i]
      p2[0] = p[0] * w
      p2[1] = (1.0 - p[1]) * h
      const dist = distance(pos, p2)
      if (dist > min_dist || dist > max_dist) continue

      closest = i
      min_dist = dist
    }
    return closest
  }
}
