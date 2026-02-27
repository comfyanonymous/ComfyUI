/**
 * Litegraph Link Adapter
 *
 * Bridges the gap between litegraph's data model and the pure canvas renderer.
 * Converts litegraph-specific types (LLink, LGraphNode, slots) into generic
 * rendering data that can be consumed by the PathRenderer.
 * Maintains backward compatibility with existing litegraph integration.
 */
import type { LLink } from '@/lib/litegraph/src/LLink'
import type { Reroute } from '@/lib/litegraph/src/Reroute'
import type { CanvasColour, Point } from '@/lib/litegraph/src/interfaces'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import {
  LinkDirection,
  LinkMarkerShape,
  LinkRenderType
} from '@/lib/litegraph/src/types/globalEnums'
import { CanvasPathRenderer } from '@/renderer/core/canvas/pathRenderer'
import type {
  ArrowShape,
  Direction,
  LinkRenderData,
  RenderContext as PathRenderContext,
  Point as PointObj,
  RenderMode
} from '@/renderer/core/canvas/pathRenderer'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import type { Bounds } from '@/renderer/core/layout/types'

export interface LinkRenderContext {
  // Canvas settings
  renderMode: LinkRenderType
  connectionWidth: number
  renderBorder: boolean
  lowQuality: boolean
  highQualityRender: boolean
  scale: number
  linkMarkerShape: LinkMarkerShape
  renderConnectionArrows: boolean

  // State
  highlightedLinks: Set<string | number>

  // Colors
  defaultLinkColor: CanvasColour
  linkTypeColors: Record<string, CanvasColour>

  // Pattern for disabled links (optional)
  disabledPattern?: CanvasPattern | null
}

export class LitegraphLinkAdapter {
  private readonly pathRenderer = new CanvasPathRenderer()

  constructor(public readonly enableLayoutStoreWrites = true) {}

  /**
   * Convert LinkDirection enum to Direction string
   */
  private convertDirection(dir: LinkDirection): Direction {
    switch (dir) {
      case LinkDirection.LEFT:
        return 'left'
      case LinkDirection.RIGHT:
        return 'right'
      case LinkDirection.UP:
        return 'up'
      case LinkDirection.DOWN:
        return 'down'
      case LinkDirection.CENTER:
      case LinkDirection.NONE:
        return 'none'
      default:
        return 'right'
    }
  }

  /**
   * Convert LinkRenderContext to PathRenderContext
   */
  private convertToPathRenderContext(
    context: LinkRenderContext
  ): PathRenderContext {
    // Match original arrow rendering conditions:
    // Arrows only render when scale >= 0.6 AND highquality_render AND render_connection_arrows
    const shouldShowArrows =
      context.scale >= 0.6 &&
      context.highQualityRender &&
      context.renderConnectionArrows

    // Only show center marker when not set to None
    const shouldShowCenterMarker =
      context.linkMarkerShape !== LinkMarkerShape.None

    return {
      style: {
        mode: this.convertRenderMode(context.renderMode),
        connectionWidth: context.connectionWidth,
        borderWidth: context.renderBorder ? 4 : undefined,
        arrowShape: this.convertArrowShape(context.linkMarkerShape),
        showArrows: shouldShowArrows,
        lowQuality: context.lowQuality,
        // Center marker settings (matches original litegraph behavior)
        showCenterMarker: shouldShowCenterMarker,
        centerMarkerShape:
          context.linkMarkerShape === LinkMarkerShape.Arrow
            ? 'arrow'
            : 'circle',
        highQuality: context.highQualityRender
      },
      colors: {
        default: String(context.defaultLinkColor),
        byType: this.convertColorMap(context.linkTypeColors),
        highlighted: '#FFF'
      },
      patterns: {
        disabled: context.disabledPattern
      },
      animation: {
        time: LiteGraph.getTime() * 0.001
      },
      scale: context.scale,
      highlightedIds: new Set(Array.from(context.highlightedLinks).map(String))
    }
  }

  /**
   * Convert LinkRenderType to RenderMode
   */
  private convertRenderMode(mode: LinkRenderType): RenderMode {
    switch (mode) {
      case LinkRenderType.LINEAR_LINK:
        return 'linear'
      case LinkRenderType.STRAIGHT_LINK:
        return 'straight'
      case LinkRenderType.SPLINE_LINK:
      default:
        return 'spline'
    }
  }

  /**
   * Convert LinkMarkerShape to ArrowShape
   */
  private convertArrowShape(shape: LinkMarkerShape): ArrowShape {
    switch (shape) {
      case LinkMarkerShape.Circle:
        return 'circle'
      case LinkMarkerShape.Arrow:
      default:
        return 'triangle'
    }
  }

  /**
   * Convert color map to ensure all values are strings
   */
  private convertColorMap(
    colors: Record<string, CanvasColour>
  ): Record<string, string> {
    const result: Record<string, string> = {}
    for (const [key, value] of Object.entries(colors)) {
      result[key] = String(value)
    }
    return result
  }

  /**
   * Apply spline offset to a point, mimicking original #addSplineOffset behavior
   * Critically: does nothing for CENTER/NONE directions (no case for them)
   */
  private applySplineOffset(
    point: PointObj,
    direction: LinkDirection,
    distance: number
  ): void {
    switch (direction) {
      case LinkDirection.LEFT:
        point.x -= distance
        break
      case LinkDirection.RIGHT:
        point.x += distance
        break
      case LinkDirection.UP:
        point.y -= distance
        break
      case LinkDirection.DOWN:
        point.y += distance
        break
      // CENTER and NONE: no offset applied (original behavior)
    }
  }

  /**
   * Direct rendering method compatible with LGraphCanvas
   * Converts data and delegates to pure renderer
   */
  renderLinkDirect(
    ctx: CanvasRenderingContext2D,
    a: Readonly<Point>,
    b: Readonly<Point>,
    link: LLink | null,
    skip_border: boolean,
    flow: number | boolean | null,
    color: CanvasColour | null,
    start_dir: LinkDirection,
    end_dir: LinkDirection,
    context: LinkRenderContext,
    extras: {
      reroute?: Reroute
      startControl?: Readonly<Point>
      endControl?: Readonly<Point>
      num_sublines?: number
      disabled?: boolean
    } = {}
  ): void {
    // Apply same defaults as original renderLink
    const startDir = start_dir || LinkDirection.RIGHT
    const endDir = end_dir || LinkDirection.LEFT

    // Convert flow to boolean
    const flowBool = flow === true || (typeof flow === 'number' && flow > 0)

    // Create LinkRenderData from direct parameters
    const linkData: LinkRenderData = {
      id: link ? String(link.id) : 'temp',
      startPoint: { x: a[0], y: a[1] },
      endPoint: { x: b[0], y: b[1] },
      startDirection: this.convertDirection(startDir),
      endDirection: this.convertDirection(endDir),
      color: color !== null && color !== undefined ? String(color) : undefined,
      type: link?.type !== undefined ? String(link.type) : undefined,
      flow: flowBool,
      disabled: extras.disabled || false
    }

    // Control points handling (spline mode):
    // - Pre-refactor, the old renderLink honored a single provided control and
    //   derived the missing side via #addSplineOffset (CENTER => no offset).
    // - Restore that behavior here so reroute segments render identically.
    if (context.renderMode === LinkRenderType.SPLINE_LINK) {
      const hasStartCtrl = !!extras.startControl
      const hasEndCtrl = !!extras.endControl

      // Compute distance once for offsets
      const dist = Math.sqrt(
        (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
      )
      const factor = 0.25

      const cps: PointObj[] = []

      if (hasStartCtrl && hasEndCtrl) {
        // Both provided explicitly
        cps.push(
          {
            x: a[0] + (extras.startControl![0] || 0),
            y: a[1] + (extras.startControl![1] || 0)
          },
          {
            x: b[0] + (extras.endControl![0] || 0),
            y: b[1] + (extras.endControl![1] || 0)
          }
        )
        linkData.controlPoints = cps
      } else if (hasStartCtrl && !hasEndCtrl) {
        // Start provided, derive end via direction offset (CENTER => no offset)
        const start = {
          x: a[0] + (extras.startControl![0] || 0),
          y: a[1] + (extras.startControl![1] || 0)
        }
        const end = { x: b[0], y: b[1] }
        this.applySplineOffset(end, endDir, dist * factor)
        cps.push(start, end)
        linkData.controlPoints = cps
      } else if (!hasStartCtrl && hasEndCtrl) {
        // End provided, derive start via direction offset (CENTER => no offset)
        const start = { x: a[0], y: a[1] }
        this.applySplineOffset(start, startDir, dist * factor)
        const end = {
          x: b[0] + (extras.endControl![0] || 0),
          y: b[1] + (extras.endControl![1] || 0)
        }
        cps.push(start, end)
        linkData.controlPoints = cps
      } else {
        // Neither provided: derive both from directions (CENTER => no offset)
        const start = { x: a[0], y: a[1] }
        const end = { x: b[0], y: b[1] }
        this.applySplineOffset(start, startDir, dist * factor)
        this.applySplineOffset(end, endDir, dist * factor)
        cps.push(start, end)
        linkData.controlPoints = cps
      }
    }

    // Convert context
    const pathContext = this.convertToPathRenderContext(context)

    // Override skip_border if needed
    if (skip_border) {
      pathContext.style.borderWidth = undefined
    }

    // Render using pure renderer
    const path = this.pathRenderer.drawLink(ctx, linkData, pathContext)

    // Store path for hit detection
    const linkSegment = extras.reroute ?? link
    if (linkSegment) {
      linkSegment.path = path

      // Copy calculated center position back to litegraph object
      // This is needed for hit detection and menu interaction
      if (linkData.centerPos) {
        linkSegment._pos = linkSegment._pos || [0, 0]
        linkSegment._pos[0] = linkData.centerPos.x
        linkSegment._pos[1] = linkData.centerPos.y

        // Store center angle if calculated (for arrow markers)
        if (linkData.centerAngle !== undefined) {
          linkSegment._centreAngle = linkData.centerAngle
        }
      }

      // Update layout store when writes are enabled (event-driven path)
      if (this.enableLayoutStoreWrites && link && link.id !== -1) {
        // Calculate bounds and center only when writing
        const bounds = this.calculateLinkBounds(
          [linkData.startPoint.x, linkData.startPoint.y] as Readonly<Point>,
          [linkData.endPoint.x, linkData.endPoint.y] as Readonly<Point>,
          linkData
        )
        const centerPos = linkData.centerPos || {
          x: (linkData.startPoint.x + linkData.endPoint.x) / 2,
          y: (linkData.startPoint.y + linkData.endPoint.y) / 2
        }

        // Update whole link layout (only if not a reroute segment)
        if (!extras.reroute) {
          layoutStore.updateLinkLayout(link.id, {
            id: link.id,
            path: path,
            bounds: bounds,
            centerPos: centerPos,
            sourceNodeId: String(link.origin_id),
            targetNodeId: String(link.target_id),
            sourceSlot: link.origin_slot,
            targetSlot: link.target_slot
          })
        }

        // Always update segment layout (for both regular links and reroute segments)
        const rerouteId = extras.reroute ? extras.reroute.id : null
        layoutStore.updateLinkSegmentLayout(link.id, rerouteId, {
          path: path,
          bounds: bounds,
          centerPos: centerPos
        })
      }
    }
  }

  renderDraggingLink(
    ctx: CanvasRenderingContext2D,
    from: Readonly<Point>,
    to: Readonly<Point>,
    colour: CanvasColour,
    startDir: LinkDirection,
    endDir: LinkDirection,
    context: LinkRenderContext
  ): void {
    this.renderLinkDirect(
      ctx,
      from,
      to,
      null,
      false,
      null,
      colour,
      startDir,
      endDir,
      {
        ...context,
        linkMarkerShape: LinkMarkerShape.None
      },
      {
        disabled: false
      }
    )
  }

  /**
   * Calculate bounding box for a link
   * Includes padding for line width and control points
   */
  private calculateLinkBounds(
    startPos: Readonly<Point>,
    endPos: Readonly<Point>,
    linkData: LinkRenderData
  ): Bounds {
    let minX = Math.min(startPos[0], endPos[0])
    let maxX = Math.max(startPos[0], endPos[0])
    let minY = Math.min(startPos[1], endPos[1])
    let maxY = Math.max(startPos[1], endPos[1])

    // Include control points if they exist (for spline links)
    if (linkData.controlPoints) {
      for (const cp of linkData.controlPoints) {
        minX = Math.min(minX, cp.x)
        maxX = Math.max(maxX, cp.x)
        minY = Math.min(minY, cp.y)
        maxY = Math.max(maxY, cp.y)
      }
    }

    // Add padding for line width and hit tolerance
    const padding = 20

    return {
      x: minX - padding,
      y: minY - padding,
      width: maxX - minX + 2 * padding,
      height: maxY - minY + 2 * padding
    }
  }
}
