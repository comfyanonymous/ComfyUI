import type { Rectangle } from './infrastructure/Rectangle'
import type { CanvasColour } from './interfaces'
import { LiteGraph } from './litegraph'
import { RenderShape, TitleMode } from './types/globalEnums'

const ELLIPSIS = '\u2026'
const TWO_DOT_LEADER = '\u2025'
const ONE_DOT_LEADER = '\u2024'

export enum SlotType {
  Array = 'array',
  Event = -1
}

/** @see RenderShape */
export enum SlotShape {
  Box = RenderShape.BOX,
  Arrow = RenderShape.ARROW,
  Grid = RenderShape.GRID,
  Circle = RenderShape.CIRCLE,
  HollowCircle = RenderShape.HollowCircle
}

/** @see LinkDirection */
export enum SlotDirection {}

export enum LabelPosition {
  Left = 'left',
  Right = 'right'
}

export interface IDrawBoundingOptions {
  /** The shape to render */
  shape?: RenderShape
  /** The radius of the rounded corners for {@link RenderShape.ROUND} and {@link RenderShape.CARD} */
  round_radius?: number
  /** Shape will extend above the Y-axis 0 by this amount @deprecated This is node-specific: it should be removed entirely, and behaviour defined by the caller more explicitly */
  title_height?: number
  /** @deprecated This is node-specific: it should be removed entirely, and behaviour defined by the caller more explicitly */
  title_mode?: TitleMode
  /** The color that should be drawn */
  color?: CanvasColour
  /** The distance between the edge of the {@link area} and the middle of the line */
  padding?: number
  /** @deprecated This is node-specific: it should be removed entirely, and behaviour defined by the caller more explicitly */
  collapsed?: boolean
  /** Thickness of the line drawn (`lineWidth`) */
  lineWidth?: number
}

interface IDrawTextInAreaOptions {
  /** The canvas to draw the text on. */
  ctx: CanvasRenderingContext2D
  /** The text to draw. */
  text: string
  /** The area the text will be drawn in. */
  area: Rectangle
  /** The alignment of the text. */
  align?: 'left' | 'right' | 'center'
}

/**
 * Draws only the path of a shape on the canvas, without filling.
 * Used to draw indicators for node status, e.g. "selected".
 * @param ctx The 2D context to draw on
 * @param area The position and size of the shape to render
 */
export function strokeShape(
  ctx: CanvasRenderingContext2D,
  area: Rectangle,
  {
    shape = RenderShape.BOX,
    round_radius,
    title_height,
    title_mode = TitleMode.NORMAL_TITLE,
    color,
    padding = 6,
    collapsed = false,
    lineWidth: thickness = 1
  }: IDrawBoundingOptions = {}
): void {
  // These param defaults are not compile-time static, and must be re-evaluated at runtime
  round_radius ??= LiteGraph.ROUND_RADIUS
  color ??= LiteGraph.NODE_BOX_OUTLINE_COLOR

  // Adjust area if title is transparent
  if (title_mode === TitleMode.TRANSPARENT_TITLE) {
    const height = title_height ?? LiteGraph.NODE_TITLE_HEIGHT
    area[1] -= height
    area[3] += height
  }

  // Set up context
  const { lineWidth, strokeStyle } = ctx
  ctx.lineWidth = thickness
  ctx.globalAlpha = 0.8
  ctx.strokeStyle = color
  ctx.beginPath()

  // Draw shape based on type
  const [x, y, width, height] = area
  switch (shape) {
    case RenderShape.BOX: {
      ctx.rect(
        x - padding,
        y - padding,
        width + 2 * padding,
        height + 2 * padding
      )
      break
    }
    case RenderShape.ROUND:
    case RenderShape.CARD: {
      const radius = round_radius + padding
      const isCollapsed = shape === RenderShape.CARD && collapsed
      const cornerRadii =
        isCollapsed || shape === RenderShape.ROUND
          ? [radius]
          : [radius, 2, radius, 2]
      ctx.roundRect(
        x - padding,
        y - padding,
        width + 2 * padding,
        height + 2 * padding,
        cornerRadii
      )
      break
    }
    case RenderShape.CIRCLE: {
      const centerX = x + width / 2
      const centerY = y + height / 2
      const radius = Math.max(width, height) / 2 + padding
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
      break
    }
  }

  // Stroke the shape
  ctx.stroke()

  // Reset context
  ctx.lineWidth = lineWidth
  ctx.strokeStyle = strokeStyle

  // TODO: Store and reset value properly.  Callers currently expect this behaviour (e.g. muted nodes).
  ctx.globalAlpha = 1
}

/**
 * Truncates text using binary search to fit within a given width, appending an ellipsis if needed.
 * @param ctx The canvas rendering context.
 * @param text The text to truncate.
 * @param maxWidth The maximum width the text (plus ellipsis) can occupy.
 * @returns The truncated text, or the original text if it fits.
 */
function truncateTextToWidth(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number
): string {
  if (!(maxWidth > 0)) return ''

  // Text fits
  const fullWidth = ctx.measureText(text).width
  if (fullWidth <= maxWidth) return text

  const ellipsisWidth = ctx.measureText(ELLIPSIS).width * 0.75

  // Can't even fit ellipsis
  if (ellipsisWidth > maxWidth) {
    const twoDotsWidth = ctx.measureText(TWO_DOT_LEADER).width * 0.75
    if (twoDotsWidth < maxWidth) return TWO_DOT_LEADER

    const oneDotWidth = ctx.measureText(ONE_DOT_LEADER).width * 0.75
    return oneDotWidth < maxWidth ? ONE_DOT_LEADER : ''
  }

  let min = 0
  let max = text.length
  let bestLen = 0

  // Binary search for the longest substring that fits with the ellipsis
  while (min <= max) {
    const mid = Math.floor((min + max) * 0.5)

    // Avoid measuring empty string + ellipsis
    if (mid === 0) {
      min = mid + 1
      continue
    }

    const sub = text.substring(0, mid)
    const currentWidth = ctx.measureText(sub).width + ellipsisWidth

    if (currentWidth <= maxWidth) {
      // This length fits, try potentially longer
      bestLen = mid
      min = mid + 1
    } else {
      // Too long, try shorter
      max = mid - 1
    }
  }

  return bestLen === 0 ? ELLIPSIS : text.substring(0, bestLen) + ELLIPSIS
}

/**
 * Draws text within an area, truncating it and adding an ellipsis if necessary.
 */
export function drawTextInArea({
  ctx,
  text,
  area,
  align = 'left'
}: IDrawTextInAreaOptions) {
  const { left, right, bottom, width, centreX } = area

  // Text already fits
  const fullWidth = ctx.measureText(text).width
  if (fullWidth <= width) {
    ctx.textAlign = align
    const x = align === 'left' ? left : align === 'right' ? right : centreX
    ctx.fillText(text, x, bottom)
    return
  }

  // Need to truncate text
  const truncated = truncateTextToWidth(ctx, text, width)
  if (truncated.length === 0) return

  // Draw text - left-aligned to prevent bouncing during resize
  ctx.textAlign = 'left'
  ctx.fillText(truncated.slice(0, -1), left, bottom)
  ctx.rect(left, bottom, width, 1)

  // Draw the ellipsis, right-aligned to the button
  ctx.textAlign = 'right'
  const ellipsis = truncated.at(-1)!
  ctx.fillText(ellipsis, right, bottom, ctx.measureText(ellipsis).width * 0.75)
}
