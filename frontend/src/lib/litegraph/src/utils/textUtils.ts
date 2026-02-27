/**
 * Truncates text to fit within a given width using binary search for optimal performance.
 * @param ctx The canvas rendering context used for text measurement
 * @param text The text to truncate
 * @param maxWidth The maximum width the text should occupy
 * @param ellipsis The ellipsis string to append (default: "...")
 * @returns The truncated text with ellipsis if needed
 */
export function truncateText(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
  ellipsis: string = '...'
): string {
  const textWidth = ctx.measureText(text).width

  if (textWidth <= maxWidth || maxWidth <= 0) {
    return text
  }

  const ellipsisWidth = ctx.measureText(ellipsis).width
  const availableWidth = maxWidth - ellipsisWidth

  if (availableWidth <= 0) {
    return ellipsis
  }

  // Binary search for the right length
  let low = 0
  let high = text.length
  let bestFit = 0

  while (low <= high) {
    const mid = Math.floor((low + high) / 2)
    const testText = text.substring(0, mid)
    const testWidth = ctx.measureText(testText).width

    if (testWidth <= availableWidth) {
      bestFit = mid
      low = mid + 1
    } else {
      high = mid - 1
    }
  }

  return text.substring(0, bestFit) + ellipsis
}
