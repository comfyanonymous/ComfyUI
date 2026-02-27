import type { CSSProperties } from 'vue'

interface GridOptions {
  /** Minimum width for each grid item (default: 15rem) */
  minWidth?: string
  /** Maximum width for each grid item (default: 1fr) */
  maxWidth?: string
  /** Padding around the grid (default: 0) */
  padding?: string
  /** Gap between grid items (default: 1rem) */
  gap?: string
  /** Fixed number of columns (overrides auto-fill with minmax) */
  columns?: number
}

/**
 * @deprecated Just use tailwind utilities directly.
 * TODO: Create a common grid layout component if needed.
 * Creates CSS grid styles for responsive grid layouts
 * @param options Grid configuration options
 * @returns CSS properties object for grid styling
 */
export function createGridStyle(options: GridOptions = {}): CSSProperties {
  const {
    minWidth = '15rem',
    maxWidth = '1fr',
    padding = '0',
    gap = '1rem',
    columns
  } = options

  // Runtime validation for columns
  if (columns !== undefined && columns < 1) {
    console.warn('createGridStyle: columns must be >= 1, defaulting to 1')
  }

  const gridTemplateColumns = columns
    ? `repeat(${Math.max(1, columns ?? 1)}, 1fr)`
    : `repeat(auto-fill, minmax(${minWidth}, ${maxWidth}))`

  return {
    display: 'grid',
    gridTemplateColumns,
    padding,
    gap
  }
}
