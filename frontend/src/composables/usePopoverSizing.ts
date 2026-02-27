import { computed } from 'vue'
import type { CSSProperties, ComputedRef } from 'vue'

interface PopoverSizeOptions {
  minWidth?: string
  maxWidth?: string
}

/**
 * Composable for managing popover sizing styles
 * @param options Popover size configuration
 * @returns Computed style object for popover sizing
 */
export function usePopoverSizing(
  options: PopoverSizeOptions
): ComputedRef<CSSProperties> {
  return computed(() => {
    const { minWidth, maxWidth } = options
    const style: CSSProperties = {}

    if (minWidth) {
      style.minWidth = minWidth
    }

    if (maxWidth) {
      style.maxWidth = maxWidth
    }

    return style
  })
}
