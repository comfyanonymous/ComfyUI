import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'
import { adjustColor } from '@/utils/colorUtil'

/**
 * Applies light theme color adjustments to a color
 */
export function applyLightThemeColor(color?: string): string {
  if (!color) return ''

  if (!useColorPaletteStore().completedActivePalette.light_theme) return color

  return adjustColor(color, { lightness: 0.5 })
}
