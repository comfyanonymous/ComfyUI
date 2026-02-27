/**
 * Widget prop filtering utilities
 * Filters out style-related and customization props from PrimeVue components
 * to maintain consistent widget appearance across the application
 */

// Props to exclude based on the widget interface specifications
export const STANDARD_EXCLUDED_PROPS = [
  'style',
  'class',
  'dt',
  'pt',
  'ptOptions',
  'unstyled'
] as const

export const INPUT_EXCLUDED_PROPS = [
  ...STANDARD_EXCLUDED_PROPS,
  'inputClass',
  'inputStyle'
] as const

export const PANEL_EXCLUDED_PROPS = [
  ...STANDARD_EXCLUDED_PROPS,
  'panelClass',
  'panelStyle',
  'overlayClass'
] as const

// export const IMAGE_EXCLUDED_PROPS = [
//   ...STANDARD_EXCLUDED_PROPS,
//   'imageClass',
//   'imageStyle'
// ] as const

export const GALLERIA_EXCLUDED_PROPS = [
  ...STANDARD_EXCLUDED_PROPS,
  'thumbnailsPosition',
  'verticalThumbnailViewPortHeight',
  'indicatorsPosition',
  'maskClass',
  'containerStyle',
  'containerClass',
  'galleriaClass'
] as const

export const BADGE_EXCLUDED_PROPS = [
  ...STANDARD_EXCLUDED_PROPS,
  'badgeClass'
] as const

/**
 * Filters widget props by excluding specified properties
 * @param props - The props object to filter
 * @param excludeList - List of property names to exclude
 * @returns Filtered props object
 */
export function filterWidgetProps<T extends object>(
  props: T | undefined,
  excludeList: readonly string[]
): Partial<T> {
  if (!props) return {}

  const filtered: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(props)) {
    if (!excludeList.includes(key)) {
      filtered[key] = value
    }
  }
  return filtered as Partial<T>
}
