import { clamp } from 'es-toolkit/math'

/**
 * Clamp a numeric value to an integer percent in the range [0, 100].
 *
 * @param value Numeric value expected to be a percentage (0-100)
 * @returns Integer percent between 0 and 100
 */
export const clampPercentInt = (value?: number): number => {
  const v = Math.round(value ?? 0)
  return clamp(v, 0, 100)
}

/**
 * Format a percentage (0-100) using the provided locale with 0 fraction digits.
 *
 * @param locale BCP-47 locale string
 * @param value0to100 Percent value in [0, 100]
 * @returns Localized percent string, e.g. "42%"
 */
export const formatPercent0 = (locale: string, value0to100: number): string => {
  const v = clampPercentInt(value0to100)
  return new Intl.NumberFormat(locale, {
    style: 'percent',
    maximumFractionDigits: 0
  }).format((v || 0) / 100)
}
