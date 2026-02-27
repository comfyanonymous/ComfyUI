/**
 * Return a local date key in YYYY-MM-DD format for grouping.
 *
 * @param ts Unix timestamp in milliseconds
 * @returns Local date key string
 */
export const dateKey = (ts: number): string => {
  const d = new Date(ts)
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

/**
 * Check if a timestamp is on the same local day as today.
 *
 * @param ts Unix timestamp in milliseconds
 * @returns True if today
 */
export const isToday = (ts: number): boolean => {
  const d = new Date(ts)
  const now = new Date()
  return (
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate()
  )
}

/**
 * Check if a timestamp corresponds to yesterday in local time.
 *
 * @param ts Unix timestamp in milliseconds
 * @returns True if yesterday
 */
export const isYesterday = (ts: number): boolean => {
  const d = new Date(ts)
  const now = new Date()
  const yest = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1)
  return (
    d.getFullYear() === yest.getFullYear() &&
    d.getMonth() === yest.getMonth() &&
    d.getDate() === yest.getDate()
  )
}

/**
 * Localized short month + day label, e.g. "Jan 2".
 *
 * @param ts Unix timestamp in milliseconds
 * @param locale BCP-47 locale string
 * @returns Localized month/day label
 */
export const formatShortMonthDay = (ts: number, locale: string): string => {
  const d = new Date(ts)
  return new Intl.DateTimeFormat(locale, {
    month: 'short',
    day: 'numeric'
  }).format(d)
}

/**
 * Localized clock time, e.g. "10:05:06" with locale defaults for 12/24 hour.
 *
 * @param ts Unix timestamp in milliseconds
 * @param locale BCP-47 locale string
 * @returns Localized time string
 */
export const formatClockTime = (ts: number, locale: string): string => {
  const d = new Date(ts)
  return new Intl.DateTimeFormat(locale, {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit'
  }).format(d)
}
