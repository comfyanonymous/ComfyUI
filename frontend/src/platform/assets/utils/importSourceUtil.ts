import type { ImportSource } from '@/platform/assets/types/importSource'

/**
 * Check if a URL belongs to a specific import source
 */
export function validateSourceUrl(url: string, source: ImportSource): boolean {
  try {
    const hostname = new URL(url).hostname.toLowerCase()
    return source.hostnames.some(
      (h) => hostname === h || hostname.endsWith(`.${h}`)
    )
  } catch {
    return false
  }
}
