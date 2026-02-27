const FF_PREFIX = 'ff:'

/**
 * Gets a dev-time feature flag override from localStorage.
 * Stripped from production builds via import.meta.env.DEV tree-shaking.
 *
 * Returns undefined (not null) as the "no override" sentinel because
 * null is a valid JSON value — JSON.parse('null') returns null.
 * Using undefined avoids ambiguity between "no override set" and
 * "override explicitly set to null".
 *
 * Usage in browser console:
 *   localStorage.setItem('ff:team_workspaces_enabled', 'true')
 *   localStorage.removeItem('ff:team_workspaces_enabled')
 */
export function getDevOverride<T>(flagKey: string): T | undefined {
  if (!import.meta.env.DEV) return undefined
  const raw = localStorage.getItem(`${FF_PREFIX}${flagKey}`)
  if (raw === null) return undefined
  try {
    return JSON.parse(raw) as T
  } catch {
    console.warn(`[ff] Invalid JSON for override "${flagKey}":`, raw)
    return undefined
  }
}
