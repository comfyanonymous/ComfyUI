import type { LocationQuery } from 'vue-router'

const decodeQueryParam = (value: string): string | null => {
  try {
    return decodeURIComponent(value)
  } catch {
    return null
  }
}

const isSafeInternalRedirectPath = (path: string): boolean => {
  // Must be a relative in-app path. Disallow protocol-relative URLs ("//evil.com").
  return path.startsWith('/') && !path.startsWith('//')
}

export const getSafePreviousFullPath = (
  query: LocationQuery
): string | null => {
  const raw = query.previousFullPath
  const value = Array.isArray(raw) ? raw[0] : raw
  if (!value) return null

  const decoded = decodeQueryParam(value)
  if (!decoded) return null

  return isSafeInternalRedirectPath(decoded) ? decoded : null
}
