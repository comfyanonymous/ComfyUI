/**
 * Distribution types and compile-time constants for managing
 * multi-distribution builds (Desktop, Localhost, Cloud)
 */

type Distribution = 'desktop' | 'localhost' | 'cloud'

declare global {
  const __DISTRIBUTION__: Distribution
  const __IS_NIGHTLY__: boolean
}

/** Current distribution - replaced at compile time */
const DISTRIBUTION: Distribution = __DISTRIBUTION__

export const isDesktop = DISTRIBUTION === 'desktop'
export const isCloud = DISTRIBUTION === 'cloud'

/**
 * Whether this is a nightly build (from main branch).
 * Nightly builds may show experimental features and surveys.
 * @public
 */
export const isNightly = __IS_NIGHTLY__
