import type { LinkId, RerouteId } from '@/renderer/core/layout/types'

/**
 * Creates a unique key for identifying link segments in spatial indexes
 */
export function makeLinkSegmentKey(
  linkId: LinkId,
  rerouteId: RerouteId | null
): string {
  return `${linkId}:${rerouteId ?? 'final'}`
}
