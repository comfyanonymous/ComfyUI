import { groupBy, uniqBy } from 'es-toolkit/compat'

import { normalizePackId } from '@/utils/packUtils'
import type {
  ConflictDetail,
  ConflictDetectionResult
} from '@/workbench/extensions/manager/types/conflictDetectionTypes'

/**
 * Checks for banned package status conflicts.
 */
export function createBannedConflict(
  isBanned?: boolean
): ConflictDetail | null {
  if (isBanned === true) {
    return {
      type: 'banned',
      current_value: 'installed',
      required_value: 'not_banned'
    }
  }
  return null
}

/**
 * Checks for pending package status conflicts.
 */
export function createPendingConflict(
  isPending?: boolean
): ConflictDetail | null {
  if (isPending === true) {
    return {
      type: 'pending',
      current_value: 'installed',
      required_value: 'not_pending'
    }
  }
  return null
}

/**
 * Groups and deduplicates conflicts by normalized package name.
 * Consolidates multiple conflict sources (registry checks, import failures, disabled packages with version suffix)
 * into a single UI entry per package.
 *
 * Example:
 * - Input: [{name: "pack@1_0_3", conflicts: [...]}, {name: "pack", conflicts: [...]}]
 * - Output: [{name: "pack", conflicts: [...combined unique conflicts...]}]
 *
 * @param conflicts Array of conflict detection results (may have duplicate packages with version suffixes)
 * @returns Array of deduplicated conflict results grouped by normalized package name
 */
export function consolidateConflictsByPackage(
  conflicts: ConflictDetectionResult[]
): ConflictDetectionResult[] {
  // Group conflicts by normalized package name using es-toolkit
  const grouped = groupBy(conflicts, (conflict) =>
    normalizePackId(conflict.package_name)
  )

  // Merge conflicts for each group
  return Object.entries(grouped).map(([packageName, packageConflicts]) => {
    // Flatten all conflicts from the group
    const allConflicts = packageConflicts.flatMap((pc) => pc.conflicts)

    // Remove duplicate conflicts using uniqBy
    const uniqueConflicts = uniqBy(
      allConflicts,
      (conflict) =>
        `${conflict.type}|${conflict.current_value}|${conflict.required_value}`
    )

    // Use the first item as base and update with merged data
    const baseItem = packageConflicts[0]
    return {
      ...baseItem,
      package_name: packageName, // Use normalized name
      conflicts: uniqueConflicts,
      has_conflict: uniqueConflicts.length > 0,
      is_compatible: uniqueConflicts.length === 0
    }
  })
}
