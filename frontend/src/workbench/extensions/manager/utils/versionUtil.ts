import { isEmpty, isNil } from 'es-toolkit/compat'
import { clean, satisfies } from 'semver'

import config from '@/config'
import type {
  ConflictDetail,
  ConflictType
} from '@/workbench/extensions/manager/types/conflictDetectionTypes'

/**
 * Cleans a version string by removing common prefixes and normalizing format
 * @param version Raw version string (e.g., "v1.2.3", "1.2.3-alpha")
 * @returns Cleaned version string or original if cleaning fails
 */
function cleanVersion(version: string): string {
  return clean(version) || version
}

/**
 * Checks version compatibility and returns conflict details.
 * Supports all semver ranges including >=, <=, >, <, ~, ^ operators.
 * @param type Conflict type (e.g., 'comfyui_version', 'frontend_version')
 * @param currentVersion Current version string
 * @param supportedVersion Required version range string
 * @returns ConflictDetail object if incompatible, null if compatible
 */
export function checkVersionCompatibility(
  type: ConflictType,
  currentVersion?: string,
  supportedVersion?: string
): ConflictDetail | null {
  // If current version is unknown, assume compatible (no conflict)
  if (isNil(currentVersion) || isEmpty(currentVersion)) {
    return null
  }

  // If no version requirement specified, assume compatible (no conflict)
  if (isNil(supportedVersion) || isEmpty(supportedVersion?.trim())) {
    return null
  }

  // Clean and check version compatibility
  const cleanCurrent = cleanVersion(currentVersion)

  // Check if version satisfies the range
  let isCompatible = false
  try {
    isCompatible = satisfies(cleanCurrent, supportedVersion)
  } catch {
    // If semver can't parse it, return conflict
    return {
      type,
      current_value: currentVersion,
      required_value: supportedVersion
    }
  }

  if (isCompatible) return null

  return {
    type,
    current_value: currentVersion,
    required_value: supportedVersion
  }
}

/**
 * get frontend version from config.
 * @returns frontend version string or undefined
 */
export function getFrontendVersion(): string | undefined {
  return config.app_version || import.meta.env.VITE_APP_VERSION || undefined
}
