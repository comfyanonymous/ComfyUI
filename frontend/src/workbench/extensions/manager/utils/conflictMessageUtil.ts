import type { ConflictDetail } from '@/workbench/extensions/manager/types/conflictDetectionTypes'

/**
 * Generates a localized conflict message for a given conflict detail.
 * This function should be used anywhere conflict messages need to be displayed.
 *
 * @param conflict The conflict detail object
 * @param t The i18n translation function
 * @returns A localized conflict message string
 */
export function getConflictMessage(
  conflict: ConflictDetail,
  t: (key: string, params?: Record<string, unknown>) => string
): string {
  const messageKey = `manager.conflicts.conflictMessages.${conflict.type}`

  // For version and compatibility conflicts, use interpolated message
  if (
    conflict.type === 'comfyui_version' ||
    conflict.type === 'frontend_version' ||
    conflict.type === 'os' ||
    conflict.type === 'accelerator'
  ) {
    return t(messageKey, {
      current: conflict.current_value,
      required: conflict.required_value
    })
  }

  // For banned, pending, and import_failed, use simple message
  if (
    conflict.type === 'banned' ||
    conflict.type === 'pending' ||
    conflict.type === 'import_failed'
  ) {
    return t(messageKey)
  }

  // Fallback to generic message with interpolation
  return t('manager.conflicts.conflictMessages.generic', {
    current: conflict.current_value,
    required: conflict.required_value
  })
}

/**
 * Generates conflict messages for multiple conflicts and joins them.
 *
 * @param conflicts Array of conflict details
 * @param t The i18n translation function
 * @param separator The separator to use when joining messages (default: '; ')
 * @returns A single string with all conflict messages joined
 */
export function getJoinedConflictMessages(
  conflicts: ConflictDetail[],
  t: (key: string, params?: Record<string, unknown>) => string,
  separator = '; '
): string {
  return conflicts
    .map((conflict) => getConflictMessage(conflict, t))
    .join(separator)
}
