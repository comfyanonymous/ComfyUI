import { useStorage } from '@vueuse/core'
import { computed } from 'vue'

import { useConflictDetectionStore } from '@/workbench/extensions/manager/stores/conflictDetectionStore'

/**
 * LocalStorage keys for conflict acknowledgment tracking
 */
const STORAGE_KEYS = {
  CONFLICT_MODAL_DISMISSED: 'Comfy.ConflictModalDismissed',
  CONFLICT_RED_DOT_DISMISSED: 'Comfy.ConflictRedDotDismissed',
  CONFLICT_WARNING_BANNER_DISMISSED: 'Comfy.ConflictWarningBannerDismissed'
} as const

/**
 * Interface for conflict acknowledgment state
 */
export interface ConflictAcknowledgmentState {
  modal_dismissed: boolean
  red_dot_dismissed: boolean
  warning_banner_dismissed: boolean
}

// Shared state - initialized once and reused across all composable calls
const modalDismissed = useStorage(STORAGE_KEYS.CONFLICT_MODAL_DISMISSED, false)
const redDotDismissed = useStorage(
  STORAGE_KEYS.CONFLICT_RED_DOT_DISMISSED,
  false
)
const warningBannerDismissed = useStorage(
  STORAGE_KEYS.CONFLICT_WARNING_BANNER_DISMISSED,
  false
)

/**
 * Composable for managing conflict acknowledgment state in localStorage
 *
 * This handles:
 * - Tracking whether conflict modal has been dismissed
 * - Tracking whether red dot notification has been cleared
 * - Managing per-package conflict acknowledgments
 * - Detecting ComfyUI version changes to reset acknowledgment state
 */
export function useConflictAcknowledgment() {
  const conflictDetectionStore = useConflictDetectionStore()

  // Create computed state object for backward compatibility
  const state = computed<ConflictAcknowledgmentState>(() => ({
    modal_dismissed: modalDismissed.value,
    red_dot_dismissed: redDotDismissed.value,
    warning_banner_dismissed: warningBannerDismissed.value
  }))

  /**
   * Mark red dot notification as dismissed
   */
  function dismissRedDotNotification(): void {
    redDotDismissed.value = true
  }

  /**
   * Mark manager warning banner as dismissed
   */
  function dismissWarningBanner(): void {
    warningBannerDismissed.value = true
    redDotDismissed.value = true
  }

  /**
   * Mark conflicts as seen (unified function for help center and manager)
   */
  function markConflictsAsSeen(): void {
    redDotDismissed.value = true
    modalDismissed.value = true
    warningBannerDismissed.value = true
  }

  const hasConflicts = computed(() => conflictDetectionStore.hasConflicts)
  const shouldShowConflictModal = computed(() => !modalDismissed.value)
  const shouldShowRedDot = computed(() => {
    if (!hasConflicts.value) return false
    if (redDotDismissed.value) return false
    return true
  })
  const shouldShowManagerBanner = computed(() => {
    return hasConflicts.value && !warningBannerDismissed.value
  })

  return {
    // State
    acknowledgmentState: state,
    shouldShowConflictModal,
    shouldShowRedDot,
    shouldShowManagerBanner,

    // Methods
    dismissRedDotNotification,
    dismissWarningBanner,
    markConflictsAsSeen
  }
}
