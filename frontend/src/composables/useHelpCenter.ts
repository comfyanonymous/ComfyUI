import { storeToRefs } from 'pinia'
import { computed, onMounted } from 'vue'

import { useSettingStore } from '@/platform/settings/settingStore'
import { useTelemetry } from '@/platform/telemetry'
import { useReleaseStore } from '@/platform/updates/common/releaseStore'
import { useHelpCenterStore } from '@/stores/helpCenterStore'
import type { HelpCenterTriggerLocation } from '@/stores/helpCenterStore'
import { useConflictAcknowledgment } from '@/workbench/extensions/manager/composables/useConflictAcknowledgment'
import { useConflictDetection } from '@/workbench/extensions/manager/composables/useConflictDetection'
import { useNodeConflictDialog } from '@/workbench/extensions/manager/composables/useNodeConflictDialog'

export function useHelpCenter(
  triggerFrom: HelpCenterTriggerLocation = 'sidebar'
) {
  const settingStore = useSettingStore()
  const releaseStore = useReleaseStore()
  const helpCenterStore = useHelpCenterStore()
  const { isVisible: isHelpCenterVisible, triggerLocation } =
    storeToRefs(helpCenterStore)
  const { shouldShowRedDot: showReleaseRedDot } = storeToRefs(releaseStore)

  const conflictDetection = useConflictDetection()
  const { show: showNodeConflictDialog } = useNodeConflictDialog()

  // Use conflict acknowledgment state from composable - call only once
  const { shouldShowRedDot: shouldShowConflictRedDot, markConflictsAsSeen } =
    useConflictAcknowledgment()

  // Use either release red dot or conflict red dot
  const shouldShowRedDot = computed((): boolean => {
    const releaseRedDot = showReleaseRedDot.value
    return releaseRedDot || shouldShowConflictRedDot.value
  })

  const sidebarLocation = computed(() =>
    settingStore.get('Comfy.Sidebar.Location')
  )

  /**
   * Toggle Help Center and track UI button click.
   */
  const toggleHelpCenter = () => {
    useTelemetry()?.trackUiButtonClicked({
      button_id: `${triggerFrom}_help_center_toggled`
    })
    helpCenterStore.toggle(triggerFrom)
  }

  const closeHelpCenter = () => {
    helpCenterStore.hide()
  }

  /**
   * Handle What's New popup dismissal
   * Check if conflict modal should be shown after ComfyUI update
   */
  const handleWhatsNewDismissed = async () => {
    try {
      // Check if conflict modal should be shown after update
      const shouldShow =
        await conflictDetection.shouldShowConflictModalAfterUpdate()
      if (shouldShow) {
        showConflictModal()
      }
    } catch (error) {
      console.error('[HelpCenter] Error checking conflict modal:', error)
    }
  }

  /**
   * Show the node conflict dialog with current conflict data
   */
  const showConflictModal = () => {
    void showNodeConflictDialog({
      showAfterWhatsNew: true,
      dialogComponentProps: {
        onClose: () => {
          markConflictsAsSeen()
        }
      }
    })
  }

  // Initialize release store on mount
  onMounted(async () => {
    // Initialize release store to fetch releases for toast and popup
    await releaseStore.initialize()
  })

  return {
    isHelpCenterVisible,
    triggerLocation,
    shouldShowRedDot,
    sidebarLocation,
    toggleHelpCenter,
    closeHelpCenter,
    handleWhatsNewDismissed
  }
}
