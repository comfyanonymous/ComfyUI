import { useDialogService } from '@/services/dialogService'
import { useDialogStore } from '@/stores/dialogStore'

import SettingDialog from '@/platform/settings/components/SettingDialog.vue'
import type { SettingPanelType } from '@/platform/settings/types'

const DIALOG_KEY = 'global-settings'

export function useSettingsDialog() {
  const dialogService = useDialogService()
  const dialogStore = useDialogStore()

  function hide() {
    dialogStore.closeDialog({ key: DIALOG_KEY })
  }

  function show(panel?: SettingPanelType, settingId?: string) {
    dialogService.showLayoutDialog({
      key: DIALOG_KEY,
      component: SettingDialog,
      props: {
        onClose: hide,
        ...(panel ? { defaultPanel: panel } : {}),
        ...(settingId ? { scrollToSettingId: settingId } : {})
      }
    })
  }

  function showAbout() {
    show('about')
  }

  return { show, hide, showAbout }
}
