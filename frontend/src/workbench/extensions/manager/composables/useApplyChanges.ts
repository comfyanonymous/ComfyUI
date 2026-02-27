import { createSharedComposable, useEventListener } from '@vueuse/core'
import { ref } from 'vue'

import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import { api } from '@/scripts/api'
import { useCommandStore } from '@/stores/commandStore'
import { useConflictDetection } from '@/workbench/extensions/manager/composables/useConflictDetection'
import { useComfyManagerService } from '@/workbench/extensions/manager/services/comfyManagerService'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'

export const useApplyChanges = createSharedComposable(() => {
  const comfyManagerStore = useComfyManagerStore()
  const settingStore = useSettingStore()
  const { runFullConflictAnalysis } = useConflictDetection()

  const isRestarting = ref(false)
  const isRestartCompleted = ref(false)

  async function applyChanges(onClose?: () => void) {
    if (isRestarting.value) return

    isRestarting.value = true
    isRestartCompleted.value = false

    const originalToastSetting = settingStore.get(
      'Comfy.Toast.DisableReconnectingToast'
    )
    const onReconnect = async () => {
      try {
        comfyManagerStore.setStale()
        await useCommandStore().execute('Comfy.RefreshNodeDefinitions')
        await useWorkflowService().reloadCurrentWorkflow()
        runFullConflictAnalysis().catch((err) => {
          console.error('[useApplyChanges] Conflict analysis failed:', err)
        })
      } catch (err) {
        console.error('[useApplyChanges] Post-reconnect tasks failed:', err)
      } finally {
        try {
          await settingStore.set(
            'Comfy.Toast.DisableReconnectingToast',
            originalToastSetting
          )
        } catch (err) {
          console.error(
            '[useApplyChanges] Failed to restore reconnect toast setting:',
            err
          )
        }
        isRestarting.value = false
        isRestartCompleted.value = true

        setTimeout(() => {
          onClose?.()
        }, 3000)
      }
    }

    let hasReconnected = false
    const stopReconnectListener = useEventListener(
      api,
      'reconnected',
      () => {
        clearTimeout(reconnectTimeout)
        hasReconnected = true
        void onReconnect()
      },
      { once: true }
    )

    const RECONNECT_TIMEOUT_MS = 120_000 // 2 minutes
    let reconnectTimeout: ReturnType<typeof setTimeout> | undefined

    try {
      await settingStore.set('Comfy.Toast.DisableReconnectingToast', true)
      await useComfyManagerService().rebootComfyUI()
      reconnectTimeout = setTimeout(async () => {
        if (hasReconnected) return
        stopReconnectListener()
        try {
          await settingStore.set(
            'Comfy.Toast.DisableReconnectingToast',
            originalToastSetting
          )
        } catch (err) {
          console.error(
            '[useApplyChanges] Failed to restore reconnect toast setting:',
            err
          )
        }
        isRestarting.value = false
        isRestartCompleted.value = false
        console.error('[useApplyChanges] Reconnect timed out')
      }, RECONNECT_TIMEOUT_MS)
    } catch (error) {
      stopReconnectListener()
      try {
        await settingStore.set(
          'Comfy.Toast.DisableReconnectingToast',
          originalToastSetting
        )
      } catch (restoreErr) {
        console.error(
          '[useApplyChanges] Failed to restore reconnect toast setting:',
          restoreErr
        )
      } finally {
        isRestarting.value = false
        isRestartCompleted.value = false
        onClose?.()
      }
      throw error
    }
  }

  return { isRestarting, isRestartCompleted, applyChanges }
})
