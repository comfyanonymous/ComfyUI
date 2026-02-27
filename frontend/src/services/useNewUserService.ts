import { ref, shallowRef } from 'vue'
import { createSharedComposable } from '@vueuse/core'
import { useSettingStore } from '@/platform/settings/settingStore'

function _useNewUserService() {
  const settingStore = useSettingStore()
  const pendingCallbacks = shallowRef<Array<() => Promise<void>>>([])
  const isNewUserDetermined = ref(false)
  const isNewUserCached = ref<boolean | null>(null)

  function reset() {
    pendingCallbacks.value = []
    isNewUserDetermined.value = false
    isNewUserCached.value = null
  }

  function checkIsNewUser(): boolean {
    const isNewUserSettings =
      Object.keys(settingStore.settingValues).length === 0 ||
      !settingStore.get('Comfy.TutorialCompleted')
    const hasNoWorkflow = !localStorage.getItem('workflow')
    const hasNoPreviousWorkflow = !localStorage.getItem(
      'Comfy.PreviousWorkflow'
    )

    return isNewUserSettings && hasNoWorkflow && hasNoPreviousWorkflow
  }

  async function registerInitCallback(callback: () => Promise<void>) {
    if (isNewUserDetermined.value) {
      if (isNewUserCached.value) {
        try {
          await callback()
        } catch (error) {
          console.error('New user initialization callback failed:', error)
        }
      }
    } else {
      pendingCallbacks.value = [...pendingCallbacks.value, callback]
    }
  }

  async function initializeIfNewUser() {
    if (isNewUserDetermined.value) return

    isNewUserCached.value = checkIsNewUser()
    isNewUserDetermined.value = true

    if (!isNewUserCached.value) {
      pendingCallbacks.value = []
      return
    }

    await settingStore.set(
      'Comfy.InstalledVersion',
      __COMFYUI_FRONTEND_VERSION__
    )

    for (const callback of pendingCallbacks.value) {
      try {
        await callback()
      } catch (error) {
        console.error('New user initialization callback failed:', error)
      }
    }

    pendingCallbacks.value = []
  }

  function isNewUser(): boolean | null {
    return isNewUserDetermined.value ? isNewUserCached.value : null
  }

  return {
    registerInitCallback,
    initializeIfNewUser,
    isNewUser,
    reset
  }
}

export const useNewUserService = createSharedComposable(_useNewUserService)
