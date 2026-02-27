import { useLocalStorage } from '@vueuse/core'
import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'

import { useErrorHandling } from '@/composables/useErrorHandling'
import { t } from '@/i18n'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'
import type { ApiKeyAuthHeader } from '@/types/authTypes'
import type { operations } from '@/types/comfyRegistryTypes'

type ComfyApiUser =
  operations['createCustomer']['responses']['201']['content']['application/json']

const STORAGE_KEY = 'comfy_api_key'

export const useApiKeyAuthStore = defineStore('apiKeyAuth', () => {
  const firebaseAuthStore = useFirebaseAuthStore()
  const apiKey = useLocalStorage<string | null>(STORAGE_KEY, null)
  const toastStore = useToastStore()
  const { wrapWithErrorHandlingAsync, toastErrorHandler } = useErrorHandling()

  const currentUser = ref<ComfyApiUser | null>(null)
  const isAuthenticated = computed(() => !!currentUser.value)

  const initializeUserFromApiKey = async () => {
    const createCustomerResponse = await firebaseAuthStore
      .createCustomer()
      .catch((err) => {
        console.error(err)
        return
      })
    if (!createCustomerResponse) {
      apiKey.value = null
      throw new Error(t('auth.login.noAssociatedUser'))
    }
    currentUser.value = createCustomerResponse
  }

  watch(
    apiKey,
    () => {
      if (apiKey.value) {
        // IF API key is set, initialize user
        void initializeUserFromApiKey()
      } else {
        // IF API key is cleared, clear user
        currentUser.value = null
      }
    },
    { immediate: true }
  )

  const reportError = (error: unknown) => {
    if (error instanceof Error && error.message === 'STORAGE_FAILED') {
      toastStore.add({
        severity: 'error',
        summary: t('auth.apiKey.storageFailed'),
        detail: t('auth.apiKey.storageFailedDetail')
      })
    } else {
      toastErrorHandler(error)
    }
  }

  const storeApiKey = wrapWithErrorHandlingAsync(async (newApiKey: string) => {
    apiKey.value = newApiKey
    toastStore.add({
      severity: 'success',
      summary: t('auth.apiKey.stored'),
      detail: t('auth.apiKey.storedDetail'),
      life: 5000
    })
    return true
  }, reportError)

  const clearStoredApiKey = wrapWithErrorHandlingAsync(async () => {
    apiKey.value = null
    toastStore.add({
      severity: 'success',
      summary: t('auth.apiKey.cleared'),
      detail: t('auth.apiKey.clearedDetail'),
      life: 5000
    })
    return true
  }, reportError)

  const getApiKey = () => apiKey.value

  /**
   * Retrieves the appropriate authentication header for API requests if an
   * API key is available, otherwise returns null.
   */
  const getAuthHeader = (): ApiKeyAuthHeader | null => {
    const comfyOrgApiKey = getApiKey()
    if (comfyOrgApiKey) {
      return {
        'X-API-KEY': comfyOrgApiKey
      }
    }
    return null
  }

  return {
    // State
    currentUser,
    isAuthenticated,

    // Actions
    storeApiKey,
    clearStoredApiKey,
    getAuthHeader,
    getApiKey
  }
})
