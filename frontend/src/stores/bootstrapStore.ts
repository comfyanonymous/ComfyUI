import { until, useAsyncState } from '@vueuse/core'
import { defineStore, storeToRefs } from 'pinia'

import { isCloud } from '@/platform/distribution/types'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { api } from '@/scripts/api'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'
import { useUserStore } from '@/stores/userStore'

export const useBootstrapStore = defineStore('bootstrap', () => {
  const settingStore = useSettingStore()
  const workflowStore = useWorkflowStore()

  const {
    isReady: isI18nReady,
    error: i18nError,
    execute: loadI18n
  } = useAsyncState(
    async () => {
      const { mergeCustomNodesI18n } = await import('@/i18n')
      const i18nData = await api.getCustomNodesI18n()
      mergeCustomNodesI18n(i18nData)
    },
    undefined,
    { immediate: false }
  )

  let storesLoaded = false

  function loadAuthenticatedStores() {
    if (storesLoaded) return
    storesLoaded = true
    void settingStore.load()
    void workflowStore.loadWorkflows()
  }

  async function startStoreBootstrap() {
    const userStore = useUserStore()
    await userStore.initialize()

    if (isCloud) {
      const { isInitialized } = storeToRefs(useFirebaseAuthStore())
      await until(isInitialized).toBe(true)
    }

    const { needsLogin } = storeToRefs(userStore)
    await until(needsLogin).toBe(false)

    void loadI18n()
    loadAuthenticatedStores()
  }

  return {
    isI18nReady,
    i18nError,
    startStoreBootstrap
  }
})
