import { defineStore } from 'pinia'
import { computed } from 'vue'

import type { ActionBarButton } from '@/types/comfy'

import { useExtensionStore } from './extensionStore'

export const useActionBarButtonStore = defineStore('actionBarButton', () => {
  const extensionStore = useExtensionStore()

  const buttons = computed<ActionBarButton[]>(() =>
    extensionStore.extensions.flatMap((e) => e.actionBarButtons ?? [])
  )

  return {
    buttons
  }
})
