import { get, useAsyncState } from '@vueuse/core'
import type { Ref } from 'vue'

import { useComfyRegistryStore } from '@/stores/comfyRegistryStore'
import type { UseNodePacksOptions } from '@/workbench/extensions/manager/types/comfyManagerTypes'

/**
 * Handles fetching node packs from the registry given a list of node pack IDs
 */
export const useNodePacks = (
  packsIds: string[] | Ref<string[]>,
  options: UseNodePacksOptions = {}
) => {
  const { immediate = false } = options
  const { getPacksByIds } = useComfyRegistryStore()

  const fetchPacks = () => getPacksByIds.call(get(packsIds).filter(Boolean))

  const {
    isReady,
    isLoading,
    error,
    execute,
    state: nodePacks
  } = useAsyncState(fetchPacks, [], {
    immediate
  })

  const cleanup = () => {
    getPacksByIds.cancel()
    isReady.value = false
    isLoading.value = false
  }

  return {
    error,
    isLoading,
    isReady,
    nodePacks,
    startFetch: execute,
    cleanup
  }
}
