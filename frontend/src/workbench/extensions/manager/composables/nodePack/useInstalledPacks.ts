import { whenever } from '@vueuse/core'
import { computed, onUnmounted, ref } from 'vue'

import type { components } from '@/types/comfyRegistryTypes'
import { useNodePacks } from '@/workbench/extensions/manager/composables/nodePack/useNodePacks'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import type { UseNodePacksOptions } from '@/workbench/extensions/manager/types/comfyManagerTypes'

export const useInstalledPacks = (options: UseNodePacksOptions = {}) => {
  const comfyManagerStore = useComfyManagerStore()

  // Flag to prevent duplicate fetches during initialization
  const isInitializing = ref(false)
  const lastFetchedIds = ref<string>('')

  const installedPackIds = computed(() =>
    Array.from(comfyManagerStore.installedPacksIds)
  )

  const { startFetch, cleanup, error, isLoading, nodePacks, isReady } =
    useNodePacks(installedPackIds, options)

  const filterInstalledPack = (packs: components['schemas']['Node'][]) =>
    packs.filter((pack) => comfyManagerStore.isPackInstalled(pack.id))

  const startFetchInstalled = async () => {
    // Prevent duplicate calls during initialization
    if (isInitializing.value) {
      return
    }

    isInitializing.value = true
    try {
      if (comfyManagerStore.installedPacksIds.size === 0) {
        await comfyManagerStore.refreshInstalledList()
      }
      await startFetch()
    } finally {
      isInitializing.value = false
    }
  }

  // When installedPackIds changes, we need to update the nodePacks
  // But only if the IDs actually changed (not just array reference)
  whenever(installedPackIds, async (newIds) => {
    const newIdsStr = newIds.sort().join(',')
    if (newIdsStr !== lastFetchedIds.value && !isInitializing.value) {
      lastFetchedIds.value = newIdsStr
      await startFetch()
    }
  })

  onUnmounted(() => {
    cleanup()
  })

  // Create a computed property that provides installed pack info with versions
  const installedPacksWithVersions = computed(() => {
    const result: Array<{ id: string; version: string }> = []

    for (const pack of Object.values(comfyManagerStore.installedPacks)) {
      const id = pack.cnr_id || pack.aux_id
      if (id) {
        result.push({
          id,
          version: pack.ver ?? ''
        })
      }
    }

    return result
  })

  return {
    error,
    isLoading,
    isReady,
    installedPacks: nodePacks,
    installedPacksWithVersions,
    startFetchInstalled,
    filterInstalledPack
  }
}
