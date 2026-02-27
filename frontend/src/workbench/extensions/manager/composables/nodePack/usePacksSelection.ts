import { valid } from 'semver'
import { computed } from 'vue'
import type { Ref } from 'vue'

import type { components } from '@/types/comfyRegistryTypes'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'

type NodePack = components['schemas']['Node']

type SelectionState = 'all-installed' | 'none-installed' | 'mixed'

/**
 * Composable for managing multi-package selection states
 * Handles installation status tracking and selection state determination
 */
export function usePacksSelection(nodePacks: Ref<NodePack[]>) {
  const managerStore = useComfyManagerStore()

  const installedPacks = computed(() =>
    nodePacks.value.filter((pack) => managerStore.isPackInstalled(pack.id))
  )

  const notInstalledPacks = computed(() =>
    nodePacks.value.filter((pack) => !managerStore.isPackInstalled(pack.id))
  )

  const isAllInstalled = computed(
    () => installedPacks.value.length === nodePacks.value.length
  )

  const isNoneInstalled = computed(
    () => notInstalledPacks.value.length === nodePacks.value.length
  )

  const isMixed = computed(
    () => installedPacks.value.length > 0 && notInstalledPacks.value.length > 0
  )

  const selectionState = computed<SelectionState>(() => {
    if (isAllInstalled.value) return 'all-installed'
    if (isNoneInstalled.value) return 'none-installed'
    return 'mixed'
  })

  /**
   * Nightly packs are installed packs with a non-semver version (git hash)
   * that are also enabled
   */
  const nightlyPacks = computed(() =>
    installedPacks.value.filter((pack) => {
      if (!pack.id) return false
      const version = managerStore.getInstalledPackVersion(pack.id)
      const isNightly = !!version && !valid(version)
      const isEnabled = managerStore.isPackEnabled(pack.id)
      return isNightly && isEnabled
    })
  )

  const hasNightlyPacks = computed(() => nightlyPacks.value.length > 0)

  return {
    installedPacks,
    notInstalledPacks,
    isAllInstalled,
    isNoneInstalled,
    isMixed,
    selectionState,
    nightlyPacks,
    hasNightlyPacks
  }
}
