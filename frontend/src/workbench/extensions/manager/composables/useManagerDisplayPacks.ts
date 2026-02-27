import { whenever } from '@vueuse/core'
import { orderBy } from 'es-toolkit/compat'
import { compare, valid } from 'semver'
import type { Ref } from 'vue'
import { computed } from 'vue'

import { useRegistrySearchGateway } from '@/services/gateway/registrySearchGateway'
import type { components } from '@/types/comfyRegistryTypes'
import { useInstalledPacks } from '@/workbench/extensions/manager/composables/nodePack/useInstalledPacks'
import { useWorkflowPacks } from '@/workbench/extensions/manager/composables/nodePack/useWorkflowPacks'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import { useConflictDetectionStore } from '@/workbench/extensions/manager/stores/conflictDetectionStore'
import { ManagerTab } from '@/workbench/extensions/manager/types/comfyManagerTypes'

type NodePack = components['schemas']['Node']

export function useManagerDisplayPacks(
  selectedTabId: Ref<string | null>,
  searchResults: Ref<NodePack[]>,
  searchQuery: Ref<string>,
  sortField: Ref<string>
) {
  const comfyManagerStore = useComfyManagerStore()
  const conflictDetectionStore = useConflictDetectionStore()
  const { getSortValue, getSortableFields } = useRegistrySearchGateway()

  const {
    startFetchInstalled,
    filterInstalledPack,
    installedPacks,
    isLoading: isLoadingInstalled,
    isReady: installedPacksReady
  } = useInstalledPacks()

  const {
    startFetchWorkflowPacks,
    filterWorkflowPack,
    workflowPacks,
    isLoading: isLoadingWorkflow,
    isReady: workflowPacksReady
  } = useWorkflowPacks()

  const tabType = computed(() => selectedTabId.value as ManagerTab | null)
  const isEmptySearch = computed(() => searchQuery.value === '')

  // Sorting function for packs not from searchResults
  const sortPacks = (packs: NodePack[]) => {
    if (!sortField.value || packs.length === 0) return packs

    const sortableFields = getSortableFields()
    const fieldConfig = sortableFields.find((f) => f.id === sortField.value)
    const direction = fieldConfig?.direction || 'desc'

    return orderBy(
      packs,
      [(pack) => getSortValue(pack, sortField.value)],
      [direction]
    )
  }

  // Filter functions
  const filterNotInstalled = (packs: NodePack[]) =>
    packs.filter((p) => !comfyManagerStore.isPackInstalled(p.id))

  const filterConflicting = (packs: NodePack[]) =>
    packs.filter(
      (p) =>
        !!p.id &&
        conflictDetectionStore.conflictedPackages.some(
          (c) => c.package_id === p.id
        )
    )

  const filterOutdated = (packs: NodePack[]) =>
    packs.filter((p) => {
      const installedVersion = comfyManagerStore.getInstalledPackVersion(
        p.id ?? ''
      )
      const latestVersion = p.latest_version?.version
      if (
        !comfyManagerStore.isPackInstalled(p.id) ||
        !installedVersion ||
        !latestVersion ||
        !valid(installedVersion) // nightly builds
      ) {
        return false
      }
      return compare(latestVersion, installedVersion) > 0
    })

  // Data fetching triggers using whenever
  const needsInstalledPacks = computed(() =>
    [
      ManagerTab.AllInstalled,
      ManagerTab.UpdateAvailable,
      ManagerTab.Conflicting
    ].includes(tabType.value as ManagerTab)
  )

  const needsWorkflowPacks = computed(() =>
    [ManagerTab.Workflow, ManagerTab.Missing].includes(
      tabType.value as ManagerTab
    )
  )

  whenever(
    () =>
      needsInstalledPacks.value &&
      !installedPacksReady.value &&
      !isLoadingInstalled.value,
    () => startFetchInstalled()
  )

  whenever(
    () =>
      needsWorkflowPacks.value &&
      !workflowPacksReady.value &&
      !isLoadingWorkflow.value,
    () => startFetchWorkflowPacks()
  )

  // For Missing tab, also need installed packs to determine what's missing
  whenever(
    () =>
      tabType.value === ManagerTab.Missing &&
      !installedPacksReady.value &&
      !isLoadingInstalled.value,
    () => startFetchInstalled()
  )

  // Single computed for display packs - replaces 7 watches
  const displayPacks = computed(() => {
    const tab = tabType.value
    const hasSearch = !isEmptySearch.value

    switch (tab) {
      case ManagerTab.All:
        return searchResults.value

      case ManagerTab.NotInstalled:
        return filterNotInstalled(searchResults.value)

      case ManagerTab.AllInstalled:
        return hasSearch
          ? filterInstalledPack(searchResults.value)
          : sortPacks(installedPacks.value)

      case ManagerTab.UpdateAvailable:
        return sortPacks(
          filterOutdated(
            hasSearch
              ? filterInstalledPack(searchResults.value)
              : installedPacks.value
          )
        )

      case ManagerTab.Conflicting:
        return sortPacks(
          filterConflicting(
            hasSearch
              ? filterInstalledPack(searchResults.value)
              : installedPacks.value
          )
        )

      case ManagerTab.Workflow: {
        return hasSearch
          ? filterWorkflowPack(searchResults.value)
          : sortPacks(workflowPacks.value)
      }

      case ManagerTab.Missing: {
        const base = hasSearch
          ? filterWorkflowPack(searchResults.value)
          : workflowPacks.value
        return sortPacks(filterNotInstalled(base))
      }

      default:
        return searchResults.value
    }
  })

  // Loading state - single computed
  const isLoading = computed(() => {
    const tab = tabType.value
    if (
      [
        ManagerTab.AllInstalled,
        ManagerTab.UpdateAvailable,
        ManagerTab.Conflicting
      ].includes(tab as ManagerTab)
    ) {
      return isLoadingInstalled.value
    }
    if ([ManagerTab.Workflow, ManagerTab.Missing].includes(tab as ManagerTab)) {
      return isLoadingWorkflow.value
    }
    return false
  })

  const missingNodePacks = computed(() =>
    filterNotInstalled(workflowPacks.value)
  )

  return {
    displayPacks,
    isLoading,
    isLoadingInstalled,
    isLoadingWorkflow,
    installedPacks,
    workflowPacks,
    filterInstalledPack,
    filterWorkflowPack,
    missingNodePacks
  }
}
