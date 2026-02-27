import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

import type { TaskItemImpl } from '@/stores/queueStore'
import { useQueueStore } from '@/stores/queueStore'
import { useSidebarTabStore } from '@/stores/workspace/sidebarTabStore'

const getAddedAssetCount = (task: TaskItemImpl): number => {
  if (typeof task.outputsCount === 'number') {
    return Math.max(task.outputsCount, 0)
  }

  return task.previewOutput ? 1 : 0
}

export const useAssetsSidebarBadgeStore = defineStore(
  'assetsSidebarBadge',
  () => {
    const queueStore = useQueueStore()
    const sidebarTabStore = useSidebarTabStore()

    const unseenAddedAssetsCount = ref(0)
    const countedHistoryAssetsByJobId = ref(new Map<string, number>())
    const hasInitializedHistory = ref(false)

    const markCurrentHistoryAsSeen = () => {
      countedHistoryAssetsByJobId.value = new Map(
        queueStore.historyTasks.map((task) => [
          task.jobId,
          getAddedAssetCount(task)
        ])
      )
    }

    watch(
      () =>
        [
          queueStore.historyTasks,
          queueStore.hasFetchedHistorySnapshot
        ] as const,
      ([historyTasks, hasFetchedHistorySnapshot]) => {
        if (!hasFetchedHistorySnapshot) {
          return
        }

        if (!hasInitializedHistory.value) {
          hasInitializedHistory.value = true
          markCurrentHistoryAsSeen()
          return
        }

        const isAssetsTabOpen = sidebarTabStore.activeSidebarTabId === 'assets'
        const previousCountedAssetsByJobId = countedHistoryAssetsByJobId.value
        const nextCountedAssetsByJobId = new Map<string, number>()

        for (const task of historyTasks) {
          const jobId = task.jobId
          if (!jobId) {
            continue
          }

          const countedAssets = previousCountedAssetsByJobId.get(jobId) ?? 0
          const currentAssets = getAddedAssetCount(task)
          const hasSeenJob = previousCountedAssetsByJobId.has(jobId)

          if (!isAssetsTabOpen && !hasSeenJob) {
            unseenAddedAssetsCount.value += currentAssets
          } else if (!isAssetsTabOpen && currentAssets > countedAssets) {
            unseenAddedAssetsCount.value += currentAssets - countedAssets
          }

          nextCountedAssetsByJobId.set(
            jobId,
            Math.max(countedAssets, currentAssets)
          )
        }

        countedHistoryAssetsByJobId.value = nextCountedAssetsByJobId
      },
      { immediate: true }
    )

    watch(
      () => sidebarTabStore.activeSidebarTabId,
      (activeSidebarTabId) => {
        if (activeSidebarTabId !== 'assets') {
          return
        }

        unseenAddedAssetsCount.value = 0
        markCurrentHistoryAsSeen()
      }
    )

    return {
      unseenAddedAssetsCount
    }
  }
)
