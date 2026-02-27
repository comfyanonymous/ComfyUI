<template>
  <div
    v-show="isVisible"
    :class="['flex', 'justify-end', 'w-full', 'pointer-events-none']"
  >
    <div
      class="pointer-events-auto flex w-[350px] min-w-[310px] max-h-[60vh] flex-col overflow-hidden rounded-lg border font-inter transition-colors duration-200 ease-in-out"
      :class="containerClass"
      @mouseenter="isHovered = true"
      @mouseleave="isHovered = false"
    >
      <!-- Expanded state -->
      <QueueOverlayExpanded
        v-if="isExpanded"
        v-model:selected-job-tab="selectedJobTab"
        v-model:selected-workflow-filter="selectedWorkflowFilter"
        v-model:selected-sort-mode="selectedSortMode"
        class="flex-1 min-h-0"
        :header-title="headerTitle"
        :queued-count="queuedCount"
        :displayed-job-groups="displayedJobGroups"
        :has-failed-jobs="hasFailedJobs"
        @show-assets="toggleAssetsSidebar"
        @clear-history="onClearHistoryFromMenu"
        @clear-queued="cancelQueuedWorkflows"
        @cancel-item="onCancelItem"
        @delete-item="onDeleteItem"
        @view-item="inspectJobAsset"
      />

      <QueueOverlayActive
        v-else-if="hasActiveJob"
        :total-progress-style="totalProgressStyle"
        :current-node-progress-style="currentNodeProgressStyle"
        :total-percent-formatted="totalPercentFormatted"
        :current-node-percent-formatted="currentNodePercentFormatted"
        :current-node-name="currentNodeName"
        :running-count="runningCount"
        :queued-count="queuedCount"
        :bottom-row-class="bottomRowClass"
        @interrupt-all="interruptAll"
        @clear-queued="cancelQueuedWorkflows"
        @view-all-jobs="viewAllJobs"
      />
    </div>
  </div>

  <ResultGallery
    v-model:active-index="galleryActiveIndex"
    :all-gallery-items="galleryItems"
  />
</template>

<script setup lang="ts">
import { computed, nextTick, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import QueueOverlayActive from '@/components/queue/QueueOverlayActive.vue'
import QueueOverlayExpanded from '@/components/queue/QueueOverlayExpanded.vue'
import ResultGallery from '@/components/sidebar/tabs/queue/ResultGallery.vue'
import { useJobList } from '@/composables/queue/useJobList'
import type { JobListItem } from '@/composables/queue/useJobList'
import { useQueueClearHistoryDialog } from '@/composables/queue/useQueueClearHistoryDialog'
import { useQueueProgress } from '@/composables/queue/useQueueProgress'
import { useResultGallery } from '@/composables/queue/useResultGallery'
import { useErrorHandling } from '@/composables/useErrorHandling'
import { useAssetSelectionStore } from '@/platform/assets/composables/useAssetSelectionStore'
import { isCloud } from '@/platform/distribution/types'
import { api } from '@/scripts/api'
import { useAssetsStore } from '@/stores/assetsStore'
import { useCommandStore } from '@/stores/commandStore'
import { useExecutionStore } from '@/stores/executionStore'
import { useQueueStore } from '@/stores/queueStore'
import { useSidebarTabStore } from '@/stores/workspace/sidebarTabStore'

type OverlayState = 'hidden' | 'active' | 'expanded'

const { expanded, menuHovered = false } = defineProps<{
  expanded?: boolean
  menuHovered?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:expanded', value: boolean): void
}>()

const { t, n } = useI18n()
const queueStore = useQueueStore()
const commandStore = useCommandStore()
const executionStore = useExecutionStore()
const sidebarTabStore = useSidebarTabStore()
const assetsStore = useAssetsStore()
const assetSelectionStore = useAssetSelectionStore()
const { showQueueClearHistoryDialog } = useQueueClearHistoryDialog()
const { wrapWithErrorHandlingAsync } = useErrorHandling()

const {
  totalPercentFormatted,
  currentNodePercentFormatted,
  totalProgressStyle,
  currentNodeProgressStyle
} = useQueueProgress()
const isHovered = ref(false)
const isOverlayHovered = computed(() => isHovered.value || menuHovered)
const internalExpanded = ref(false)
const isExpanded = computed({
  get: () => (expanded === undefined ? internalExpanded.value : expanded),
  set: (value) => {
    if (expanded === undefined) {
      internalExpanded.value = value
    }
    emit('update:expanded', value)
  }
})

const runningCount = computed(() => queueStore.runningTasks.length)
const queuedCount = computed(() => queueStore.pendingTasks.length)
const isExecuting = computed(() => !executionStore.isIdle)
const hasActiveJob = computed(() => runningCount.value > 0 || isExecuting.value)

const overlayState = computed<OverlayState>(() => {
  if (isExpanded.value) return 'expanded'
  if (hasActiveJob.value) return 'active'
  return 'hidden'
})

const showBackground = computed(
  () =>
    overlayState.value === 'expanded' ||
    (overlayState.value === 'active' && isOverlayHovered.value)
)

const isVisible = computed(() => overlayState.value !== 'hidden')

const containerClass = computed(() =>
  showBackground.value
    ? 'border-interface-stroke bg-interface-panel-surface shadow-interface'
    : 'border-transparent bg-transparent shadow-none'
)

const bottomRowClass = computed(
  () =>
    `flex items-center justify-end gap-4 transition-opacity duration-200 ease-in-out ${
      overlayState.value === 'active' && isOverlayHovered.value
        ? 'opacity-100 pointer-events-auto'
        : 'opacity-0 pointer-events-none'
    }`
)
const runningJobsLabel = computed(() =>
  t('sideToolbar.queueProgressOverlay.runningJobsLabel', {
    count: n(runningCount.value)
  })
)
const queuedJobsLabel = computed(() =>
  t('sideToolbar.queueProgressOverlay.queuedJobsLabel', {
    count: n(queuedCount.value)
  })
)
const headerTitle = computed(() => {
  if (!hasActiveJob.value) {
    return t('sideToolbar.queueProgressOverlay.jobQueue')
  }

  if (queuedCount.value === 0) {
    return runningJobsLabel.value
  }

  if (runningCount.value === 0) {
    return queuedJobsLabel.value
  }

  return t('sideToolbar.queueProgressOverlay.runningQueuedSummary', {
    running: runningJobsLabel.value,
    queued: queuedJobsLabel.value
  })
})

const {
  selectedJobTab,
  selectedWorkflowFilter,
  selectedSortMode,
  hasFailedJobs,
  filteredTasks,
  groupedJobItems,
  currentNodeName
} = useJobList()

const displayedJobGroups = computed(() => groupedJobItems.value)

const onCancelItem = wrapWithErrorHandlingAsync(async (item: JobListItem) => {
  const jobId = item.taskRef?.jobId
  if (!jobId) return

  if (item.state === 'running' || item.state === 'initialization') {
    // Running/initializing jobs: interrupt execution
    // Cloud backend uses deleteItem, local uses interrupt
    if (isCloud) {
      await api.deleteItem('queue', jobId)
    } else {
      await api.interrupt(jobId)
    }
    executionStore.clearInitializationByJobId(jobId)
    await queueStore.update()
  } else if (item.state === 'pending') {
    // Pending jobs: remove from queue
    await api.deleteItem('queue', jobId)
    await queueStore.update()
  }
})

const onDeleteItem = wrapWithErrorHandlingAsync(async (item: JobListItem) => {
  if (!item.taskRef) return
  await queueStore.delete(item.taskRef)
})

const {
  galleryActiveIndex,
  galleryItems,
  onViewItem: openResultGallery
} = useResultGallery(() => filteredTasks.value)

const setExpanded = (expanded: boolean) => {
  isExpanded.value = expanded
}

const viewAllJobs = () => {
  setExpanded(true)
}

const toggleAssetsSidebar = () => {
  sidebarTabStore.toggleSidebarTab('assets')
}

const openAssetsSidebar = () => {
  sidebarTabStore.activeSidebarTabId = 'assets'
}

const focusAssetInSidebar = async (item: JobListItem) => {
  const task = item.taskRef
  const jobId = task?.jobId
  const preview = task?.previewOutput
  if (!jobId || !preview) return

  const assetId = String(jobId)
  openAssetsSidebar()
  await nextTick()
  await assetsStore.updateHistory()
  const asset = assetsStore.historyAssets.find(
    (existingAsset) => existingAsset.id === assetId
  )
  if (!asset) {
    throw new Error('Asset not found in media assets panel')
  }
  assetSelectionStore.setSelection([assetId])
  assetSelectionStore.setLastSelectedAssetId(assetId)
}

const inspectJobAsset = wrapWithErrorHandlingAsync(
  async (item: JobListItem) => {
    await openResultGallery(item)
    await focusAssetInSidebar(item)
  }
)

const cancelQueuedWorkflows = wrapWithErrorHandlingAsync(async () => {
  // Capture pending jobIds before clearing
  const pendingJobIds = queueStore.pendingTasks
    .map((task) => task.jobId)
    .filter((id): id is string => typeof id === 'string' && id.length > 0)

  await commandStore.execute('Comfy.ClearPendingTasks')

  // Clear initialization state for removed jobs
  executionStore.clearInitializationByJobIds(pendingJobIds)
})

const interruptAll = wrapWithErrorHandlingAsync(async () => {
  const tasks = queueStore.runningTasks
  const jobIds = tasks
    .map((task) => task.jobId)
    .filter((id): id is string => typeof id === 'string' && id.length > 0)

  if (!jobIds.length) return

  // Cloud backend supports cancelling specific jobs via /queue delete,
  // while /interrupt always targets the "first" job. Use the targeted API
  // on cloud to ensure we cancel the workflow the user clicked.
  if (isCloud) {
    await Promise.all(jobIds.map((id) => api.deleteItem('queue', id)))
    executionStore.clearInitializationByJobIds(jobIds)
    await queueStore.update()
    return
  }

  await Promise.all(jobIds.map((id) => api.interrupt(id)))
  executionStore.clearInitializationByJobIds(jobIds)
  await queueStore.update()
})

const onClearHistoryFromMenu = () => {
  showQueueClearHistoryDialog()
}
</script>
