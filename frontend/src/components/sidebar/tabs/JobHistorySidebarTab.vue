<template>
  <SidebarTabTemplate :title="$t('queue.jobHistory')">
    <template #alt-title>
      <div class="ml-auto flex shrink-0 items-center">
        <JobHistoryActionsMenu @clear-history="showQueueClearHistoryDialog" />
      </div>
    </template>
    <template #header>
      <div class="flex flex-col gap-2 pb-1">
        <div class="px-3 py-2">
          <JobFilterTabs
            v-model:selected-job-tab="selectedJobTab"
            :has-failed-jobs="hasFailedJobs"
          />
        </div>
        <JobFilterActions
          v-model:selected-workflow-filter="selectedWorkflowFilter"
          v-model:selected-sort-mode="selectedSortMode"
          v-model:search-query="searchQuery"
          class="px-3"
          :hide-show-assets-action="true"
          :show-search="true"
          :search-placeholder="t('sideToolbar.queueProgressOverlay.searchJobs')"
        />
      </div>
      <div
        class="flex items-center justify-between px-3 pb-1 text-xs leading-none text-text-primary"
      >
        <span class="text-text-secondary">{{ activeQueueSummary }}</span>
        <div class="flex items-center gap-2">
          <span class="text-xs text-base-foreground">
            {{ t('sideToolbar.queueProgressOverlay.clearQueueTooltip') }}
          </span>
          <Button
            variant="destructive"
            size="icon"
            :aria-label="
              t('sideToolbar.queueProgressOverlay.clearQueueTooltip')
            "
            :disabled="queuedCount === 0"
            @click="clearQueuedWorkflows"
          >
            <i class="icon-[lucide--list-x] size-4" />
          </Button>
        </div>
      </div>
    </template>
    <template #body>
      <JobAssetsList
        :displayed-job-groups="displayedJobGroups"
        @cancel-item="onCancelItem"
        @delete-item="onDeleteItem"
        @view-item="onViewItem"
        @menu="onMenuItem"
      />
      <JobContextMenu
        ref="jobContextMenuRef"
        :entries="jobMenuEntries"
        @action="onJobMenuAction"
      />
      <ResultGallery
        v-model:active-index="galleryActiveIndex"
        :all-gallery-items="galleryItems"
      />
    </template>
  </SidebarTabTemplate>
</template>

<script setup lang="ts">
import { computed, defineAsyncComponent, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import JobFilterActions from '@/components/queue/job/JobFilterActions.vue'
import JobFilterTabs from '@/components/queue/job/JobFilterTabs.vue'
import JobAssetsList from '@/components/queue/job/JobAssetsList.vue'
import JobContextMenu from '@/components/queue/job/JobContextMenu.vue'
import JobHistoryActionsMenu from '@/components/queue/JobHistoryActionsMenu.vue'
import type { MenuEntry } from '@/composables/queue/useJobMenu'
import { useJobMenu } from '@/composables/queue/useJobMenu'
import { useJobList } from '@/composables/queue/useJobList'
import type { JobListItem } from '@/composables/queue/useJobList'
import { useQueueClearHistoryDialog } from '@/composables/queue/useQueueClearHistoryDialog'
import { useResultGallery } from '@/composables/queue/useResultGallery'
import { useErrorHandling } from '@/composables/useErrorHandling'
import SidebarTabTemplate from '@/components/sidebar/tabs/SidebarTabTemplate.vue'
import ResultGallery from '@/components/sidebar/tabs/queue/ResultGallery.vue'
import Button from '@/components/ui/button/Button.vue'
import { useCommandStore } from '@/stores/commandStore'
import { useDialogStore } from '@/stores/dialogStore'
import { useExecutionStore } from '@/stores/executionStore'
import { useQueueStore } from '@/stores/queueStore'

const Load3dViewerContent = defineAsyncComponent(
  () => import('@/components/load3d/Load3dViewerContent.vue')
)

const { t, n } = useI18n()
const commandStore = useCommandStore()
const dialogStore = useDialogStore()
const executionStore = useExecutionStore()
const queueStore = useQueueStore()
const { showQueueClearHistoryDialog } = useQueueClearHistoryDialog()
const { wrapWithErrorHandlingAsync } = useErrorHandling()
const {
  selectedJobTab,
  selectedWorkflowFilter,
  selectedSortMode,
  searchQuery,
  hasFailedJobs,
  filteredTasks,
  groupedJobItems
} = useJobList()

const displayedJobGroups = computed(() => groupedJobItems.value)
const runningCount = computed(() => queueStore.runningTasks.length)
const queuedCount = computed(() => queueStore.pendingTasks.length)

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
const activeQueueSummary = computed(() => {
  if (runningCount.value === 0 && queuedCount.value === 0) {
    return t('sideToolbar.queueProgressOverlay.noActiveJobs')
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

const clearQueuedWorkflows = wrapWithErrorHandlingAsync(async () => {
  const pendingJobIds = queueStore.pendingTasks
    .map((task) => task.jobId)
    .filter((id): id is string => typeof id === 'string' && id.length > 0)

  await commandStore.execute('Comfy.ClearPendingTasks')
  executionStore.clearInitializationByJobIds(pendingJobIds)
})

const {
  galleryActiveIndex,
  galleryItems,
  onViewItem: openResultGallery
} = useResultGallery(() => filteredTasks.value)

const onViewItem = wrapWithErrorHandlingAsync(async (item: JobListItem) => {
  const previewOutput = item.taskRef?.previewOutput

  if (previewOutput?.is3D) {
    dialogStore.showDialog({
      key: 'asset-3d-viewer',
      title: item.title,
      component: Load3dViewerContent,
      props: {
        modelUrl: previewOutput.url || ''
      },
      dialogComponentProps: {
        style: 'width: 80vw; height: 80vh;',
        maximizable: true
      }
    })
    return
  }

  await openResultGallery(item)
})

const onInspectAsset = (item: JobListItem) => {
  void onViewItem(item)
}

const currentMenuItem = ref<JobListItem | null>(null)
const jobContextMenuRef = ref<InstanceType<typeof JobContextMenu> | null>(null)

const { jobMenuEntries, cancelJob } = useJobMenu(
  () => currentMenuItem.value,
  onInspectAsset
)

const onCancelItem = wrapWithErrorHandlingAsync(async (item: JobListItem) => {
  await cancelJob(item)
})

const onDeleteItem = wrapWithErrorHandlingAsync(async (item: JobListItem) => {
  if (!item.taskRef) return
  await queueStore.delete(item.taskRef)
})

const onMenuItem = (item: JobListItem, event: Event) => {
  currentMenuItem.value = item
  jobContextMenuRef.value?.open(event)
}

const onJobMenuAction = wrapWithErrorHandlingAsync(async (entry: MenuEntry) => {
  if (entry.kind === 'divider') return
  if (entry.onClick) await entry.onClick()
  jobContextMenuRef.value?.hide()
})
</script>
