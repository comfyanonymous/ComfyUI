<template>
  <div class="flex flex-col gap-4 px-3 pb-4">
    <div
      v-for="group in displayedJobGroups"
      :key="group.key"
      class="flex flex-col gap-2"
    >
      <div class="text-[12px] leading-none text-text-secondary">
        {{ group.label }}
      </div>
      <QueueJobItem
        v-for="ji in group.items"
        :key="ji.id"
        :job-id="ji.id"
        :workflow-id="ji.taskRef?.workflowId"
        :state="ji.state"
        :title="ji.title"
        :right-text="ji.meta"
        :icon-name="ji.iconName"
        :icon-image-url="ji.iconImageUrl"
        :show-clear="ji.showClear"
        :show-menu="true"
        :progress-total-percent="ji.progressTotalPercent"
        :progress-current-percent="ji.progressCurrentPercent"
        :running-node-name="ji.runningNodeName"
        :active-details-id="activeDetailsId"
        @cancel="emitCancelItem(ji)"
        @delete="emitDeleteItem(ji)"
        @menu="(ev) => $emit('menu', ji, ev)"
        @view="$emit('viewItem', ji)"
        @details-enter="onDetailsEnter"
        @details-leave="onDetailsLeave"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'

import QueueJobItem from '@/components/queue/job/QueueJobItem.vue'
import type { JobGroup, JobListItem } from '@/composables/queue/useJobList'

const props = defineProps<{ displayedJobGroups: JobGroup[] }>()

const emit = defineEmits<{
  (e: 'cancelItem', item: JobListItem): void
  (e: 'deleteItem', item: JobListItem): void
  (e: 'menu', item: JobListItem, ev: MouseEvent): void
  (e: 'viewItem', item: JobListItem): void
}>()

const emitCancelItem = (item: JobListItem) => {
  emit('cancelItem', item)
}

const emitDeleteItem = (item: JobListItem) => {
  emit('deleteItem', item)
}

const activeDetailsId = ref<string | null>(null)
const hideTimer = ref<number | null>(null)
const showTimer = ref<number | null>(null)
const clearHideTimer = () => {
  if (hideTimer.value !== null) {
    clearTimeout(hideTimer.value)
    hideTimer.value = null
  }
}
const clearShowTimer = () => {
  if (showTimer.value !== null) {
    clearTimeout(showTimer.value)
    showTimer.value = null
  }
}
const onDetailsEnter = (jobId: string) => {
  clearHideTimer()
  clearShowTimer()
  showTimer.value = window.setTimeout(() => {
    activeDetailsId.value = jobId
    showTimer.value = null
  }, 200)
}
const onDetailsLeave = (jobId: string) => {
  clearHideTimer()
  clearShowTimer()
  hideTimer.value = window.setTimeout(() => {
    if (activeDetailsId.value === jobId) activeDetailsId.value = null
    hideTimer.value = null
  }, 150)
}

const resetActiveDetails = () => {
  clearHideTimer()
  clearShowTimer()
  activeDetailsId.value = null
}

watch(
  () => props.displayedJobGroups,
  (groups) => {
    const activeId = activeDetailsId.value
    if (!activeId) return

    const hasActiveJob = groups.some((group) =>
      group.items.some((item) => item.id === activeId)
    )

    if (!hasActiveJob) resetActiveDetails()
  }
)

onBeforeUnmount(resetActiveDetails)
</script>
