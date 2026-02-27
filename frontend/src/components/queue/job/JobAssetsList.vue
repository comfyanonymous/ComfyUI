<template>
  <div class="flex flex-col gap-4 px-3 pb-4">
    <div
      v-for="group in displayedJobGroups"
      :key="group.key"
      class="flex flex-col gap-2"
    >
      <div class="text-xs leading-none text-text-secondary">
        {{ group.label }}
      </div>
      <AssetsListItem
        v-for="job in group.items"
        :key="job.id"
        class="w-full shrink-0 cursor-default text-text-primary transition-colors hover:bg-secondary-background-hover"
        :preview-url="getJobPreviewUrl(job)"
        :is-video-preview="isVideoPreviewJob(job)"
        :preview-alt="job.title"
        :icon-name="job.iconName ?? iconForJobState(job.state)"
        :icon-class="getJobIconClass(job)"
        :primary-text="job.title"
        :secondary-text="job.meta"
        :progress-total-percent="job.progressTotalPercent"
        :progress-current-percent="job.progressCurrentPercent"
        @mouseenter="hoveredJobId = job.id"
        @mouseleave="onJobLeave(job.id)"
        @contextmenu.prevent.stop="$emit('menu', job, $event)"
        @dblclick.stop="emitViewItem(job)"
        @preview-click="emitViewItem(job)"
        @click.stop
      >
        <template v-if="hoveredJobId === job.id" #actions>
          <Button
            v-if="isCancelable(job)"
            variant="destructive"
            size="icon"
            :aria-label="t('g.cancel')"
            @click.stop="$emit('cancelItem', job)"
          >
            <i class="icon-[lucide--x] size-4" />
          </Button>
          <Button
            v-else-if="isFailedDeletable(job)"
            variant="destructive"
            size="icon"
            :aria-label="t('g.delete')"
            @click.stop="$emit('deleteItem', job)"
          >
            <i class="icon-[lucide--trash-2] size-4" />
          </Button>
          <Button
            v-else-if="job.state === 'completed'"
            variant="textonly"
            size="sm"
            @click.stop="$emit('viewItem', job)"
          >
            {{ t('menuLabels.View') }}
          </Button>
          <Button
            variant="secondary"
            size="icon"
            :aria-label="t('g.more')"
            @click.stop="$emit('menu', job, $event)"
          >
            <i class="icon-[lucide--ellipsis] size-4" />
          </Button>
        </template>
      </AssetsListItem>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import type { JobGroup, JobListItem } from '@/composables/queue/useJobList'
import AssetsListItem from '@/platform/assets/components/AssetsListItem.vue'
import { iconForJobState } from '@/utils/queueDisplay'
import { isActiveJobState } from '@/utils/queueUtil'

defineProps<{ displayedJobGroups: JobGroup[] }>()

const emit = defineEmits<{
  (e: 'cancelItem', item: JobListItem): void
  (e: 'deleteItem', item: JobListItem): void
  (e: 'menu', item: JobListItem, ev: MouseEvent): void
  (e: 'viewItem', item: JobListItem): void
}>()

const { t } = useI18n()
const hoveredJobId = ref<string | null>(null)

const onJobLeave = (jobId: string) => {
  if (hoveredJobId.value === jobId) {
    hoveredJobId.value = null
  }
}

const isCancelable = (job: JobListItem) =>
  job.showClear !== false && isActiveJobState(job.state)

const isFailedDeletable = (job: JobListItem) =>
  job.showClear !== false && job.state === 'failed'

const getPreviewOutput = (job: JobListItem) => job.taskRef?.previewOutput

const getJobPreviewUrl = (job: JobListItem) => {
  const preview = getPreviewOutput(job)
  if (preview?.isImage || preview?.isVideo) {
    return preview.url
  }
  return job.iconImageUrl
}

const isVideoPreviewJob = (job: JobListItem) =>
  job.state === 'completed' && !!getPreviewOutput(job)?.isVideo

const isPreviewableCompletedJob = (job: JobListItem) =>
  job.state === 'completed' && !!getPreviewOutput(job)

const emitViewItem = (job: JobListItem) => {
  if (isPreviewableCompletedJob(job)) {
    emit('viewItem', job)
  }
}

const getJobIconClass = (job: JobListItem): string | undefined => {
  const iconName = job.iconName ?? iconForJobState(job.state)
  if (!job.iconImageUrl && iconName === iconForJobState('pending')) {
    return 'animate-spin'
  }
  return undefined
}
</script>
