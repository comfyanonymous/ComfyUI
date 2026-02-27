<template>
  <div
    role="status"
    tabindex="0"
    :aria-label="t('sideToolbar.activeJobStatus', { status: statusText })"
    class="flex flex-col gap-2 p-2 rounded-lg"
    @mouseenter="hovered = true"
    @mouseleave="hovered = false"
    @focusin="hovered = true"
    @focusout="hovered = false"
  >
    <!-- Thumbnail -->
    <div class="relative aspect-square overflow-hidden rounded-lg">
      <!-- Running state with preview image -->
      <img
        v-if="isRunning && job.iconImageUrl"
        :src="job.iconImageUrl"
        :alt="statusText"
        class="size-full object-cover"
      />
      <!-- Placeholder for queued/failed states or running without preview -->
      <div
        v-else
        class="absolute inset-0 flex items-center justify-center bg-modal-card-placeholder-background"
      >
        <!-- Spinner for queued/initialization states -->
        <i
          v-if="isQueued"
          class="icon-[lucide--loader-circle] size-8 animate-spin text-muted-foreground"
        />
        <!-- Error icon for failed state -->
        <i
          v-else-if="isFailed"
          class="icon-[lucide--circle-alert] size-8 text-red-500"
        />
        <!-- Spinner for running without preview -->
        <i
          v-else
          class="icon-[lucide--loader-circle] size-8 animate-spin text-muted-foreground"
        />
      </div>
      <!-- Cancel/Delete button overlay -->
      <Button
        v-if="hovered && showActionButton"
        variant="destructive"
        size="icon"
        :aria-label="activeAction.label"
        class="absolute top-2 right-2"
        @click.stop="runActiveAction()"
      >
        <i :class="activeAction.icon" />
      </Button>
    </div>

    <!-- Footer: Progress bar or status text -->
    <div class="flex gap-1.5 items-center h-5">
      <!-- Running state: percentage + progress bar -->
      <template v-if="isRunning && hasProgressPercent(progressPercent)">
        <span class="shrink-0 text-sm text-muted-foreground">
          {{ Math.round(progressPercent ?? 0) }}%
        </span>
        <div class="flex-1 relative h-1 rounded-sm bg-secondary-background">
          <div
            :class="progressBarPrimaryClass"
            :style="progressPercentStyle(progressPercent)"
          />
        </div>
      </template>
      <!-- Non-running states: status text only -->
      <template v-else>
        <div class="w-full truncate text-center text-sm text-muted-foreground">
          {{ statusText }}
        </div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useJobActions } from '@/composables/queue/useJobActions'
import type { JobListItem } from '@/composables/queue/useJobList'
import { useProgressBarBackground } from '@/composables/useProgressBarBackground'

const { job } = defineProps<{ job: JobListItem }>()

const { t } = useI18n()
const hovered = ref(false)

const {
  cancelAction,
  canCancelJob,
  runCancelJob,
  deleteAction,
  canDeleteJob,
  runDeleteJob
} = useJobActions(() => job)

const showActionButton = computed(
  () => canCancelJob.value || canDeleteJob.value
)
const activeAction = computed(() =>
  canCancelJob.value ? cancelAction : deleteAction
)
const runActiveAction = () => {
  if (canCancelJob.value) {
    runCancelJob()
  } else if (canDeleteJob.value) {
    runDeleteJob()
  }
}

const { progressBarPrimaryClass, hasProgressPercent, progressPercentStyle } =
  useProgressBarBackground()

const statusText = computed(() => job.title)
const progressPercent = computed(() => job.progressTotalPercent)

const isQueued = computed(
  () => job.state === 'pending' || job.state === 'initialization'
)
const isRunning = computed(() => job.state === 'running')
const isFailed = computed(() => job.state === 'failed')
</script>
