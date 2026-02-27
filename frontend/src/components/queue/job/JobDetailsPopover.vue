<template>
  <div
    class="w-[300px] min-w-[260px] rounded-lg border border-interface-stroke bg-interface-panel-surface shadow-md"
  >
    <div class="flex items-center border-b border-interface-stroke p-4">
      <span
        class="text-[0.875rem] leading-normal font-normal text-text-primary"
        >{{ t('queue.jobDetails.header') }}</span
      >
    </div>
    <div class="flex flex-col gap-6 px-4 pt-4 pb-4">
      <div class="grid grid-cols-2 items-center gap-x-2 gap-y-2">
        <template v-for="row in baseRows" :key="row.label">
          <div
            class="flex items-center text-[0.75rem] leading-normal font-normal text-text-primary"
          >
            {{ row.label }}
          </div>
          <div
            class="flex min-w-0 items-center text-[0.75rem] leading-normal font-normal text-text-secondary"
          >
            <span class="block min-w-0 truncate">{{ row.value }}</span>
            <Button
              v-if="row.canCopy"
              size="icon"
              variant="muted-textonly"
              :aria-label="copyAriaLabel"
              @click.stop="copyJobId"
            >
              <i class="icon-[lucide--copy] size-4" />
            </Button>
          </div>
        </template>
      </div>

      <div
        v-if="extraRows.length"
        class="grid grid-cols-2 items-center gap-x-2 gap-y-2"
      >
        <template v-for="row in extraRows" :key="row.label">
          <div
            class="flex items-center text-[0.75rem] leading-normal font-normal text-text-primary"
          >
            {{ row.label }}
          </div>
          <div
            class="flex min-w-0 items-center text-[0.75rem] leading-normal font-normal text-text-secondary"
          >
            <span class="block min-w-0 truncate">{{ row.value }}</span>
          </div>
        </template>
      </div>

      <div v-if="jobState === 'failed'" class="grid grid-cols-2 gap-x-2">
        <div
          class="flex items-center text-[0.75rem] leading-normal font-normal text-text-primary"
        >
          {{ t('queue.jobDetails.errorMessage') }}
        </div>
        <div class="flex items-center justify-between gap-4">
          <Button
            class="justify-start px-0"
            variant="muted-textonly"
            size="sm"
            icon-position="right"
            @click.stop="copyErrorMessage"
          >
            <span>{{ copyAriaLabel }}</span>
            <i class="icon-[lucide--copy] block size-3.5 leading-none" />
          </Button>
          <Button
            class="justify-start px-0"
            variant="muted-textonly"
            size="sm"
            icon-position="right"
            @click.stop="reportJobError"
          >
            <span>{{ t('queue.jobDetails.report') }}</span>
            <i
              class="icon-[lucide--message-circle-warning] block size-3.5 leading-none"
            />
          </Button>
        </div>
        <div
          class="col-span-2 mt-2 rounded bg-interface-panel-hover-surface px-4 py-2 text-[0.75rem] leading-normal text-text-secondary"
        >
          {{ errorMessageValue }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useCopyToClipboard } from '@/composables/useCopyToClipboard'
import { isCloud } from '@/platform/distribution/types'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useDialogService } from '@/services/dialogService'
import { useExecutionStore } from '@/stores/executionStore'
import { useQueueStore } from '@/stores/queueStore'
import type { TaskItemImpl } from '@/stores/queueStore'
import { formatClockTime } from '@/utils/dateTimeUtil'
import { jobStateFromTask } from '@/utils/queueUtil'

import { useJobErrorReporting } from './useJobErrorReporting'
import { formatElapsedTime, useQueueEstimates } from './useQueueEstimates'

const props = defineProps<{
  jobId: string
  workflowId?: string
}>()

const copyAriaLabel = computed(() => t('g.copy'))

const workflowStore = useWorkflowStore()
const queueStore = useQueueStore()
const executionStore = useExecutionStore()
const dialog = useDialogService()
const { locale, t } = useI18n()

const workflowValue = computed(() => {
  const wid = props.workflowId
  if (!wid) return ''
  const activeId = workflowStore.activeWorkflow?.activeState?.id
  if (activeId && activeId === wid) {
    return workflowStore.activeWorkflow?.filename ?? wid
  }
  return wid
})
const jobIdValue = computed(() => props.jobId)

const { copyToClipboard } = useCopyToClipboard()
const copyJobId = () => void copyToClipboard(jobIdValue.value)

const taskForJob = computed(() => {
  const pid = props.jobId
  const findIn = (arr: TaskItemImpl[]) =>
    arr.find((t) => String(t.jobId ?? '') === String(pid))
  return (
    findIn(queueStore.pendingTasks) ||
    findIn(queueStore.runningTasks) ||
    findIn(queueStore.historyTasks) ||
    null
  )
})

const jobState = computed(() => {
  const task = taskForJob.value
  if (!task) return null
  const isInitializing = executionStore.isJobInitializing(String(task?.jobId))
  return jobStateFromTask(task, isInitializing)
})

const firstSeenTs = computed<number | undefined>(() => {
  const task = taskForJob.value
  return task?.createTime
})

const queuedAtValue = computed(() =>
  firstSeenTs.value !== undefined
    ? formatClockTime(firstSeenTs.value, locale.value)
    : ''
)

const currentJobPriority = computed<number | null>(() => {
  const task = taskForJob.value
  return task ? Number(task.job.priority) : null
})

const jobsAhead = computed<number | null>(() => {
  const idx = currentJobPriority.value
  if (idx == null) return null
  const ahead = queueStore.pendingTasks.filter(
    (t: TaskItemImpl) => Number(t.job.priority) < idx
  )
  return ahead.length
})

const queuePositionValue = computed(() => {
  if (jobsAhead.value == null) return ''
  const n = jobsAhead.value
  return t('queue.jobDetails.queuePositionValue', { count: n }, n)
})

const nowTs = ref<number>(Date.now())
let timer: number | null = null
onMounted(() => {
  timer = window.setInterval(() => {
    nowTs.value = Date.now()
  }, 1000)
})
onUnmounted(() => {
  if (timer != null) {
    clearInterval(timer)
    timer = null
  }
})

const {
  showParallelQueuedStats,
  estimateRangeSeconds,
  estimateRemainingRangeSeconds,
  timeElapsedValue
} = useQueueEstimates({
  queueStore,
  executionStore,
  taskForJob,
  jobState,
  firstSeenTs,
  jobsAhead,
  nowTs
})

const formatEta = (lo: number, hi: number): string => {
  if (hi <= 60) {
    const hiS = Math.max(1, Math.round(hi))
    const loS = Math.max(1, Math.min(hiS, Math.round(lo)))
    if (loS === hiS)
      return t('queue.jobDetails.eta.seconds', { count: hiS }, hiS)
    return t('queue.jobDetails.eta.secondsRange', { lo: loS, hi: hiS })
  }
  if (lo >= 60 && hi < 90) {
    return t('queue.jobDetails.eta.minutes', { count: 1 }, 1)
  }
  const loM = Math.max(1, Math.floor(lo / 60))
  const hiM = Math.max(loM, Math.ceil(hi / 60))
  if (loM === hiM) {
    return t('queue.jobDetails.eta.minutes', { count: loM }, loM)
  }
  return t('queue.jobDetails.eta.minutesRange', { lo: loM, hi: hiM })
}

const estimatedStartInValue = computed(() => {
  const range = estimateRangeSeconds.value
  if (!range) return ''
  const [lo, hi] = range
  return formatEta(lo, hi)
})

const estimatedFinishInValue = computed(() => {
  const range = estimateRemainingRangeSeconds.value
  if (!range) return ''
  const [lo, hi] = range
  return formatEta(lo, hi)
})

type DetailRow = { label: string; value: string; canCopy?: boolean }

const baseRows = computed<DetailRow[]>(() => [
  { label: t('queue.jobDetails.workflow'), value: workflowValue.value },
  { label: t('queue.jobDetails.jobId'), value: jobIdValue.value, canCopy: true }
])

const extraRows = computed<DetailRow[]>(() => {
  if (jobState.value === 'pending') {
    if (!firstSeenTs.value) return []
    const rows: DetailRow[] = [
      { label: t('queue.jobDetails.queuedAt'), value: queuedAtValue.value }
    ]
    if (showParallelQueuedStats.value) {
      rows.push(
        {
          label: t('queue.jobDetails.queuePosition'),
          value: queuePositionValue.value
        },
        {
          label: t('queue.jobDetails.timeElapsed'),
          value: timeElapsedValue.value
        },
        {
          label: t('queue.jobDetails.estimatedStartIn'),
          value: estimatedStartInValue.value
        }
      )
    }
    return rows
  }
  if (jobState.value === 'running') {
    if (!firstSeenTs.value) return []
    return [
      { label: t('queue.jobDetails.queuedAt'), value: queuedAtValue.value },
      {
        label: t('queue.jobDetails.timeElapsed'),
        value: timeElapsedValue.value
      },
      {
        label: t('queue.jobDetails.estimatedFinishIn'),
        value: estimatedFinishInValue.value
      }
    ]
  }
  if (jobState.value === 'completed') {
    const task = taskForJob.value
    const endTs: number | undefined = task?.executionEndTimestamp
    const execMs: number | undefined = task?.executionTime
    const generatedOnValue = endTs ? formatClockTime(endTs, locale.value) : ''
    const totalGenTimeValue =
      execMs !== undefined ? formatElapsedTime(execMs) : ''
    const computeHoursValue =
      execMs !== undefined ? (execMs / 3600000).toFixed(3) + ' hours' : ''

    const rows: DetailRow[] = [
      { label: t('queue.jobDetails.generatedOn'), value: generatedOnValue },
      {
        label: t('queue.jobDetails.totalGenerationTime'),
        value: totalGenTimeValue
      }
    ]
    if (isCloud) {
      rows.push({
        label: t('queue.jobDetails.computeHoursUsed'),
        value: computeHoursValue
      })
    }
    return rows
  }
  if (jobState.value === 'failed') {
    const task = taskForJob.value
    const execMs: number | undefined = task?.executionTime
    const failedAfterValue =
      execMs !== undefined ? formatElapsedTime(execMs) : ''
    const computeHoursValue =
      execMs !== undefined ? (execMs / 3600000).toFixed(3) + ' hours' : ''
    const rows: DetailRow[] = [
      { label: t('queue.jobDetails.queuedAt'), value: queuedAtValue.value },
      { label: t('queue.jobDetails.failedAfter'), value: failedAfterValue }
    ]
    if (isCloud) {
      rows.push({
        label: t('queue.jobDetails.computeHoursUsed'),
        value: computeHoursValue
      })
    }
    return rows
  }
  return []
})

const { errorMessageValue, copyErrorMessage, reportJobError } =
  useJobErrorReporting({
    taskForJob,
    copyToClipboard,
    dialog
  })
</script>
