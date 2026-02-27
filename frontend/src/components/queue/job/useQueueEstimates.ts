import { computed } from 'vue'
import type { ComputedRef, Ref } from 'vue'

import type { useExecutionStore } from '@/stores/executionStore'
import type { TaskItemImpl, useQueueStore } from '@/stores/queueStore'
import type { JobState } from '@/types/queue'

type QueueStore = ReturnType<typeof useQueueStore>
type ExecutionStore = ReturnType<typeof useExecutionStore>

export type UseQueueEstimatesOptions = {
  queueStore: QueueStore
  executionStore: ExecutionStore
  taskForJob: ComputedRef<TaskItemImpl | null>
  jobState: ComputedRef<JobState | null>
  firstSeenTs: ComputedRef<number | undefined>
  jobsAhead: ComputedRef<number | null>
  nowTs: Ref<number>
}

type EstimateRange = [number, number]

export const formatElapsedTime = (ms: number): string => {
  const totalSec = Math.max(0, Math.floor(ms / 1000))
  const minutes = Math.floor(totalSec / 60)
  const seconds = totalSec % 60
  return `${minutes}m ${seconds}s`
}

const pickRecentDurations = (queueStore: QueueStore) =>
  queueStore.historyTasks
    .map((task: TaskItemImpl) => Number(task.executionTimeInSeconds))
    .filter(
      (value: number | undefined) =>
        typeof value === 'number' && !Number.isNaN(value)
    ) as number[]

export const useQueueEstimates = ({
  queueStore,
  executionStore,
  taskForJob,
  jobState,
  firstSeenTs,
  jobsAhead,
  nowTs
}: UseQueueEstimatesOptions) => {
  const runningWorkflowCount = computed(
    () => executionStore.runningWorkflowCount
  )

  const showParallelQueuedStats = computed(
    () =>
      jobState.value === 'pending' &&
      !!firstSeenTs.value &&
      (runningWorkflowCount.value ?? 0) > 1
  )

  const recentDurations = computed<number[]>(() =>
    pickRecentDurations(queueStore).slice(-20)
  )

  const runningRemainingRangeSeconds = computed<EstimateRange | null>(() => {
    const durations = recentDurations.value
    if (!durations.length) return null
    const sorted = durations.slice().sort((a, b) => a - b)
    const avg = sorted.reduce((sum, value) => sum + value, 0) / sorted.length
    const p75 =
      sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.75))]
    const running = queueStore.runningTasks as TaskItemImpl[]
    const now = nowTs.value
    const remaining = running
      .map((task) => task.executionStartTimestamp)
      .filter((timestamp): timestamp is number => typeof timestamp === 'number')
      .map((startTs) => {
        const elapsed = Math.max(0, Math.floor((now - startTs) / 1000))
        return {
          lo: Math.max(0, Math.round(avg - elapsed)),
          hi: Math.max(0, Math.round(p75 - elapsed))
        }
      })
    if (!remaining.length) return null
    const minLo = remaining.reduce(
      (min, range) => Math.min(min, range.lo),
      Infinity
    )
    const minHi = remaining.reduce(
      (min, range) => Math.min(min, range.hi),
      Infinity
    )
    return [minLo, minHi]
  })

  const estimateRangeSeconds = computed<EstimateRange | null>(() => {
    const durations = recentDurations.value
    if (!durations.length) return null
    const ahead = jobsAhead.value
    if (ahead == null) return null
    const sorted = durations.slice().sort((a, b) => a - b)
    const avg = sorted.reduce((sum, value) => sum + value, 0) / sorted.length
    const p75 =
      sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.75))]
    if (ahead <= 0) {
      return runningRemainingRangeSeconds.value ?? [0, 0]
    }
    const runningCount = Math.max(1, runningWorkflowCount.value || 1)
    const batches = Math.ceil(ahead / runningCount)
    return [Math.round(avg * batches), Math.round(p75 * batches)]
  })

  const estimateRemainingRangeSeconds = computed<EstimateRange | null>(() => {
    const durations = recentDurations.value
    if (!durations.length) return null
    const sorted = durations.slice().sort((a, b) => a - b)
    const avg = sorted.reduce((sum, value) => sum + value, 0) / sorted.length
    const p75 =
      sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.75))]
    const task = taskForJob.value as TaskItemImpl & {
      executionStartTimestamp?: number
    }
    const execStart =
      jobState.value === 'running' ? task?.executionStartTimestamp : undefined
    const baseTs = execStart ?? firstSeenTs.value
    const elapsed = baseTs
      ? Math.max(0, Math.floor((nowTs.value - baseTs) / 1000))
      : 0
    const lo = Math.max(0, Math.round(avg - elapsed))
    const hi = Math.max(0, Math.round(p75 - elapsed))
    return [lo, hi]
  })

  const timeElapsedValue = computed(() => {
    const task = taskForJob.value as TaskItemImpl & {
      executionStartTimestamp?: number
    }
    const execStart =
      jobState.value === 'running' ? task?.executionStartTimestamp : undefined
    const baseTs = execStart ?? firstSeenTs.value
    if (!baseTs) return ''
    return formatElapsedTime(nowTs.value - baseTs)
  })

  return {
    runningWorkflowCount,
    showParallelQueuedStats,
    estimateRangeSeconds,
    estimateRemainingRangeSeconds,
    timeElapsedValue
  }
}
