import { computed, ref } from 'vue'
import { describe, expect, it } from 'vitest'

import type { TaskItemImpl } from '@/stores/queueStore'
import type { JobState } from '@/types/queue'

import { formatElapsedTime, useQueueEstimates } from './useQueueEstimates'
import type { UseQueueEstimatesOptions } from './useQueueEstimates'

type QueueStore = UseQueueEstimatesOptions['queueStore']
type ExecutionStore = UseQueueEstimatesOptions['executionStore']

const makeHistoryTask = (
  executionTimeInSeconds: number | string | undefined
): TaskItemImpl =>
  ({
    executionTimeInSeconds
  }) as TaskItemImpl

const makeRunningTask = (executionStartTimestamp?: number): TaskItemImpl =>
  ({
    executionStartTimestamp
  }) as TaskItemImpl

const createQueueStore = (data?: Partial<QueueStore>): QueueStore =>
  ({
    historyTasks: [],
    runningTasks: [],
    ...data
  }) as QueueStore

const createExecutionStore = (data?: Partial<ExecutionStore>): ExecutionStore =>
  ({
    runningWorkflowCount: 1,
    ...data
  }) as ExecutionStore

type HarnessOptions = {
  queueStore?: QueueStore
  executionStore?: ExecutionStore
  task?: TaskItemImpl | null
  jobState?: JobState | null
  firstSeenTs?: number
  jobsAhead?: number | null
  now?: number
}

const createHarness = (options?: HarnessOptions) => {
  const queueStore = options?.queueStore ?? createQueueStore()
  const executionStore = options?.executionStore ?? createExecutionStore()
  const taskRef = ref<TaskItemImpl | null>(options?.task ?? null)
  const jobStateRef = ref<JobState | null>(options?.jobState ?? null)
  const firstSeenRef = ref<number | undefined>(options?.firstSeenTs)
  const jobsAheadRef = ref<number | null>(options?.jobsAhead ?? null)
  const nowRef = ref(options?.now ?? 0)

  const result = useQueueEstimates({
    queueStore,
    executionStore,
    taskForJob: computed(() => taskRef.value),
    jobState: computed(() => jobStateRef.value),
    firstSeenTs: computed(() => firstSeenRef.value),
    jobsAhead: computed(() => jobsAheadRef.value),
    nowTs: nowRef
  })

  return {
    ...result,
    queueStore,
    executionStore,
    taskRef,
    jobStateRef,
    firstSeenRef,
    jobsAheadRef,
    nowRef
  }
}

describe('formatElapsedTime', () => {
  it('formats elapsed milliseconds and clamps negatives to zero', () => {
    expect(formatElapsedTime(0)).toBe('0m 0s')
    expect(formatElapsedTime(61000)).toBe('1m 1s')
    expect(formatElapsedTime(90000)).toBe('1m 30s')
    expect(formatElapsedTime(-5000)).toBe('0m 0s')
  })
})

describe('useQueueEstimates', () => {
  it('only shows parallel queued stats for pending jobs seen with multiple runners', () => {
    const ready = createHarness({
      executionStore: createExecutionStore({ runningWorkflowCount: 2 }),
      jobState: 'pending',
      firstSeenTs: 1000
    })
    expect(ready.showParallelQueuedStats.value).toBe(true)

    const missingTimestamp = createHarness({
      executionStore: createExecutionStore({ runningWorkflowCount: 2 }),
      jobState: 'pending'
    })
    expect(missingTimestamp.showParallelQueuedStats.value).toBe(false)

    const singleRunner = createHarness({
      executionStore: createExecutionStore({ runningWorkflowCount: 1 }),
      jobState: 'pending',
      firstSeenTs: 1000
    })
    expect(singleRunner.showParallelQueuedStats.value).toBe(false)

    const runningJob = createHarness({
      executionStore: createExecutionStore({ runningWorkflowCount: 3 }),
      jobState: 'running',
      firstSeenTs: 1000
    })
    expect(runningJob.showParallelQueuedStats.value).toBe(false)
  })

  it('uses the last 20 valid durations to estimate queued batches', () => {
    const durations = Array.from({ length: 25 }, (_, idx) => idx + 1)
    const queueStore = createQueueStore({
      historyTasks: [
        ...durations.slice(0, 5).map((value) => makeHistoryTask(value)),
        makeHistoryTask('not-a-number'),
        makeHistoryTask(undefined),
        ...durations.slice(5).map((value) => makeHistoryTask(value))
      ]
    })

    const { estimateRangeSeconds } = createHarness({
      queueStore,
      executionStore: createExecutionStore({ runningWorkflowCount: 2 }),
      jobsAhead: 5
    })

    expect(estimateRangeSeconds.value).toEqual([47, 63])
  })

  it('returns null for estimateRangeSeconds when no history or jobsAhead is unknown', () => {
    const emptyHistory = createHarness({
      queueStore: createQueueStore(),
      jobsAhead: 2
    })
    expect(emptyHistory.estimateRangeSeconds.value).toBeNull()

    const missingAhead = createHarness({
      queueStore: createQueueStore({
        historyTasks: [makeHistoryTask(10)]
      })
    })
    expect(missingAhead.estimateRangeSeconds.value).toBeNull()
  })

  it('falls back to the running remaining range when there are no jobs ahead', () => {
    const now = 20000
    const queueStore = createQueueStore({
      historyTasks: [10, 20, 30].map((value) => makeHistoryTask(value)),
      runningTasks: [
        makeRunningTask(now - 5000),
        makeRunningTask(now - 15000),
        makeRunningTask(undefined)
      ]
    })

    const { estimateRangeSeconds } = createHarness({
      queueStore,
      jobsAhead: 0,
      now
    })

    expect(estimateRangeSeconds.value).toEqual([5, 15])
  })

  it('subtracts elapsed time when estimating a running job', () => {
    const now = 25000
    const queueStore = createQueueStore({
      historyTasks: [10, 20, 30].map((value) => makeHistoryTask(value))
    })

    const { estimateRemainingRangeSeconds } = createHarness({
      queueStore,
      task: makeRunningTask(5000),
      jobState: 'running',
      firstSeenTs: 2000,
      now
    })

    expect(estimateRemainingRangeSeconds.value).toEqual([0, 10])
  })

  it('uses the first-seen timestamp for pending jobs and clamps negatives to zero', () => {
    const queueStore = createQueueStore({
      historyTasks: [10, 20, 30].map((value) => makeHistoryTask(value))
    })

    const harness = createHarness({
      queueStore,
      jobState: 'pending',
      firstSeenTs: 10000,
      now: 25000
    })

    expect(harness.estimateRemainingRangeSeconds.value).toEqual([5, 15])

    harness.firstSeenRef.value = 1000
    harness.nowRef.value = 70000

    expect(harness.estimateRemainingRangeSeconds.value).toEqual([0, 0])
  })

  it('computes the elapsed label using execution start, then first-seen timestamp', () => {
    const harness = createHarness()

    harness.taskRef.value = makeRunningTask(1000)
    harness.jobStateRef.value = 'running'
    harness.nowRef.value = 4000

    expect(harness.timeElapsedValue.value).toBe('0m 3s')

    harness.jobStateRef.value = 'pending'
    harness.firstSeenRef.value = 2000
    harness.nowRef.value = 5000

    expect(harness.timeElapsedValue.value).toBe('0m 3s')

    harness.taskRef.value = null
    harness.firstSeenRef.value = undefined

    expect(harness.timeElapsedValue.value).toBe('')
  })
})
