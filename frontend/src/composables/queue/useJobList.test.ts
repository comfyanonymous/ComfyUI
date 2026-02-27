import { mount } from '@vue/test-utils'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { nextTick, reactive, ref } from 'vue'
import type { Ref } from 'vue'

import { useJobList } from '@/composables/queue/useJobList'
import type { JobState } from '@/types/queue'
import type { BuildJobDisplayCtx } from '@/utils/queueDisplay'
import { buildJobDisplay } from '@/utils/queueDisplay'
import type { TaskItemImpl } from '@/stores/queueStore'

type TestTask = {
  jobId: string
  job: { priority: number }
  mockState: JobState
  executionTime?: number
  executionEndTimestamp?: number
  createTime?: number
  workflowId?: string
}

const translations: Record<string, string> = {
  'queue.jobList.undated': 'Undated',
  'g.emDash': '--',
  'g.untitled': 'Untitled'
}
let localeRef: Ref<string>
let tMock: ReturnType<typeof vi.fn>
const ensureLocaleMocks = () => {
  if (!localeRef) {
    localeRef = ref('en-US') as Ref<string>
  }
  if (!tMock) {
    tMock = vi.fn((key: string) => translations[key] ?? key)
  }
  return { localeRef, tMock }
}

vi.mock('vue-i18n', () => ({
  useI18n: () => {
    ensureLocaleMocks()
    return {
      t: tMock,
      locale: localeRef
    }
  }
}))

vi.mock('@/i18n', () => ({
  st: vi.fn((key: string, fallback?: string) => `i18n(${key})-${fallback}`)
}))

let totalPercent: Ref<number>
let currentNodePercent: Ref<number>
const ensureProgressRefs = () => {
  if (!totalPercent) totalPercent = ref(0) as Ref<number>
  if (!currentNodePercent) currentNodePercent = ref(0) as Ref<number>
  return { totalPercent, currentNodePercent }
}
vi.mock('@/composables/queue/useQueueProgress', () => ({
  useQueueProgress: () => {
    ensureProgressRefs()
    return {
      totalPercent,
      currentNodePercent
    }
  }
}))

vi.mock('@/utils/queueDisplay', () => ({
  buildJobDisplay: vi.fn(
    (task: TaskItemImpl, state: JobState, options: BuildJobDisplayCtx) => ({
      primary: `Job ${task.jobId}`,
      secondary: `${state} meta`,
      iconName: `${state}-icon`,
      iconImageUrl: undefined,
      showClear: state === 'failed',
      options
    })
  )
}))

vi.mock('@/utils/queueUtil', () => ({
  jobStateFromTask: vi.fn(
    (task: TestTask, isInitializing?: boolean): JobState =>
      task.mockState ?? (isInitializing ? 'running' : 'completed')
  )
}))

let queueStoreMock: {
  pendingTasks: TestTask[]
  runningTasks: TestTask[]
  historyTasks: TestTask[]
}
const ensureQueueStore = () => {
  if (!queueStoreMock) {
    queueStoreMock = reactive({
      pendingTasks: [] as TestTask[],
      runningTasks: [] as TestTask[],
      historyTasks: [] as TestTask[]
    })
  }
  return queueStoreMock
}
vi.mock('@/stores/queueStore', () => ({
  useQueueStore: () => {
    return ensureQueueStore()
  }
}))

let executionStoreMock: {
  activeJobId: string | null
  executingNode: null | { title?: string; type?: string }
  isJobInitializing: (jobId?: string | number) => boolean
}
let isJobInitializingMock: (jobId?: string | number) => boolean
const ensureExecutionStore = () => {
  if (!isJobInitializingMock) {
    isJobInitializingMock = vi.fn(() => false)
  }
  if (!executionStoreMock) {
    executionStoreMock = reactive({
      activeJobId: null as string | null,
      executingNode: null as null | { title?: string; type?: string },
      isJobInitializing: (jobId?: string | number) =>
        isJobInitializingMock(jobId)
    })
  }
  return executionStoreMock
}
vi.mock('@/stores/executionStore', () => ({
  useExecutionStore: () => {
    return ensureExecutionStore()
  }
}))

let jobPreviewStoreMock: {
  previewsByPromptId: Record<string, string>
  isPreviewEnabled: boolean
}
const ensureJobPreviewStore = () => {
  if (!jobPreviewStoreMock) {
    jobPreviewStoreMock = reactive({
      previewsByPromptId: {} as Record<string, string>,
      isPreviewEnabled: true
    })
  }
  return jobPreviewStoreMock
}
vi.mock('@/stores/jobPreviewStore', () => ({
  useJobPreviewStore: () => {
    return ensureJobPreviewStore()
  }
}))

let workflowStoreMock: {
  activeWorkflow: null | { activeState?: { id?: string } }
}
const ensureWorkflowStore = () => {
  if (!workflowStoreMock) {
    workflowStoreMock = reactive({
      activeWorkflow: null as null | { activeState?: { id?: string } }
    })
  }
  return workflowStoreMock
}
vi.mock('@/platform/workflow/management/stores/workflowStore', () => ({
  useWorkflowStore: () => {
    return ensureWorkflowStore()
  }
}))

const createTask = (
  overrides: Partial<TestTask> & { mockState?: JobState } = {}
): TestTask => ({
  jobId: overrides.jobId ?? `task-${Math.random().toString(36).slice(2, 7)}`,
  job: overrides.job ?? { priority: 0 },
  mockState: overrides.mockState ?? 'pending',
  executionTime: overrides.executionTime,
  executionEndTimestamp: overrides.executionEndTimestamp,
  createTime: overrides.createTime,
  workflowId: overrides.workflowId
})

const mountUseJobList = () => {
  let composable: ReturnType<typeof useJobList>
  const wrapper = mount({
    template: '<div />',
    setup() {
      composable = useJobList()
      return {}
    }
  })
  return { wrapper, composable: composable! }
}

const resetStores = () => {
  const queueStore = ensureQueueStore()
  queueStore.pendingTasks = []
  queueStore.runningTasks = []
  queueStore.historyTasks = []

  const executionStore = ensureExecutionStore()
  executionStore.activeJobId = null
  executionStore.executingNode = null

  const jobPreviewStore = ensureJobPreviewStore()
  jobPreviewStore.previewsByPromptId = {}
  jobPreviewStore.isPreviewEnabled = true

  const workflowStore = ensureWorkflowStore()
  workflowStore.activeWorkflow = null

  ensureProgressRefs()
  totalPercent.value = 0
  currentNodePercent.value = 0

  ensureLocaleMocks()
  localeRef.value = 'en-US'
  tMock.mockClear()

  if (isJobInitializingMock) {
    vi.mocked(isJobInitializingMock).mockReset()
    vi.mocked(isJobInitializingMock).mockReturnValue(false)
  }
}

const flush = async () => {
  await nextTick()
}

describe('useJobList', () => {
  let wrapper: ReturnType<typeof mount> | null = null
  let api: ReturnType<typeof useJobList> | null = null

  beforeEach(() => {
    vi.resetAllMocks()
    resetStores()
    wrapper?.unmount()
    wrapper = null
    api = null
  })

  afterEach(() => {
    wrapper?.unmount()
    wrapper = null
    api = null
    vi.useRealTimers()
  })

  const initComposable = () => {
    const mounted = mountUseJobList()
    wrapper = mounted.wrapper
    api = mounted.composable
    return api!
  }

  it('tracks recently added pending jobs and clears the hint after expiry', async () => {
    vi.useFakeTimers()
    queueStoreMock.pendingTasks = [
      createTask({ jobId: '1', job: { priority: 1 }, mockState: 'pending' })
    ]

    const { jobItems } = initComposable()
    await flush()

    jobItems.value
    expect(buildJobDisplay).toHaveBeenCalledWith(
      expect.anything(),
      'pending',
      expect.objectContaining({ showAddedHint: true })
    )

    vi.mocked(buildJobDisplay).mockClear()
    await vi.advanceTimersByTimeAsync(3000)
    await flush()

    jobItems.value
    expect(buildJobDisplay).toHaveBeenCalledWith(
      expect.anything(),
      'pending',
      expect.objectContaining({ showAddedHint: false })
    )
  })

  it('removes pending hint immediately when the task leaves the queue', async () => {
    vi.useFakeTimers()
    const taskId = '2'
    queueStoreMock.pendingTasks = [
      createTask({ jobId: taskId, job: { priority: 1 }, mockState: 'pending' })
    ]

    const { jobItems } = initComposable()
    await flush()
    jobItems.value

    queueStoreMock.pendingTasks = []
    await flush()
    expect(vi.getTimerCount()).toBe(0)

    vi.mocked(buildJobDisplay).mockClear()
    queueStoreMock.pendingTasks = [
      createTask({ jobId: taskId, job: { priority: 2 }, mockState: 'pending' })
    ]
    await flush()
    jobItems.value
    expect(buildJobDisplay).toHaveBeenCalledWith(
      expect.anything(),
      'pending',
      expect.objectContaining({ showAddedHint: true })
    )
  })

  it('cleans up timeouts on unmount', async () => {
    vi.useFakeTimers()
    queueStoreMock.pendingTasks = [
      createTask({ jobId: '3', job: { priority: 1 }, mockState: 'pending' })
    ]

    initComposable()
    await flush()
    expect(vi.getTimerCount()).toBeGreaterThan(0)

    wrapper?.unmount()
    wrapper = null
    await flush()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('sorts all tasks by create time', async () => {
    queueStoreMock.pendingTasks = [
      createTask({
        jobId: 'p',
        job: { priority: 1 },
        mockState: 'pending',
        createTime: 3000
      })
    ]
    queueStoreMock.runningTasks = [
      createTask({
        jobId: 'r',
        job: { priority: 5 },
        mockState: 'running',
        createTime: 2000
      })
    ]
    queueStoreMock.historyTasks = [
      createTask({
        jobId: 'h',
        job: { priority: 3 },
        mockState: 'completed',
        createTime: 1000,
        executionEndTimestamp: 5000
      })
    ]

    const { allTasksSorted } = initComposable()
    await flush()

    expect(allTasksSorted.value.map((task) => task.jobId)).toEqual([
      'p',
      'r',
      'h'
    ])
  })

  it('filters by job tab and resets failed tab when failures disappear', async () => {
    queueStoreMock.historyTasks = [
      createTask({ jobId: 'c', job: { priority: 3 }, mockState: 'completed' }),
      createTask({ jobId: 'f', job: { priority: 2 }, mockState: 'failed' }),
      createTask({ jobId: 'p', job: { priority: 1 }, mockState: 'pending' })
    ]

    const instance = initComposable()
    await flush()

    instance.selectedJobTab.value = 'Completed'
    await flush()
    expect(instance.filteredTasks.value.map((t) => t.jobId)).toEqual(['c'])

    instance.selectedJobTab.value = 'Failed'
    await flush()
    expect(instance.filteredTasks.value.map((t) => t.jobId)).toEqual(['f'])
    expect(instance.hasFailedJobs.value).toBe(true)

    queueStoreMock.historyTasks = [
      createTask({ jobId: 'c', job: { priority: 3 }, mockState: 'completed' })
    ]
    await flush()

    expect(instance.hasFailedJobs.value).toBe(false)
    expect(instance.selectedJobTab.value).toBe('All')
  })

  it('filters by active workflow when requested', async () => {
    queueStoreMock.pendingTasks = [
      createTask({
        jobId: 'wf-1',
        job: { priority: 2 },
        mockState: 'pending',
        workflowId: 'workflow-1'
      }),
      createTask({
        jobId: 'wf-2',
        job: { priority: 1 },
        mockState: 'pending',
        workflowId: 'workflow-2'
      })
    ]

    const instance = initComposable()
    await flush()

    instance.selectedWorkflowFilter.value = 'current'
    await flush()
    expect(instance.filteredTasks.value).toEqual([])

    workflowStoreMock.activeWorkflow = { activeState: { id: 'workflow-1' } }
    await flush()

    expect(instance.filteredTasks.value.map((t) => t.jobId)).toEqual(['wf-1'])
  })

  it('filters jobs by search query', async () => {
    vi.useFakeTimers()
    queueStoreMock.historyTasks = [
      createTask({
        jobId: 'alpha',
        job: { priority: 2 },
        mockState: 'completed',
        createTime: 2000,
        executionEndTimestamp: 2000
      }),
      createTask({
        jobId: 'beta',
        job: { priority: 1 },
        mockState: 'failed',
        createTime: 1000,
        executionEndTimestamp: 1000
      })
    ]

    const instance = initComposable()
    await flush()
    expect(instance.filteredTasks.value.map((task) => task.jobId)).toEqual([
      'alpha',
      'beta'
    ])

    instance.searchQuery.value = 'beta'
    await vi.advanceTimersByTimeAsync(200)
    await flush()
    expect(instance.filteredTasks.value.map((task) => task.jobId)).toEqual([
      'beta'
    ])

    instance.searchQuery.value = 'failed meta'
    await vi.advanceTimersByTimeAsync(200)
    await flush()
    expect(instance.filteredTasks.value.map((task) => task.jobId)).toEqual([
      'beta'
    ])

    instance.searchQuery.value = 'does-not-exist'
    await vi.advanceTimersByTimeAsync(200)
    await flush()
    expect(instance.filteredTasks.value).toEqual([])
  })

  it('hydrates job items with active progress and compute hours', async () => {
    queueStoreMock.runningTasks = [
      createTask({
        jobId: 'active',
        job: { priority: 3 },
        mockState: 'running',
        executionTime: 7_200_000
      }),
      createTask({
        jobId: 'other',
        job: { priority: 2 },
        mockState: 'running',
        executionTime: 3_600_000
      })
    ]

    executionStoreMock.activeJobId = 'active'
    executionStoreMock.executingNode = { title: 'Render Node' }
    totalPercent.value = 80
    currentNodePercent.value = 40

    const { jobItems } = initComposable()
    await flush()

    const [activeJob, otherJob] = jobItems.value
    expect(activeJob.progressTotalPercent).toBe(80)
    expect(activeJob.progressCurrentPercent).toBe(40)
    expect(activeJob.runningNodeName).toBe('Render Node')
    expect(activeJob.computeHours).toBeCloseTo(2)

    expect(otherJob.progressTotalPercent).toBeUndefined()
    expect(otherJob.progressCurrentPercent).toBeUndefined()
    expect(otherJob.runningNodeName).toBeUndefined()
    expect(otherJob.computeHours).toBeCloseTo(1)
  })

  it('assigns preview urls for running jobs when previews enabled', async () => {
    queueStoreMock.runningTasks = [
      createTask({
        jobId: 'live-preview',
        job: { priority: 1 },
        mockState: 'running'
      })
    ]
    jobPreviewStoreMock.previewsByPromptId = {
      'live-preview': 'blob:preview-url'
    }
    jobPreviewStoreMock.isPreviewEnabled = true

    const { jobItems } = initComposable()
    await flush()

    expect(jobItems.value[0].iconImageUrl).toBe('blob:preview-url')
  })

  it('omits preview urls when previews are disabled', async () => {
    queueStoreMock.runningTasks = [
      createTask({
        jobId: 'disabled-preview',
        job: { priority: 1 },
        mockState: 'running'
      })
    ]
    jobPreviewStoreMock.previewsByPromptId = {
      'disabled-preview': 'blob:preview-url'
    }
    jobPreviewStoreMock.isPreviewEnabled = false

    const { jobItems } = initComposable()
    await flush()

    expect(jobItems.value[0].iconImageUrl).toBeUndefined()
  })

  it('derives current node name from execution store fallbacks', async () => {
    const instance = initComposable()
    await flush()

    expect(instance.currentNodeName.value).toBe('--')

    executionStoreMock.executingNode = { title: '  Visible Node  ' }
    await flush()
    expect(instance.currentNodeName.value).toBe('Visible Node')

    executionStoreMock.executingNode = {
      title: '   ',
      type: 'My Node Type'
    }
    await flush()
    expect(instance.currentNodeName.value).toBe(
      'i18n(nodeDefs.My Node Type.display_name)-My Node Type'
    )
  })

  it('groups job items by date label and sorts by total generation time when requested', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2024-01-10T12:00:00Z'))
    queueStoreMock.historyTasks = [
      createTask({
        jobId: 'today-small',
        job: { priority: 4 },
        mockState: 'completed',
        executionEndTimestamp: Date.now(),
        executionTime: 2_000
      }),
      createTask({
        jobId: 'today-large',
        job: { priority: 3 },
        mockState: 'completed',
        executionEndTimestamp: Date.now(),
        executionTime: 5_000
      }),
      createTask({
        jobId: 'yesterday',
        job: { priority: 2 },
        mockState: 'failed',
        executionEndTimestamp: Date.now() - 86_400_000,
        executionTime: 1_000
      }),
      createTask({
        jobId: 'undated',
        job: { priority: 1 },
        mockState: 'pending'
      })
    ]

    const instance = initComposable()
    instance.selectedSortMode.value = 'totalGenerationTime'
    await flush()

    const groups = instance.groupedJobItems.value
    expect(groups.map((g) => g.label)).toEqual([
      'Today',
      'Yesterday',
      'Undated'
    ])

    const todayGroup = groups[0]
    expect(todayGroup.items.map((item) => item.id)).toEqual([
      'today-large',
      'today-small'
    ])
  })
})
