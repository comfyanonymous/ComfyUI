import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { JobListItem } from '@/platform/remote/comfyui/jobs/jobTypes'
import type { TaskOutput } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { TaskItemImpl, useQueueStore } from '@/stores/queueStore'

// Fixture factory for JobListItem
function createJob(
  id: string,
  status: JobListItem['status'],
  createTime: number = Date.now(),
  priority?: number
): JobListItem {
  return {
    id,
    status,
    create_time: createTime,
    update_time: createTime,
    last_state_update: createTime,
    priority: priority ?? createTime
  }
}

function createRunningJob(createTime: number, id: string): JobListItem {
  return createJob(id, 'in_progress', createTime)
}

function createPendingJob(createTime: number, id: string): JobListItem {
  return createJob(id, 'pending', createTime)
}

function createHistoryJob(createTime: number, id: string): JobListItem {
  return createJob(id, 'completed', createTime)
}

const createTaskOutput = (
  nodeId: string = 'node-1',
  images: {
    type?: 'output' | 'input' | 'temp'
    filename?: string
    subfolder?: string
  }[] = []
): TaskOutput => ({
  [nodeId]: {
    images
  }
})

type QueueResponse = { Running: JobListItem[]; Pending: JobListItem[] }
type QueueResolver = (value: QueueResponse) => void

// Mock API
vi.mock('@/scripts/api', () => ({
  api: {
    getQueue: vi.fn(),
    getHistory: vi.fn(),
    clearItems: vi.fn(),
    deleteItem: vi.fn(),
    apiURL: vi.fn((path) => `/api${path}`),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn()
  }
}))

describe('TaskItemImpl', () => {
  it('should remove animated property from outputs during construction', () => {
    const job = createHistoryJob(0, 'job-id')
    const taskItem = new TaskItemImpl(job, {
      'node-1': {
        images: [{ filename: 'test.png', type: 'output', subfolder: '' }],
        animated: [false]
      }
    })

    // Check that animated property was removed
    expect('animated' in taskItem.outputs['node-1']).toBe(false)

    expect(taskItem.outputs['node-1'].images).toBeDefined()
    expect(taskItem.outputs['node-1'].images?.[0]?.filename).toBe('test.png')
  })

  it('should handle outputs without animated property', () => {
    const job = createHistoryJob(0, 'job-id')
    const taskItem = new TaskItemImpl(job, {
      'node-1': {
        images: [{ filename: 'test.png', type: 'output', subfolder: '' }]
      }
    })

    expect(taskItem.outputs['node-1'].images).toBeDefined()
    expect(taskItem.outputs['node-1'].images?.[0]?.filename).toBe('test.png')
  })

  it('should recognize webm video from core', () => {
    const job = createHistoryJob(0, 'job-id')
    const taskItem = new TaskItemImpl(job, {
      'node-1': {
        video: [{ filename: 'test.webm', type: 'output', subfolder: '' }]
      }
    })

    const output = taskItem.flatOutputs[0]

    expect(output.htmlVideoType).toBe('video/webm')
    expect(output.isVideo).toBe(true)
    expect(output.isVhsFormat).toBe(false)
    expect(output.isImage).toBe(false)
  })

  // https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/0a75c7958fe320efcb052f1d9f8451fd20c730a8/videohelpersuite/nodes.py#L578-L590
  it('should recognize webm video from VHS', () => {
    const job = createHistoryJob(0, 'job-id')
    const taskItem = new TaskItemImpl(job, {
      'node-1': {
        gifs: [
          {
            filename: 'test.webm',
            type: 'output',
            subfolder: '',
            format: 'video/webm',
            frame_rate: 30
          }
        ]
      }
    })

    const output = taskItem.flatOutputs[0]

    expect(output.htmlVideoType).toBe('video/webm')
    expect(output.isVideo).toBe(true)
    expect(output.isVhsFormat).toBe(true)
    expect(output.isImage).toBe(false)
  })

  it('should recognize mp4 video from core', () => {
    const job = createHistoryJob(0, 'job-id')
    const taskItem = new TaskItemImpl(job, {
      'node-1': {
        images: [
          {
            filename: 'test.mp4',
            type: 'output',
            subfolder: ''
          }
        ],
        animated: [true]
      }
    })

    const output = taskItem.flatOutputs[0]

    expect(output.htmlVideoType).toBe('video/mp4')
    expect(output.isVideo).toBe(true)
    expect(output.isImage).toBe(false)
  })

  describe('audio format detection', () => {
    const audioFormats = [
      { extension: 'mp3', mimeType: 'audio/mpeg' },
      { extension: 'wav', mimeType: 'audio/wav' },
      { extension: 'ogg', mimeType: 'audio/ogg' },
      { extension: 'flac', mimeType: 'audio/flac' }
    ]

    audioFormats.forEach(({ extension, mimeType }) => {
      it(`should recognize ${extension} audio`, () => {
        const job = createHistoryJob(0, 'job-id')
        const taskItem = new TaskItemImpl(job, {
          'node-1': {
            audio: [
              {
                filename: `test.${extension}`,
                type: 'output',
                subfolder: ''
              }
            ]
          }
        })

        const output = taskItem.flatOutputs[0]

        expect(output.htmlAudioType).toBe(mimeType)
        expect(output.isAudio).toBe(true)
        expect(output.isVideo).toBe(false)
        expect(output.isImage).toBe(false)
        expect(output.supportsPreview).toBe(true)
      })
    })
  })

  describe('error extraction getters', () => {
    it('errorMessage returns undefined when no execution_error', () => {
      const job = createHistoryJob(0, 'job-id')
      const taskItem = new TaskItemImpl(job)
      expect(taskItem.errorMessage).toBeUndefined()
    })

    it('errorMessage returns the exception_message from execution_error', () => {
      const job: JobListItem = {
        ...createHistoryJob(0, 'job-id'),
        status: 'failed',
        execution_error: {
          node_id: 'node-1',
          node_type: 'KSampler',
          exception_message: 'GPU out of memory',
          exception_type: 'RuntimeError',
          traceback: ['line 1', 'line 2'],
          current_inputs: {},
          current_outputs: {}
        }
      }
      const taskItem = new TaskItemImpl(job)
      expect(taskItem.errorMessage).toBe('GPU out of memory')
    })

    it('executionError returns undefined when no execution_error', () => {
      const job = createHistoryJob(0, 'job-id')
      const taskItem = new TaskItemImpl(job)
      expect(taskItem.executionError).toBeUndefined()
    })

    it('executionError returns the full error object from execution_error', () => {
      const errorDetail = {
        node_id: 'node-1',
        node_type: 'KSampler',
        executed: ['node-0'],
        exception_message: 'Invalid dimensions',
        exception_type: 'ValueError',
        traceback: ['traceback line'],
        current_inputs: { input1: 'value' },
        current_outputs: {}
      }
      const job: JobListItem = {
        ...createHistoryJob(0, 'job-id'),
        status: 'failed',
        execution_error: errorDetail
      }
      const taskItem = new TaskItemImpl(job)
      expect(taskItem.executionError).toEqual(errorDetail)
    })
  })
})

describe('useQueueStore', () => {
  let store: ReturnType<typeof useQueueStore>

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useQueueStore()
    vi.clearAllMocks()
  })

  const mockGetQueue = vi.mocked(api.getQueue)
  const mockGetHistory = vi.mocked(api.getHistory)
  const mockClearItems = vi.mocked(api.clearItems)
  const mockDeleteItem = vi.mocked(api.deleteItem)

  describe('initial state', () => {
    it('should have empty state on initialization', () => {
      expect(store.runningTasks).toEqual([])
      expect(store.pendingTasks).toEqual([])
      expect(store.historyTasks).toEqual([])
      expect(store.isLoading).toBe(false)
      expect(store.maxHistoryItems).toBe(64)
    })

    it('should have empty computed tasks', () => {
      expect(store.tasks).toEqual([])
      expect(store.flatTasks).toEqual([])
      expect(store.hasPendingTasks).toBe(false)
      expect(store.lastJobHistoryPriority).toBe(-1)
    })
  })

  describe('update() - basic functionality', () => {
    it('should load running and pending tasks from API', async () => {
      const runningJob = createRunningJob(1, 'run-1')
      const pendingJob1 = createPendingJob(2, 'pend-1')
      const pendingJob2 = createPendingJob(3, 'pend-2')

      // API returns pre-sorted data (newest first)
      mockGetQueue.mockResolvedValue({
        Running: [runningJob],
        Pending: [pendingJob2, pendingJob1] // Pre-sorted by create_time desc
      })
      mockGetHistory.mockResolvedValue([])

      await store.update()

      expect(store.runningTasks).toHaveLength(1)
      expect(store.pendingTasks).toHaveLength(2)
      expect(store.runningTasks[0].jobId).toBe('run-1')
      expect(store.pendingTasks[0].jobId).toBe('pend-2')
      expect(store.pendingTasks[1].jobId).toBe('pend-1')
    })

    it('should load history tasks from API', async () => {
      const historyJob1 = createHistoryJob(5, 'hist-1')
      const historyJob2 = createHistoryJob(4, 'hist-2')

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([historyJob1, historyJob2])

      await store.update()

      expect(store.historyTasks).toHaveLength(2)
      expect(store.historyTasks[0].jobId).toBe('hist-1')
      expect(store.historyTasks[1].jobId).toBe('hist-2')
    })

    it('should set loading state correctly', async () => {
      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([])

      expect(store.isLoading).toBe(false)

      const updatePromise = store.update()
      expect(store.isLoading).toBe(true)

      await updatePromise
      expect(store.isLoading).toBe(false)
    })

    it('should clear loading state even if API fails', async () => {
      mockGetQueue.mockRejectedValue(new Error('API error'))
      mockGetHistory.mockResolvedValue([])

      await expect(store.update()).rejects.toThrow('API error')
      expect(store.isLoading).toBe(false)
    })
  })

  describe('update() - sorting', () => {
    it('should sort tasks by job.priority descending', async () => {
      const job1 = createHistoryJob(1, 'hist-1')
      const job2 = createHistoryJob(5, 'hist-2')
      const job3 = createHistoryJob(3, 'hist-3')

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([job1, job2, job3])

      await store.update()

      expect(store.historyTasks[0].job.priority).toBe(5)
      expect(store.historyTasks[1].job.priority).toBe(3)
      expect(store.historyTasks[2].job.priority).toBe(1)
    })

    it('should preserve API sort order for pending tasks', async () => {
      const pend1 = createPendingJob(10, 'pend-1')
      const pend2 = createPendingJob(15, 'pend-2')
      const pend3 = createPendingJob(12, 'pend-3')

      // API returns pre-sorted data (newest first)
      mockGetQueue.mockResolvedValue({
        Running: [],
        Pending: [pend2, pend3, pend1] // Pre-sorted by create_time desc
      })
      mockGetHistory.mockResolvedValue([])

      await store.update()

      expect(store.pendingTasks[0].job.priority).toBe(15)
      expect(store.pendingTasks[1].job.priority).toBe(12)
      expect(store.pendingTasks[2].job.priority).toBe(10)
    })
  })

  describe('update() - queue index collision (THE BUG FIX)', () => {
    it('should NOT confuse different prompts with same job.priority', async () => {
      const hist1 = createHistoryJob(50, 'prompt-uuid-aaa')

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([hist1])

      await store.update()
      expect(store.historyTasks).toHaveLength(1)
      expect(store.historyTasks[0].jobId).toBe('prompt-uuid-aaa')

      const hist2 = createHistoryJob(51, 'prompt-uuid-bbb')
      mockGetHistory.mockResolvedValue([hist2])

      await store.update()

      expect(store.historyTasks).toHaveLength(1)
      expect(store.historyTasks[0].jobId).toBe('prompt-uuid-bbb')
      expect(store.historyTasks[0].job.priority).toBe(51)
    })

    it('should correctly reconcile when job.priority is reused', async () => {
      const hist1 = createHistoryJob(100, 'first-prompt-at-100')
      const hist2 = createHistoryJob(99, 'prompt-at-99')

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([hist1, hist2])

      await store.update()
      expect(store.historyTasks).toHaveLength(2)

      const hist3 = createHistoryJob(101, 'second-prompt-at-101')
      mockGetHistory.mockResolvedValue([hist3, hist2])

      await store.update()

      expect(store.historyTasks).toHaveLength(2)
      const jobIds = store.historyTasks.map((t) => t.jobId)
      expect(jobIds).toContain('second-prompt-at-101')
      expect(jobIds).toContain('prompt-at-99')
      expect(jobIds).not.toContain('first-prompt-at-100')
    })

    it('should handle multiple job.priority collisions simultaneously', async () => {
      const hist1 = createHistoryJob(10, 'old-at-10')
      const hist2 = createHistoryJob(20, 'old-at-20')
      const hist3 = createHistoryJob(30, 'keep-at-30')

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([hist3, hist2, hist1])

      await store.update()
      expect(store.historyTasks).toHaveLength(3)

      const newHist1 = createHistoryJob(31, 'new-at-31')
      const newHist2 = createHistoryJob(32, 'new-at-32')
      mockGetHistory.mockResolvedValue([newHist2, newHist1, hist3])

      await store.update()

      expect(store.historyTasks).toHaveLength(3)
      const jobIds = store.historyTasks.map((t) => t.jobId)
      expect(jobIds).toEqual(['new-at-32', 'new-at-31', 'keep-at-30'])
    })
  })

  describe('update() - history reconciliation', () => {
    it('should keep existing items still on server (by jobId)', async () => {
      const hist1 = createHistoryJob(10, 'existing-1')
      const hist2 = createHistoryJob(9, 'existing-2')

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([hist1, hist2])

      await store.update()
      expect(store.historyTasks).toHaveLength(2)

      const hist3 = createHistoryJob(11, 'new-1')
      mockGetHistory.mockResolvedValue([hist3, hist1, hist2])

      await store.update()

      expect(store.historyTasks).toHaveLength(3)
      expect(store.historyTasks.map((t) => t.jobId)).toContain('existing-1')
      expect(store.historyTasks.map((t) => t.jobId)).toContain('existing-2')
      expect(store.historyTasks.map((t) => t.jobId)).toContain('new-1')
    })

    it('should remove items no longer on server', async () => {
      const hist1 = createHistoryJob(10, 'remove-me')
      const hist2 = createHistoryJob(9, 'keep-me')

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([hist1, hist2])

      await store.update()
      expect(store.historyTasks).toHaveLength(2)

      mockGetHistory.mockResolvedValue([hist2])

      await store.update()

      expect(store.historyTasks).toHaveLength(1)
      expect(store.historyTasks[0].jobId).toBe('keep-me')
    })

    it('should add new items from server', async () => {
      const hist1 = createHistoryJob(5, 'old-1')

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([hist1])

      await store.update()

      const hist2 = createHistoryJob(6, 'new-1')
      const hist3 = createHistoryJob(7, 'new-2')
      mockGetHistory.mockResolvedValue([hist3, hist2, hist1])

      await store.update()

      expect(store.historyTasks).toHaveLength(3)
      expect(store.historyTasks.map((t) => t.jobId)).toContain('new-1')
      expect(store.historyTasks.map((t) => t.jobId)).toContain('new-2')
    })

    it('should recreate TaskItemImpl when outputs_count changes', async () => {
      // Initial load without outputs_count
      const jobWithoutOutputsCount = createHistoryJob(10, 'job-1')
      delete jobWithoutOutputsCount.outputs_count

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([jobWithoutOutputsCount])

      await store.update()
      expect(store.historyTasks).toHaveLength(1)
      const initialTask = store.historyTasks[0]
      expect(initialTask.outputsCount).toBeUndefined()

      // Second load with outputs_count now populated
      const jobWithOutputsCount = {
        ...createHistoryJob(10, 'job-1'),
        outputs_count: 2
      }
      mockGetHistory.mockResolvedValue([jobWithOutputsCount])

      await store.update()

      // Should have recreated the TaskItemImpl with new outputs_count
      expect(store.historyTasks).toHaveLength(1)
      const updatedTask = store.historyTasks[0]
      expect(updatedTask.outputsCount).toBe(2)
      // Should be a different instance
      expect(updatedTask).not.toBe(initialTask)
    })

    it('should reuse TaskItemImpl when outputs_count unchanged', async () => {
      const job = {
        ...createHistoryJob(10, 'job-1'),
        outputs_count: 2
      }

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([job])

      await store.update()
      const initialTask = store.historyTasks[0]
      const initialHistoryTasks = store.historyTasks

      // Same job with same outputs_count
      mockGetHistory.mockResolvedValue([{ ...job }])

      await store.update()

      // Should reuse the same instance
      expect(store.historyTasks[0]).toBe(initialTask)
      // Should preserve array identity when history is unchanged
      expect(store.historyTasks).toBe(initialHistoryTasks)
    })
  })

  describe('update() - maxHistoryItems limit', () => {
    it('should enforce maxHistoryItems limit', async () => {
      store.maxHistoryItems = 3

      const jobs = Array.from({ length: 5 }, (_, i) =>
        createHistoryJob(10 - i, `hist-${i}`)
      )

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue(jobs)

      await store.update()

      expect(store.historyTasks).toHaveLength(3)
      expect(store.historyTasks[0].job.priority).toBe(10)
      expect(store.historyTasks[1].job.priority).toBe(9)
      expect(store.historyTasks[2].job.priority).toBe(8)
    })

    it('should respect maxHistoryItems when combining new and existing', async () => {
      store.maxHistoryItems = 5

      const initial = Array.from({ length: 3 }, (_, i) =>
        createHistoryJob(10 + i, `existing-${i}`)
      )

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue(initial)

      await store.update()
      expect(store.historyTasks).toHaveLength(3)

      const newJobs = Array.from({ length: 4 }, (_, i) =>
        createHistoryJob(20 + i, `new-${i}`)
      )
      mockGetHistory.mockResolvedValue([...newJobs, ...initial])

      await store.update()

      expect(store.historyTasks).toHaveLength(5)
      expect(store.historyTasks[0].job.priority).toBe(23)
    })

    it('should handle maxHistoryItems = 0', async () => {
      store.maxHistoryItems = 0

      const jobs = [createHistoryJob(10, 'hist-1')]

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue(jobs)

      await store.update()

      expect(store.historyTasks).toHaveLength(0)
    })

    it('should handle maxHistoryItems = 1', async () => {
      store.maxHistoryItems = 1

      const jobs = [
        createHistoryJob(10, 'hist-1'),
        createHistoryJob(9, 'hist-2')
      ]

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue(jobs)

      await store.update()

      expect(store.historyTasks).toHaveLength(1)
      expect(store.historyTasks[0].job.priority).toBe(10)
    })

    it('should dynamically adjust when maxHistoryItems changes', async () => {
      store.maxHistoryItems = 10

      const jobs = Array.from({ length: 15 }, (_, i) =>
        createHistoryJob(20 - i, `hist-${i}`)
      )

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue(jobs)

      await store.update()
      expect(store.historyTasks).toHaveLength(10)

      store.maxHistoryItems = 5
      mockGetHistory.mockResolvedValue(jobs)

      await store.update()
      expect(store.historyTasks).toHaveLength(5)
    })
  })

  describe('computed properties', () => {
    it('tasks should combine pending, running, and history in correct order', async () => {
      const running = createRunningJob(5, 'run-1')
      const pending1 = createPendingJob(6, 'pend-1')
      const pending2 = createPendingJob(7, 'pend-2')
      const hist1 = createHistoryJob(3, 'hist-1')
      const hist2 = createHistoryJob(4, 'hist-2')

      mockGetQueue.mockResolvedValue({
        Running: [running],
        Pending: [pending1, pending2]
      })
      mockGetHistory.mockResolvedValue([hist2, hist1])

      await store.update()

      expect(store.tasks).toHaveLength(5)
      expect(store.tasks[0].taskType).toBe('Pending')
      expect(store.tasks[1].taskType).toBe('Pending')
      expect(store.tasks[2].taskType).toBe('Running')
      expect(store.tasks[3].taskType).toBe('History')
      expect(store.tasks[4].taskType).toBe('History')
    })

    it('hasPendingTasks should be true when pending tasks exist', async () => {
      mockGetQueue.mockResolvedValue({
        Running: [],
        Pending: [createPendingJob(1, 'pend-1')]
      })
      mockGetHistory.mockResolvedValue([])

      await store.update()
      expect(store.hasPendingTasks).toBe(true)
    })

    it('hasPendingTasks should be false when no pending tasks', async () => {
      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([])

      await store.update()
      expect(store.hasPendingTasks).toBe(false)
    })

    it('lastJobHistoryPriority should return highest job priority', async () => {
      const hist1 = createHistoryJob(10, 'hist-1')
      const hist2 = createHistoryJob(25, 'hist-2')
      const hist3 = createHistoryJob(15, 'hist-3')

      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([hist1, hist2, hist3])

      await store.update()
      expect(store.lastJobHistoryPriority).toBe(25)
    })

    it('lastJobHistoryPriority should be -1 when no history', async () => {
      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([])

      await store.update()
      expect(store.lastJobHistoryPriority).toBe(-1)
    })
  })

  describe('clear()', () => {
    beforeEach(async () => {
      mockGetQueue.mockResolvedValue({
        Running: [createRunningJob(1, 'run-1')],
        Pending: [createPendingJob(2, 'pend-1')]
      })
      mockGetHistory.mockResolvedValue([createHistoryJob(3, 'hist-1')])
      await store.update()
    })

    it('should clear both queue and history by default', async () => {
      mockClearItems.mockResolvedValue(undefined)
      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([])

      await store.clear()

      expect(mockClearItems).toHaveBeenCalledTimes(2)
      expect(mockClearItems).toHaveBeenCalledWith('queue')
      expect(mockClearItems).toHaveBeenCalledWith('history')
      expect(store.runningTasks).toHaveLength(0)
      expect(store.pendingTasks).toHaveLength(0)
      expect(store.historyTasks).toHaveLength(0)
    })

    it('should clear only queue when specified', async () => {
      mockClearItems.mockResolvedValue(undefined)
      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([createHistoryJob(3, 'hist-1')])

      await store.clear(['queue'])

      expect(mockClearItems).toHaveBeenCalledTimes(1)
      expect(mockClearItems).toHaveBeenCalledWith('queue')
      expect(store.historyTasks).toHaveLength(1)
    })

    it('should clear only history when specified', async () => {
      mockClearItems.mockResolvedValue(undefined)
      mockGetQueue.mockResolvedValue({
        Running: [createRunningJob(1, 'run-1')],
        Pending: [createPendingJob(2, 'pend-1')]
      })
      mockGetHistory.mockResolvedValue([])

      await store.clear(['history'])

      expect(mockClearItems).toHaveBeenCalledTimes(1)
      expect(mockClearItems).toHaveBeenCalledWith('history')
      expect(store.runningTasks).toHaveLength(1)
      expect(store.pendingTasks).toHaveLength(1)
    })

    it('should do nothing when empty array passed', async () => {
      await store.clear([])

      expect(mockClearItems).not.toHaveBeenCalled()
    })
  })

  describe('delete()', () => {
    it('should delete task from queue', async () => {
      const job = createPendingJob(1, 'pend-1')
      const task = new TaskItemImpl(job)

      mockDeleteItem.mockResolvedValue(undefined)
      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([])

      await store.delete(task)

      expect(mockDeleteItem).toHaveBeenCalledWith('queue', 'pend-1')
    })

    it('should delete task from history', async () => {
      const job = createHistoryJob(1, 'hist-1')
      const task = new TaskItemImpl(job, createTaskOutput())

      mockDeleteItem.mockResolvedValue(undefined)
      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([])

      await store.delete(task)

      expect(mockDeleteItem).toHaveBeenCalledWith('history', 'hist-1')
    })

    it('should refresh store after deletion', async () => {
      const job = createPendingJob(1, 'pend-1')
      const task = new TaskItemImpl(job)

      mockDeleteItem.mockResolvedValue(undefined)
      mockGetQueue.mockResolvedValue({ Running: [], Pending: [] })
      mockGetHistory.mockResolvedValue([])

      await store.delete(task)

      expect(mockGetQueue).toHaveBeenCalled()
      expect(mockGetHistory).toHaveBeenCalled()
    })
  })

  describe('update deduplication', () => {
    it('should discard stale responses when newer request completes first', async () => {
      let resolveFirst: QueueResolver
      let resolveSecond: QueueResolver

      const firstQueuePromise = new Promise<QueueResponse>((resolve) => {
        resolveFirst = resolve
      })
      const secondQueuePromise = new Promise<QueueResponse>((resolve) => {
        resolveSecond = resolve
      })

      mockGetHistory.mockResolvedValue([])

      mockGetQueue
        .mockReturnValueOnce(firstQueuePromise)
        .mockReturnValueOnce(secondQueuePromise)

      const firstUpdate = store.update()
      const secondUpdate = store.update()

      resolveSecond!({ Running: [], Pending: [createPendingJob(2, 'new-job')] })
      await secondUpdate

      expect(store.pendingTasks).toHaveLength(1)
      expect(store.pendingTasks[0].jobId).toBe('new-job')

      resolveFirst!({
        Running: [],
        Pending: [createPendingJob(1, 'stale-job')]
      })
      await firstUpdate

      expect(store.pendingTasks).toHaveLength(1)
      expect(store.pendingTasks[0].jobId).toBe('new-job')
    })

    it('should set isLoading to false only for the latest request', async () => {
      let resolveFirst: QueueResolver
      let resolveSecond: QueueResolver

      const firstQueuePromise = new Promise<QueueResponse>((resolve) => {
        resolveFirst = resolve
      })
      const secondQueuePromise = new Promise<QueueResponse>((resolve) => {
        resolveSecond = resolve
      })

      mockGetHistory.mockResolvedValue([])

      mockGetQueue
        .mockReturnValueOnce(firstQueuePromise)
        .mockReturnValueOnce(secondQueuePromise)

      const firstUpdate = store.update()
      expect(store.isLoading).toBe(true)

      const secondUpdate = store.update()
      expect(store.isLoading).toBe(true)

      resolveSecond!({ Running: [], Pending: [] })
      await secondUpdate

      expect(store.isLoading).toBe(false)

      resolveFirst!({ Running: [], Pending: [] })
      await firstUpdate

      expect(store.isLoading).toBe(false)
    })

    it('should handle stale request failure without affecting latest state', async () => {
      let resolveSecond: QueueResolver

      const secondQueuePromise = new Promise<QueueResponse>((resolve) => {
        resolveSecond = resolve
      })

      mockGetHistory.mockResolvedValue([])

      mockGetQueue
        .mockRejectedValueOnce(new Error('stale network error'))
        .mockReturnValueOnce(secondQueuePromise)

      const firstUpdate = store.update()
      const secondUpdate = store.update()

      resolveSecond!({ Running: [], Pending: [createPendingJob(2, 'new-job')] })
      await secondUpdate

      expect(store.pendingTasks).toHaveLength(1)
      expect(store.pendingTasks[0].jobId).toBe('new-job')
      expect(store.isLoading).toBe(false)

      await expect(firstUpdate).rejects.toThrow('stale network error')

      expect(store.pendingTasks).toHaveLength(1)
      expect(store.pendingTasks[0].jobId).toBe('new-job')
      expect(store.isLoading).toBe(false)
    })
  })
})
