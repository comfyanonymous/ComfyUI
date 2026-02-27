import { beforeEach, describe, expect, it, vi } from 'vitest'

import { extractWorkflow } from '@/platform/remote/comfyui/jobs/fetchJobs'
import { api } from '@/scripts/api'
import type {
  JobDetail,
  JobListItem
} from '@/platform/remote/comfyui/jobs/jobTypes'
import type { ComfyWorkflowJSON } from '@/platform/workflow/validation/schemas/workflowSchema'
import {
  findActiveIndex,
  getJobDetail,
  getJobWorkflow,
  getOutputsForTask
} from '@/services/jobOutputCache'
import { ResultItemImpl, TaskItemImpl } from '@/stores/queueStore'

vi.mock('@/platform/remote/comfyui/jobs/fetchJobs', () => ({
  fetchJobDetail: vi.fn(),
  extractWorkflow: vi.fn()
}))

vi.mock('@/scripts/api', () => ({
  api: {
    getJobDetail: vi.fn(),
    apiURL: vi.fn((path: string) => `/api${path}`),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn()
  }
}))

function createResultItem(url: string, supportsPreview = true): ResultItemImpl {
  const item = new ResultItemImpl({
    filename: url,
    subfolder: '',
    type: 'output',
    nodeId: 'node-1',
    mediaType: supportsPreview ? 'images' : 'unknown'
  })
  Object.defineProperty(item, 'url', { get: () => url })
  Object.defineProperty(item, 'supportsPreview', { get: () => supportsPreview })
  return item
}

function createMockJob(id: string, outputsCount = 1): JobListItem {
  return {
    id,
    status: 'completed',
    create_time: Date.now(),
    preview_output: null,
    outputs_count: outputsCount,
    priority: 0
  }
}

function createTask(
  preview?: ResultItemImpl,
  allOutputs?: ResultItemImpl[],
  outputsCount = 1
): TaskItemImpl {
  const job = createMockJob(
    `task-${Math.random().toString(36).slice(2)}`,
    outputsCount
  )
  const flatOutputs = allOutputs ?? (preview ? [preview] : [])
  return new TaskItemImpl(job, {}, flatOutputs)
}

// Generate unique IDs per test to avoid cache collisions
let testCounter = 0
function uniqueId(prefix: string): string {
  return `${prefix}-${++testCounter}-${Date.now()}`
}

describe('jobOutputCache', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('findActiveIndex', () => {
    it('returns index of matching URL', () => {
      const items = [
        createResultItem('a'),
        createResultItem('b'),
        createResultItem('c')
      ]

      expect(findActiveIndex(items, 'b')).toBe(1)
    })

    it('returns 0 when URL not found', () => {
      const items = [createResultItem('a'), createResultItem('b')]

      expect(findActiveIndex(items, 'missing')).toBe(0)
    })

    it('returns 0 when URL is undefined', () => {
      const items = [createResultItem('a'), createResultItem('b')]

      expect(findActiveIndex(items, undefined)).toBe(0)
    })
  })

  describe('getOutputsForTask', () => {
    it('returns previewable outputs directly when no lazy load needed', async () => {
      const outputs = [createResultItem('p-1'), createResultItem('p-2')]
      const task = createTask(undefined, outputs, 1)

      const result = await getOutputsForTask(task)

      expect(result).toEqual(outputs)
    })

    it('lazy loads when outputsCount > 1', async () => {
      const previewOutput = createResultItem('preview')
      const fullOutputs = [
        createResultItem('full-1'),
        createResultItem('full-2')
      ]

      const job = createMockJob(uniqueId('task'), 3)
      const task = new TaskItemImpl(job, {}, [previewOutput])
      const loadedTask = new TaskItemImpl(job, {}, fullOutputs)
      task.loadFullOutputs = vi.fn().mockResolvedValue(loadedTask)

      const result = await getOutputsForTask(task)

      expect(result).toEqual(fullOutputs)
      expect(task.loadFullOutputs).toHaveBeenCalled()
    })

    it('caches loaded tasks', async () => {
      const fullOutputs = [createResultItem('full-1')]

      const job = createMockJob(uniqueId('task'), 3)
      const task = new TaskItemImpl(job, {}, [createResultItem('preview')])
      const loadedTask = new TaskItemImpl(job, {}, fullOutputs)
      task.loadFullOutputs = vi.fn().mockResolvedValue(loadedTask)

      // First call should load
      await getOutputsForTask(task)
      expect(task.loadFullOutputs).toHaveBeenCalledTimes(1)

      // Second call should use cache
      await getOutputsForTask(task)
      expect(task.loadFullOutputs).toHaveBeenCalledTimes(1)
    })

    it('falls back to preview outputs on load error', async () => {
      const previewOutput = createResultItem('preview')

      const job = createMockJob(uniqueId('task'), 3)
      const task = new TaskItemImpl(job, {}, [previewOutput])
      task.loadFullOutputs = vi
        .fn()
        .mockRejectedValue(new Error('Network error'))

      const result = await getOutputsForTask(task)

      expect(result).toEqual([previewOutput])
    })

    it('returns null when request is superseded', async () => {
      const job1 = createMockJob(uniqueId('task'), 3)
      const job2 = createMockJob(uniqueId('task'), 3)

      const task1 = new TaskItemImpl(job1, {}, [createResultItem('preview-1')])
      const task2 = new TaskItemImpl(job2, {}, [createResultItem('preview-2')])

      const loadedTask1 = new TaskItemImpl(job1, {}, [
        createResultItem('full-1')
      ])
      const loadedTask2 = new TaskItemImpl(job2, {}, [
        createResultItem('full-2')
      ])

      // Task1 loads slowly, task2 loads quickly
      task1.loadFullOutputs = vi.fn().mockImplementation(
        () =>
          new Promise((resolve) => {
            setTimeout(() => resolve(loadedTask1), 50)
          })
      )
      task2.loadFullOutputs = vi.fn().mockResolvedValue(loadedTask2)

      // Start task1, then immediately start task2
      const promise1 = getOutputsForTask(task1)
      const promise2 = getOutputsForTask(task2)

      const [result1, result2] = await Promise.all([promise1, promise2])

      // Task2 should succeed, task1 should return null (superseded)
      expect(result1).toBeNull()
      expect(result2).toEqual([createResultItem('full-2')])
    })
  })

  describe('getPreviewableOutputsFromJobDetail', () => {
    it('returns empty array when job detail or outputs are missing', async () => {
      const { getPreviewableOutputsFromJobDetail } =
        await import('@/services/jobOutputCache')

      expect(getPreviewableOutputsFromJobDetail(undefined)).toEqual([])

      const jobDetail: JobDetail = {
        id: 'job-empty',
        status: 'completed',
        create_time: Date.now(),
        priority: 0
      }

      expect(getPreviewableOutputsFromJobDetail(jobDetail)).toEqual([])
    })

    it('maps previewable outputs and skips animated/text entries', async () => {
      const { getPreviewableOutputsFromJobDetail } =
        await import('@/services/jobOutputCache')
      const jobDetail: JobDetail = {
        id: 'job-previewable',
        status: 'completed',
        create_time: Date.now(),
        priority: 0,
        outputs: {
          'node-1': {
            images: [
              { filename: 'image.png', subfolder: '', type: 'output' },
              { filename: 'image.webp', subfolder: '', type: 'temp' }
            ],
            animated: [true],
            text: 'hello'
          },
          'node-2': {
            video: [{ filename: 'clip.mp4', subfolder: '', type: 'output' }],
            audio: [{ filename: 'sound.mp3', subfolder: '', type: 'output' }]
          }
        }
      }

      const result = getPreviewableOutputsFromJobDetail(jobDetail)

      expect(result).toHaveLength(4)
      expect(result.map((item) => item.filename).sort()).toEqual(
        ['image.png', 'image.webp', 'clip.mp4', 'sound.mp3'].sort()
      )

      const image = result.find((item) => item.filename === 'image.png')
      const video = result.find((item) => item.filename === 'clip.mp4')
      const { ResultItemImpl: ResultItemImplClass } =
        await import('@/stores/queueStore')

      expect(image).toBeInstanceOf(ResultItemImplClass)
      expect(image?.nodeId).toBe('node-1')
      expect(image?.mediaType).toBe('images')
      expect(video?.nodeId).toBe('node-2')
      expect(video?.mediaType).toBe('video')
    })

    it('filters non-previewable outputs and non-object items', async () => {
      const { getPreviewableOutputsFromJobDetail } =
        await import('@/services/jobOutputCache')
      const jobDetail: JobDetail = {
        id: 'job-filter',
        status: 'completed',
        create_time: Date.now(),
        priority: 0,
        outputs: {
          'node-3': {
            images: [{ filename: 'valid.png', subfolder: '', type: 'output' }],
            text: ['not-object'],
            unknown: [{ filename: 'data.bin', subfolder: '', type: 'output' }]
          }
        }
      }

      const result = getPreviewableOutputsFromJobDetail(jobDetail)

      expect(result.map((item) => item.filename)).toEqual(['valid.png'])
    })
  })

  describe('getJobDetail', () => {
    it('fetches and caches job detail', async () => {
      const jobId = uniqueId('job')

      const mockDetail: JobDetail = {
        id: jobId,
        status: 'completed',
        create_time: Date.now(),
        priority: 0,
        outputs: {}
      }
      vi.mocked(api.getJobDetail).mockResolvedValue(mockDetail)

      const result = await getJobDetail(jobId)

      expect(result).toEqual(mockDetail)
      expect(api.getJobDetail).toHaveBeenCalledWith(jobId)
    })

    it('returns cached job detail on subsequent calls', async () => {
      const jobId = uniqueId('job')

      const mockDetail: JobDetail = {
        id: jobId,
        status: 'completed',
        create_time: Date.now(),
        priority: 0,
        outputs: {}
      }
      vi.mocked(api.getJobDetail).mockResolvedValue(mockDetail)

      // First call
      await getJobDetail(jobId)
      expect(api.getJobDetail).toHaveBeenCalledTimes(1)

      // Second call should use cache
      const result = await getJobDetail(jobId)
      expect(result).toEqual(mockDetail)
      expect(api.getJobDetail).toHaveBeenCalledTimes(1)
    })

    it('returns undefined on fetch error', async () => {
      const jobId = uniqueId('job-error')

      vi.mocked(api.getJobDetail).mockRejectedValue(new Error('Network error'))

      const result = await getJobDetail(jobId)

      expect(result).toBeUndefined()
    })
  })

  describe('getJobWorkflow', () => {
    it('fetches job detail and extracts workflow', async () => {
      const jobId = uniqueId('job-wf')

      const mockDetail: JobDetail = {
        id: jobId,
        status: 'completed',
        create_time: Date.now(),
        priority: 0,
        outputs: {}
      }
      const mockWorkflow = { version: 1 } as Partial<ComfyWorkflowJSON>

      vi.mocked(api.getJobDetail).mockResolvedValue(mockDetail)
      vi.mocked(extractWorkflow).mockResolvedValue(
        mockWorkflow as ComfyWorkflowJSON
      )

      const result = await getJobWorkflow(jobId)

      expect(result).toEqual(mockWorkflow)
      expect(extractWorkflow).toHaveBeenCalledWith(mockDetail)
    })

    it('returns undefined when job detail not found', async () => {
      const jobId = uniqueId('missing')

      vi.mocked(api.getJobDetail).mockResolvedValue(undefined)
      vi.mocked(extractWorkflow).mockResolvedValue(undefined)

      const result = await getJobWorkflow(jobId)

      expect(result).toBeUndefined()
    })
  })
})
