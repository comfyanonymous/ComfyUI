import { describe, expect, it, vi } from 'vitest'

import {
  extractWorkflow,
  fetchHistory,
  fetchJobDetail,
  fetchQueue
} from '@/platform/remote/comfyui/jobs/fetchJobs'
import type {
  RawJobListItem,
  zJobsListResponse
} from '@/platform/remote/comfyui/jobs/jobTypes'
import type { z } from 'zod'

type JobsListResponse = z.infer<typeof zJobsListResponse>

function createMockJob(
  id: string,
  status: 'pending' | 'in_progress' | 'completed' | 'failed' = 'completed',
  overrides: Partial<RawJobListItem> = {}
): RawJobListItem {
  return {
    id,
    status,
    create_time: Date.now(),
    ...overrides
  }
}

function createMockResponse(
  jobs: RawJobListItem[],
  total: number = jobs.length
): JobsListResponse {
  return {
    jobs,
    pagination: {
      offset: 0,
      limit: 200,
      total,
      has_more: false
    }
  }
}

describe('fetchJobs', () => {
  describe('fetchHistory', () => {
    it('fetches completed jobs', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve(
            createMockResponse([
              createMockJob('job1', 'completed'),
              createMockJob('job2', 'completed')
            ])
          )
      })

      const result = await fetchHistory(mockFetch)

      expect(mockFetch).toHaveBeenCalledWith(
        '/jobs?status=completed,failed,cancelled&limit=200&offset=0'
      )
      expect(result).toHaveLength(2)
      expect(result[0].id).toBe('job1')
      expect(result[1].id).toBe('job2')
    })

    it('assigns synthetic priorities', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve(
            createMockResponse(
              [
                createMockJob('job1', 'completed'),
                createMockJob('job2', 'completed'),
                createMockJob('job3', 'completed')
              ],
              3
            )
          )
      })

      const result = await fetchHistory(mockFetch)

      // Priority should be assigned from total down
      expect(result[0].priority).toBe(3) // total - 0 - 0
      expect(result[1].priority).toBe(2) // total - 0 - 1
      expect(result[2].priority).toBe(1) // total - 0 - 2
    })

    it('calculates priority correctly with non-zero offset', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve(
            createMockResponse(
              [
                createMockJob('job4', 'completed'),
                createMockJob('job5', 'completed')
              ],
              10 // total of 10 jobs
            )
          )
      })

      // Fetch page 2 (offset=5)
      const result = await fetchHistory(mockFetch, 200, 5)

      expect(mockFetch).toHaveBeenCalledWith(
        '/jobs?status=completed,failed,cancelled&limit=200&offset=5'
      )
      // Priority base is total - offset = 10 - 5 = 5
      expect(result[0].priority).toBe(5) // (total - offset) - 0
      expect(result[1].priority).toBe(4) // (total - offset) - 1
    })

    it('preserves server-provided priority', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve(
            createMockResponse([
              createMockJob('job1', 'completed', { priority: 999 })
            ])
          )
      })

      const result = await fetchHistory(mockFetch)

      expect(result[0].priority).toBe(999)
    })

    it('returns empty array on error', async () => {
      const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'))

      const result = await fetchHistory(mockFetch)

      expect(result).toEqual([])
    })

    it('returns empty array on non-ok response', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500
      })

      const result = await fetchHistory(mockFetch)

      expect(result).toEqual([])
    })
  })

  describe('fetchQueue', () => {
    it('fetches running and pending jobs', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve(
            createMockResponse([
              createMockJob('running1', 'in_progress'),
              createMockJob('pending1', 'pending'),
              createMockJob('pending2', 'pending')
            ])
          )
      })

      const result = await fetchQueue(mockFetch)

      expect(mockFetch).toHaveBeenCalledWith(
        '/jobs?status=in_progress,pending&limit=200&offset=0'
      )
      expect(result.Running).toHaveLength(1)
      expect(result.Pending).toHaveLength(2)
      expect(result.Running[0].id).toBe('running1')
      expect(result.Pending[0].id).toBe('pending1')
    })

    it('assigns queue priorities above history', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve(
            createMockResponse([
              createMockJob('running1', 'in_progress'),
              createMockJob('pending1', 'pending')
            ])
          )
      })

      const result = await fetchQueue(mockFetch)

      // Queue priorities should be above 1_000_000 (QUEUE_PRIORITY_BASE)
      expect(result.Running[0].priority).toBeGreaterThan(1_000_000)
      expect(result.Pending[0].priority).toBeGreaterThan(1_000_000)
      // Pending should have higher priority than running
      expect(result.Pending[0].priority).toBeGreaterThan(
        result.Running[0].priority
      )
    })

    it('returns empty arrays on error', async () => {
      const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'))

      const result = await fetchQueue(mockFetch)

      expect(result).toEqual({ Running: [], Pending: [] })
    })
  })

  describe('fetchJobDetail', () => {
    it('fetches job detail by id', async () => {
      const jobDetail = {
        ...createMockJob('job1', 'completed'),
        workflow: { extra_data: { extra_pnginfo: { workflow: {} } } },
        outputs: {
          '1': {
            images: [{ filename: 'test.png', subfolder: '', type: 'output' }]
          }
        }
      }
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(jobDetail)
      })

      const result = await fetchJobDetail(mockFetch, 'job1')

      expect(mockFetch).toHaveBeenCalledWith('/jobs/job1')
      expect(result?.id).toBe('job1')
      expect(result?.outputs).toBeDefined()
    })

    it('returns undefined for non-ok response', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 404
      })

      const result = await fetchJobDetail(mockFetch, 'nonexistent')

      expect(result).toBeUndefined()
    })

    it('returns undefined on error', async () => {
      const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'))

      const result = await fetchJobDetail(mockFetch, 'job1')

      expect(result).toBeUndefined()
    })
  })

  describe('extractWorkflow', () => {
    const validWorkflow = {
      version: 0.4,
      last_node_id: 1,
      last_link_id: 0,
      nodes: [],
      links: []
    }

    it('extracts and validates workflow from nested structure', async () => {
      const jobDetail = {
        ...createMockJob('job1', 'completed'),
        workflow: {
          extra_data: {
            extra_pnginfo: {
              workflow: validWorkflow
            }
          }
        }
      }

      const workflow = await extractWorkflow(jobDetail)

      expect(workflow).toEqual(validWorkflow)
    })

    it('returns undefined if workflow not present', async () => {
      const jobDetail = createMockJob('job1', 'completed')

      const workflow = await extractWorkflow(jobDetail)

      expect(workflow).toBeUndefined()
    })

    it('returns undefined for undefined input', async () => {
      const workflow = await extractWorkflow(undefined)

      expect(workflow).toBeUndefined()
    })

    it('returns undefined for invalid workflow and logs warning', async () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const jobDetail = {
        ...createMockJob('job1', 'completed'),
        workflow: {
          extra_data: {
            extra_pnginfo: {
              workflow: { invalid: 'data' }
            }
          }
        }
      }

      const workflow = await extractWorkflow(jobDetail)

      expect(workflow).toBeUndefined()
      expect(consoleSpy).toHaveBeenCalledWith(
        '[extractWorkflow] Workflow validation failed:',
        expect.any(String)
      )
      consoleSpy.mockRestore()
    })
  })
})
