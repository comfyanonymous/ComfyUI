import { describe, expect, it, vi } from 'vitest'

import type { JobDetail } from '@/platform/remote/comfyui/jobs/jobTypes'
import type { ComfyWorkflowJSON } from '@/platform/workflow/validation/schemas/workflowSchema'
import {
  extractWorkflow,
  fetchJobDetail
} from '@/platform/remote/comfyui/jobs/fetchJobs'

const mockWorkflow: ComfyWorkflowJSON = {
  last_node_id: 5,
  last_link_id: 3,
  nodes: [],
  links: [],
  groups: [],
  config: {},
  extra: {},
  version: 0.4
}

// Jobs API detail response structure (matches actual /jobs/{id} response)
// workflow is nested at: workflow.extra_data.extra_pnginfo.workflow
const mockJobDetailResponse: JobDetail = {
  id: 'test-job-id',
  status: 'completed',
  create_time: 1234567890,
  update_time: 1234567900,
  workflow: {
    extra_data: {
      extra_pnginfo: {
        workflow: mockWorkflow
      }
    }
  },
  outputs: {
    '20': {
      images: [
        { filename: 'test.png', subfolder: '', type: 'output' },
        { filename: 'test2.png', subfolder: '', type: 'output' }
      ]
    }
  }
}

describe('fetchJobDetail', () => {
  it('should fetch job detail from /jobs/{job_id} endpoint', async () => {
    const mockFetchApi = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockJobDetailResponse
    })

    await fetchJobDetail(mockFetchApi, 'test-job-id')

    expect(mockFetchApi).toHaveBeenCalledWith('/jobs/test-job-id')
  })

  it('should return job detail with workflow and outputs', async () => {
    const mockFetchApi = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockJobDetailResponse
    })

    const result = await fetchJobDetail(mockFetchApi, 'test-job-id')

    expect(result).toBeDefined()
    expect(result?.id).toBe('test-job-id')
    expect(result?.outputs).toEqual(mockJobDetailResponse.outputs)
    expect(result?.workflow).toBeDefined()
  })

  it('should return undefined when job not found (non-OK response)', async () => {
    const mockFetchApi = vi.fn().mockResolvedValue({
      ok: false,
      status: 404
    })

    const result = await fetchJobDetail(mockFetchApi, 'nonexistent-id')

    expect(result).toBeUndefined()
  })

  it('should handle fetch errors gracefully', async () => {
    const mockFetchApi = vi.fn().mockRejectedValue(new Error('Network error'))

    const result = await fetchJobDetail(mockFetchApi, 'test-job-id')

    expect(result).toBeUndefined()
  })

  it('should handle malformed JSON responses', async () => {
    const mockFetchApi = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => {
        throw new Error('Invalid JSON')
      }
    })

    const result = await fetchJobDetail(mockFetchApi, 'test-job-id')

    expect(result).toBeUndefined()
  })
})

describe('extractWorkflow', () => {
  it('should extract workflow from job detail', async () => {
    const result = await extractWorkflow(mockJobDetailResponse)

    expect(result).toEqual(mockWorkflow)
  })

  it('should return undefined when job is undefined', async () => {
    const result = await extractWorkflow(undefined)

    expect(result).toBeUndefined()
  })

  it('should return undefined when workflow is missing', async () => {
    const jobWithoutWorkflow: JobDetail = {
      ...mockJobDetailResponse,
      workflow: {}
    }

    const result = await extractWorkflow(jobWithoutWorkflow)

    expect(result).toBeUndefined()
  })
})
