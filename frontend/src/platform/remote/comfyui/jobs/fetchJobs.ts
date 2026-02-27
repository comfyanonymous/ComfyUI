/**
 * @fileoverview Jobs API Fetchers
 * @module platform/remote/comfyui/jobs/fetchJobs
 *
 * Unified jobs API fetcher for history, queue, and job details.
 * All distributions use the /jobs endpoint.
 */

import type { ComfyWorkflowJSON } from '@/platform/workflow/validation/schemas/workflowSchema'
import { validateComfyWorkflow } from '@/platform/workflow/validation/schemas/workflowSchema'
import type { JobId } from '@/schemas/apiSchema'

import type {
  JobDetail,
  JobListItem,
  JobStatus,
  RawJobListItem
} from './jobTypes'
import { zJobDetail, zJobsListResponse, zWorkflowContainer } from './jobTypes'

interface FetchJobsRawResult {
  jobs: RawJobListItem[]
  total: number
  offset: number
}

/**
 * Fetches raw jobs from /jobs endpoint
 * @internal
 */
async function fetchJobsRaw(
  fetchApi: (url: string) => Promise<Response>,
  statuses: JobStatus[],
  maxItems: number = 200,
  offset: number = 0
): Promise<FetchJobsRawResult> {
  const statusParam = statuses.join(',')
  const url = `/jobs?status=${statusParam}&limit=${maxItems}&offset=${offset}`
  try {
    const res = await fetchApi(url)
    if (!res.ok) {
      console.error(`[Jobs API] Failed to fetch jobs: ${res.status}`)
      return { jobs: [], total: 0, offset: 0 }
    }
    const data = zJobsListResponse.parse(await res.json())
    return { jobs: data.jobs, total: data.pagination.total, offset }
  } catch (error) {
    console.error('[Jobs API] Error fetching jobs:', error)
    return { jobs: [], total: 0, offset: 0 }
  }
}

// Large offset to ensure running/pending jobs sort above history
const QUEUE_PRIORITY_BASE = 1_000_000

/**
 * Assigns synthetic priority to jobs.
 * Only assigns if job doesn't already have a server-provided priority.
 */
function assignPriority(
  jobs: RawJobListItem[],
  basePriority: number
): JobListItem[] {
  return jobs.map((job, index) => ({
    ...job,
    priority: job.priority ?? basePriority - index
  }))
}

/**
 * Fetches history (terminal state jobs: completed, failed, cancelled)
 * Assigns synthetic priority starting from total (lower than queue jobs).
 */
export async function fetchHistory(
  fetchApi: (url: string) => Promise<Response>,
  maxItems: number = 200,
  offset: number = 0
): Promise<JobListItem[]> {
  const { jobs, total } = await fetchJobsRaw(
    fetchApi,
    ['completed', 'failed', 'cancelled'],
    maxItems,
    offset
  )
  // History gets priority based on total count (lower than queue)
  return assignPriority(jobs, total - offset)
}

/**
 * Fetches queue (in_progress + pending jobs)
 * Pending jobs get highest priority, then running jobs.
 */
export async function fetchQueue(
  fetchApi: (url: string) => Promise<Response>
): Promise<{ Running: JobListItem[]; Pending: JobListItem[] }> {
  const { jobs } = await fetchJobsRaw(
    fetchApi,
    ['in_progress', 'pending'],
    200,
    0
  )

  const running = jobs.filter((j) => j.status === 'in_progress')
  const pending = jobs.filter((j) => j.status === 'pending')

  // Pending gets highest priority, then running
  // Both are above any history job due to QUEUE_PRIORITY_BASE
  return {
    Running: assignPriority(running, QUEUE_PRIORITY_BASE + running.length),
    Pending: assignPriority(
      pending,
      QUEUE_PRIORITY_BASE + running.length + pending.length
    )
  }
}

/**
 * Fetches full job details from /jobs/{job_id}
 */
export async function fetchJobDetail(
  fetchApi: (url: string) => Promise<Response>,
  jobId: JobId
): Promise<JobDetail | undefined> {
  try {
    const res = await fetchApi(`/jobs/${encodeURIComponent(jobId)}`)

    if (!res.ok) {
      console.warn(`Job not found for job ${jobId}`)
      return undefined
    }

    return zJobDetail.parse(await res.json())
  } catch (error) {
    console.error(`Failed to fetch job detail for job ${jobId}:`, error)
    return undefined
  }
}

/**
 * Extracts and validates workflow from job detail response.
 * The workflow is nested at: workflow.extra_data.extra_pnginfo.workflow
 *
 * Uses Zod validation via validateComfyWorkflow to ensure the workflow
 * conforms to the expected schema. Logs validation failures for debugging
 * but still returns undefined to allow graceful degradation.
 */
export async function extractWorkflow(
  job: JobDetail | undefined
): Promise<ComfyWorkflowJSON | undefined> {
  const parsed = zWorkflowContainer.safeParse(job?.workflow)
  if (!parsed.success) return undefined

  const rawWorkflow = parsed.data.extra_data?.extra_pnginfo?.workflow
  if (!rawWorkflow) return undefined

  const validated = await validateComfyWorkflow(rawWorkflow, (error) => {
    console.warn('[extractWorkflow] Workflow validation failed:', error)
  })

  return validated ?? undefined
}
