/**
 * @fileoverview Jobs API types - Backend job API format
 * @module platform/remote/comfyui/jobs/jobTypes
 *
 * These types represent the jobs API format returned by the backend.
 * Jobs API provides a memory-optimized alternative to history API.
 */

import { z } from 'zod'

import { resultItemType, zTaskOutput } from '@/schemas/apiSchema'

const zJobStatus = z.enum([
  'pending',
  'in_progress',
  'completed',
  'failed',
  'cancelled'
])

const zPreviewOutput = z.object({
  filename: z.string(),
  subfolder: z.string(),
  type: resultItemType,
  nodeId: z.string(),
  mediaType: z.string()
})

/**
 * Execution error from Jobs API.
 * Similar to ExecutionErrorWsMessage but with optional prompt_id/timestamp/executed
 * since these may not be present in stored errors or infrastructure-generated errors.
 */
const zExecutionError = z
  .object({
    prompt_id: z.string().optional(),
    timestamp: z.number().optional(),
    node_id: z.string(),
    node_type: z.string(),
    executed: z.array(z.string()).optional(),
    exception_message: z.string(),
    exception_type: z.string(),
    traceback: z.array(z.string()),
    current_inputs: z.unknown(),
    current_outputs: z.unknown()
  })
  .passthrough()

export type ExecutionError = z.infer<typeof zExecutionError>

/**
 * Raw job from API - uses passthrough to allow extra fields
 */
const zRawJobListItem = z
  .object({
    id: z.string(),
    status: zJobStatus,
    create_time: z.number(),
    execution_start_time: z.number().nullable().optional(),
    execution_end_time: z.number().nullable().optional(),
    preview_output: zPreviewOutput.nullable().optional(),
    outputs_count: z.number().nullable().optional(),
    execution_error: zExecutionError.nullable().optional(),
    workflow_id: z.string().nullable().optional(),
    priority: z.number().optional()
  })
  .passthrough()

/**
 * Job detail - returned by GET /api/jobs/{job_id} (detail endpoint)
 * Includes full workflow and outputs for re-execution and downloads
 */
export const zJobDetail = zRawJobListItem
  .extend({
    workflow: z.unknown().optional(),
    outputs: zTaskOutput.optional(),
    update_time: z.number().optional(),
    execution_status: z.unknown().optional(),
    execution_meta: z.unknown().optional()
  })
  .passthrough()

const zPaginationInfo = z.object({
  offset: z.number(),
  limit: z.number(),
  total: z.number(),
  has_more: z.boolean()
})

export const zJobsListResponse = z.object({
  jobs: z.array(zRawJobListItem),
  pagination: zPaginationInfo
})

/** Schema for workflow container structure in job detail responses */
export const zWorkflowContainer = z.object({
  extra_data: z
    .object({
      extra_pnginfo: z
        .object({
          workflow: z.unknown()
        })
        .optional()
    })
    .optional()
})

export type JobStatus = z.infer<typeof zJobStatus>
export type RawJobListItem = z.infer<typeof zRawJobListItem>
/** Job list item with priority always set (server-provided or synthetic) */
export type JobListItem = RawJobListItem & { priority: number }
export type JobDetail = z.infer<typeof zJobDetail>

/** Task type used in the API (queue vs history endpoints) */
export type APITaskType = 'queue' | 'history'

/** Internal task type derived from job status for UI display */
export type TaskType = 'Running' | 'Pending' | 'History'
