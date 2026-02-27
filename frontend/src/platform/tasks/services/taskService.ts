/**
 * Task Service for polling background task status.
 *
 * CAVEAT: The `payload` and `result` schemas below are specific to
 * `task:download_file` tasks. Other task types may have different
 * payload/result structures. We are not generalizing this until
 * additional use cases arise.
 */
import { z } from 'zod'
import { fromZodError } from 'zod-validation-error'

import { api } from '@/scripts/api'

const TASKS_ENDPOINT = '/tasks'

const zTaskStatus = z.enum(['created', 'running', 'completed', 'failed'])

const zDownloadFileResult = z.object({
  success: z.boolean(),
  file_path: z.string().optional(),
  bytes_downloaded: z.number().optional(),
  content_type: z.string().optional(),
  hash: z.string().optional(),
  filename: z.string().optional(),
  asset_id: z.string().optional(),
  metadata: z.record(z.unknown()).optional(),
  error: z.string().optional()
})

const zTaskResponse = z.object({
  id: z.string().uuid(),
  idempotency_key: z.string(),
  task_name: z.string(),
  payload: z.record(z.unknown()),
  status: zTaskStatus,
  result: zDownloadFileResult.optional(),
  error_message: z.string().optional(),
  create_time: z.string().datetime(),
  update_time: z.string().datetime(),
  started_at: z.string().datetime().optional(),
  completed_at: z.string().datetime().optional()
})

export type TaskResponse = z.infer<typeof zTaskResponse>

function createTaskService() {
  async function getTask(taskId: string): Promise<TaskResponse> {
    const res = await api.fetchApi(`${TASKS_ENDPOINT}/${taskId}`)

    if (!res.ok) {
      if (res.status === 404) {
        throw new Error(`Task not found: ${taskId}`)
      }
      throw new Error(`Failed to get task ${taskId}: ${res.status}`)
    }

    const data = await res.json()
    const result = zTaskResponse.safeParse(data)

    if (!result.success) {
      throw new Error(fromZodError(result.error).message)
    }

    return result.data
  }

  return { getTask }
}

export const taskService = createTaskService()
