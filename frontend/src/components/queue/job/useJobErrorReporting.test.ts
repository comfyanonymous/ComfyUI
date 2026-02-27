import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { computed, ref } from 'vue'
import type { ComputedRef } from 'vue'

import type { TaskItemImpl } from '@/stores/queueStore'
import type { JobErrorDialogService } from '@/components/queue/job/useJobErrorReporting'
import { useJobErrorReporting } from '@/components/queue/job/useJobErrorReporting'
import type { ExecutionError } from '@/platform/remote/comfyui/jobs/jobTypes'

const createTaskWithError = (
  jobId: string,
  errorMessage?: string,
  executionError?: ExecutionError,
  createTime?: number
): TaskItemImpl =>
  ({
    jobId,
    errorMessage,
    executionError,
    createTime: createTime ?? Date.now()
  }) as Partial<TaskItemImpl> as TaskItemImpl

describe('useJobErrorReporting', () => {
  let taskState = ref<TaskItemImpl | null>(null)
  let taskForJob: ComputedRef<TaskItemImpl | null>
  let copyToClipboard: ReturnType<typeof vi.fn>
  let showErrorDialog: ReturnType<typeof vi.fn>
  let showExecutionErrorDialog: ReturnType<typeof vi.fn>
  let dialog: JobErrorDialogService
  let composable: ReturnType<typeof useJobErrorReporting>

  beforeEach(() => {
    vi.clearAllMocks()
    taskState = ref<TaskItemImpl | null>(null)
    taskForJob = computed(() => taskState.value)
    copyToClipboard = vi.fn()
    showErrorDialog = vi.fn()
    showExecutionErrorDialog = vi.fn()
    dialog = {
      showErrorDialog,
      showExecutionErrorDialog
    } as Partial<JobErrorDialogService> as JobErrorDialogService
    composable = useJobErrorReporting({
      taskForJob,
      copyToClipboard: copyToClipboard as (
        value: string
      ) => void | Promise<void>,
      dialog
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('exposes a computed message that reflects the current task error', () => {
    taskState.value = createTaskWithError('job-1', 'First failure')
    expect(composable.errorMessageValue.value).toBe('First failure')

    taskState.value = createTaskWithError('job-2', 'Second failure')
    expect(composable.errorMessageValue.value).toBe('Second failure')
  })

  it('returns empty string when no error message', () => {
    taskState.value = createTaskWithError('job-1')
    expect(composable.errorMessageValue.value).toBe('')
  })

  it('returns empty string when task is null', () => {
    taskState.value = null
    expect(composable.errorMessageValue.value).toBe('')
  })

  it('only calls the copy handler when a message exists', () => {
    taskState.value = createTaskWithError('job-1', 'Clipboard failure')
    composable.copyErrorMessage()
    expect(copyToClipboard).toHaveBeenCalledTimes(1)
    expect(copyToClipboard).toHaveBeenCalledWith('Clipboard failure')

    copyToClipboard.mockClear()
    taskState.value = createTaskWithError('job-2')
    composable.copyErrorMessage()
    expect(copyToClipboard).not.toHaveBeenCalled()
  })

  it('shows simple error dialog when only errorMessage present', () => {
    taskState.value = createTaskWithError('job-1', 'Queue job error')
    composable.reportJobError()

    expect(showErrorDialog).toHaveBeenCalledTimes(1)
    const [errorArg, optionsArg] = showErrorDialog.mock.calls[0]
    expect(errorArg).toBeInstanceOf(Error)
    expect(errorArg.message).toBe('Queue job error')
    expect(optionsArg).toEqual({ reportType: 'queueJobError' })
    expect(showExecutionErrorDialog).not.toHaveBeenCalled()
  })

  it('does nothing when no task exists', () => {
    taskState.value = null
    composable.reportJobError()
    expect(showErrorDialog).not.toHaveBeenCalled()
    expect(showExecutionErrorDialog).not.toHaveBeenCalled()
  })

  it('shows rich error dialog when execution_error available on task', () => {
    const executionError: ExecutionError = {
      prompt_id: 'job-1',
      timestamp: 12345,
      node_id: '5',
      node_type: 'KSampler',
      executed: ['1', '2'],
      exception_message: 'CUDA out of memory',
      exception_type: 'RuntimeError',
      traceback: ['line 1', 'line 2'],
      current_inputs: {},
      current_outputs: {}
    }
    taskState.value = createTaskWithError(
      'job-1',
      'CUDA out of memory',
      executionError,
      12345
    )

    composable.reportJobError()

    expect(showExecutionErrorDialog).toHaveBeenCalledTimes(1)
    expect(showExecutionErrorDialog).toHaveBeenCalledWith(executionError)
    expect(showErrorDialog).not.toHaveBeenCalled()
  })

  it('does nothing when no error message and no execution_error', () => {
    taskState.value = createTaskWithError('job-1')

    composable.reportJobError()

    expect(showErrorDialog).not.toHaveBeenCalled()
    expect(showExecutionErrorDialog).not.toHaveBeenCalled()
  })
})
