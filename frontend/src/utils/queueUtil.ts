import type { TaskItemImpl } from '@/stores/queueStore'
import type { JobState } from '@/types/queue'

/**
 * Checks if a job state represents an active (in-progress) job.
 */
export function isActiveJobState(state: JobState): boolean {
  return (
    state === 'pending' || state === 'initialization' || state === 'running'
  )
}

/**
 * Map a task to a UI job state, including initialization override.
 *
 * @param task Task item from the queue store
 * @param isInitializing True if the prompt is currently initializing
 * @returns JobState for UI
 */
export const jobStateFromTask = (
  task: TaskItemImpl,
  isInitializing: boolean
): JobState => {
  if (isInitializing) return 'initialization'
  const status = task.displayStatus
  switch (status) {
    case 'Running':
      return 'running'
    case 'Pending':
      return 'pending'
    case 'Completed':
      return 'completed'
    case 'Failed':
    case 'Cancelled':
      return 'failed'
  }
  return 'failed'
}
