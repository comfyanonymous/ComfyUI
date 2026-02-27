/**
 * Job execution state used across queue UI components.
 */
export type JobState =
  | 'pending'
  | 'initialization'
  | 'running'
  | 'completed'
  | 'failed'
