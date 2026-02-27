import type { Meta, StoryObj } from '@storybook/vue3-vite'

import type {
  JobListItem,
  JobStatus
} from '@/platform/remote/comfyui/jobs/jobTypes'
import { useExecutionStore } from '@/stores/executionStore'
import { TaskItemImpl, useQueueStore } from '@/stores/queueStore'

import JobDetailsPopover from './JobDetailsPopover.vue'

const meta: Meta<typeof JobDetailsPopover> = {
  title: 'Queue/JobDetailsPopover',
  component: JobDetailsPopover,
  args: {
    workflowId: 'WF-1234'
  },
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'dark'
    }
  },
  globals: {
    theme: 'dark'
  }
}

export default meta
type Story = StoryObj<typeof meta>

function resetStores() {
  const queue = useQueueStore()
  const exec = useExecutionStore()

  queue.pendingTasks = []
  queue.runningTasks = []
  queue.historyTasks = []

  exec.nodeProgressStatesByJob = {}
}

function makeTask(
  id: string,
  priority: number,
  fields: Partial<JobListItem> & { status: JobStatus; create_time: number }
): TaskItemImpl {
  const job: JobListItem = {
    id,
    priority,
    last_state_update: null,
    update_time: fields.create_time,
    ...fields
  }
  return new TaskItemImpl(job)
}

function makePendingTask(
  id: string,
  priority: number,
  createTimeMs: number
): TaskItemImpl {
  return makeTask(id, priority, {
    status: 'pending',
    create_time: createTimeMs
  })
}

function makeRunningTask(
  id: string,
  priority: number,
  createTimeMs: number
): TaskItemImpl {
  return makeTask(id, priority, {
    status: 'in_progress',
    create_time: createTimeMs
  })
}

function makeRunningTaskWithStart(
  id: string,
  priority: number,
  startedSecondsAgo: number
): TaskItemImpl {
  const start = Date.now() - startedSecondsAgo * 1000
  return makeTask(id, priority, {
    status: 'in_progress',
    create_time: start - 5000,
    update_time: start
  })
}

function makeHistoryTask(
  id: string,
  priority: number,
  durationSec: number,
  ok: boolean,
  errorMessage?: string
): TaskItemImpl {
  const now = Date.now()
  const executionEndTime = now
  const executionStartTime = now - durationSec * 1000
  return makeTask(id, priority, {
    status: ok ? 'completed' : 'failed',
    create_time: executionStartTime - 5000,
    update_time: now,
    execution_start_time: executionStartTime,
    execution_end_time: executionEndTime,
    execution_error: errorMessage
      ? {
          prompt_id: id,
          timestamp: now,
          node_id: '1',
          node_type: 'ExampleNode',
          exception_message: errorMessage,
          exception_type: 'RuntimeError',
          traceback: [],
          current_inputs: {},
          current_outputs: {}
        }
      : undefined
  })
}

export const Queued: Story = {
  render: (args) => ({
    components: { JobDetailsPopover },
    setup() {
      resetStores()
      const queue = useQueueStore()
      const exec = useExecutionStore()

      const jobId = 'job-queued-1'
      const priority = 104

      // Current job in pending
      queue.pendingTasks = [
        makePendingTask(jobId, priority, Date.now() - 90_000)
      ]
      // Add some other pending jobs to give context
      queue.pendingTasks.push(
        makePendingTask('job-older-1', 100, Date.now() - 60_000)
      )
      queue.pendingTasks.push(
        makePendingTask('job-older-2', 101, Date.now() - 30_000)
      )

      // Queued at (in metadata on job tuple)

      // One running workflow
      exec.nodeProgressStatesByJob = {
        p1: {
          '1': {
            value: 1,
            max: 1,
            state: 'running',
            node_id: '1',
            prompt_id: 'p1'
          }
        }
      }

      return { args: { ...args, jobId } }
    },
    template: `
      <div style="padding: 12px; background: var(--color-charcoal-700); display:inline-block;">
        <JobDetailsPopover v-bind="args" />
      </div>
    `
  })
}

export const QueuedParallel: Story = {
  render: (args) => ({
    components: { JobDetailsPopover },
    setup() {
      resetStores()
      const queue = useQueueStore()
      const exec = useExecutionStore()

      const jobId = 'job-queued-parallel'
      const priority = 210

      // Current job in pending with some ahead
      queue.pendingTasks = [
        makePendingTask('job-ahead-1', 200, Date.now() - 180_000),
        makePendingTask('job-ahead-2', 205, Date.now() - 150_000),
        makePendingTask(jobId, priority, Date.now() - 120_000)
      ]

      // Seen 2 minutes ago - set via prompt metadata above

      // History durations for ETA (in seconds)
      queue.historyTasks = [
        makeHistoryTask('hist-1', 150, 25, true),
        makeHistoryTask('hist-2', 151, 40, true),
        makeHistoryTask('hist-3', 152, 60, true)
      ]

      // Two parallel workflows running
      exec.nodeProgressStatesByJob = {
        p1: {
          '1': {
            value: 1,
            max: 2,
            state: 'running',
            node_id: '1',
            prompt_id: 'p1'
          }
        },
        p2: {
          '2': {
            value: 1,
            max: 2,
            state: 'running',
            node_id: '2',
            prompt_id: 'p2'
          }
        }
      }

      return { args: { ...args, jobId } }
    },
    template: `
      <div style="padding: 12px; background: var(--color-charcoal-700); display:inline-block;">
        <JobDetailsPopover v-bind="args" />
      </div>
    `
  })
}

export const Running: Story = {
  render: (args) => ({
    components: { JobDetailsPopover },
    setup() {
      resetStores()
      const queue = useQueueStore()
      const exec = useExecutionStore()

      const jobId = 'job-running-1'
      const priority = 300
      queue.runningTasks = [
        makeRunningTask(jobId, priority, Date.now() - 65_000)
      ]
      queue.historyTasks = [
        makeHistoryTask('hist-r1', 250, 30, true),
        makeHistoryTask('hist-r2', 251, 45, true),
        makeHistoryTask('hist-r3', 252, 60, true)
      ]

      exec.nodeProgressStatesByJob = {
        p1: {
          '1': {
            value: 5,
            max: 10,
            state: 'running',
            node_id: '1',
            prompt_id: 'p1'
          }
        }
      }

      return { args: { ...args, jobId } }
    },
    template: `
      <div style="padding: 12px; background: var(--color-charcoal-700); display:inline-block;">
        <JobDetailsPopover v-bind="args" />
      </div>
    `
  })
}

export const QueuedZeroAheadSingleRunning: Story = {
  render: (args) => ({
    components: { JobDetailsPopover },
    setup() {
      resetStores()
      const queue = useQueueStore()
      const exec = useExecutionStore()

      const jobId = 'job-queued-zero-ahead-single'
      const priority = 510

      queue.pendingTasks = [
        makePendingTask(jobId, priority, Date.now() - 45_000)
      ]

      queue.historyTasks = [
        makeHistoryTask('hist-s1', 480, 30, true),
        makeHistoryTask('hist-s2', 481, 50, true),
        makeHistoryTask('hist-s3', 482, 80, true)
      ]

      queue.runningTasks = [makeRunningTaskWithStart('running-1', 505, 20)]

      exec.nodeProgressStatesByJob = {
        p1: {
          '1': {
            value: 1,
            max: 3,
            state: 'running',
            node_id: '1',
            prompt_id: 'p1'
          }
        }
      }

      return { args: { ...args, jobId } }
    },
    template: `
      <div style="padding: 12px; background: var(--color-charcoal-700); display:inline-block;">
        <JobDetailsPopover v-bind="args" />
      </div>
    `
  })
}

export const QueuedZeroAheadMultiRunning: Story = {
  render: (args) => ({
    components: { JobDetailsPopover },
    setup() {
      resetStores()
      const queue = useQueueStore()
      const exec = useExecutionStore()

      const jobId = 'job-queued-zero-ahead-multi'
      const priority = 520

      queue.pendingTasks = [
        makePendingTask(jobId, priority, Date.now() - 20_000)
      ]

      queue.historyTasks = [
        makeHistoryTask('hist-m1', 490, 40, true),
        makeHistoryTask('hist-m2', 491, 55, true),
        makeHistoryTask('hist-m3', 492, 70, true)
      ]

      queue.runningTasks = [
        makeRunningTaskWithStart('running-a', 506, 35),
        makeRunningTaskWithStart('running-b', 507, 10)
      ]

      exec.nodeProgressStatesByJob = {
        p1: {
          '1': {
            value: 2,
            max: 5,
            state: 'running',
            node_id: '1',
            prompt_id: 'p1'
          }
        },
        p2: {
          '2': {
            value: 3,
            max: 5,
            state: 'running',
            node_id: '2',
            prompt_id: 'p2'
          }
        }
      }

      return { args: { ...args, jobId } }
    },
    template: `
      <div style="padding: 12px; background: var(--color-charcoal-700); display:inline-block;">
        <JobDetailsPopover v-bind="args" />
      </div>
    `
  })
}

export const Completed: Story = {
  render: (args) => ({
    components: { JobDetailsPopover },
    setup() {
      resetStores()
      const queue = useQueueStore()

      const jobId = 'job-completed-1'
      const priority = 400
      queue.historyTasks = [makeHistoryTask(jobId, priority, 37, true)]

      return { args: { ...args, jobId } }
    },
    template: `
      <div style="padding: 12px; background: var(--color-charcoal-700); display:inline-block;">
        <JobDetailsPopover v-bind="args" />
      </div>
    `
  })
}

export const Failed: Story = {
  render: (args) => ({
    components: { JobDetailsPopover },
    setup() {
      resetStores()
      const queue = useQueueStore()

      const jobId = 'job-failed-1'
      const priority = 410
      queue.historyTasks = [
        makeHistoryTask(
          jobId,
          priority,
          12,
          false,
          'Example error: invalid inputs for node X'
        )
      ]
      // Show a queued-at time for the failed job via history extra_data (2 minutes ago)
      // Already set by makeHistoryTask using its start timestamp

      return { args: { ...args, jobId } }
    },
    template: `
      <div style="padding: 12px; background: var(--color-charcoal-700); display:inline-block;">
        <JobDetailsPopover v-bind="args" />
      </div>
    `
  })
}
