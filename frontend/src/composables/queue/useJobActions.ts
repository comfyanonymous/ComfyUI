import { computed, toValue } from 'vue'
import type { MaybeRefOrGetter } from 'vue'
import { useI18n } from 'vue-i18n'

import { useErrorHandling } from '@/composables/useErrorHandling'
import type { JobListItem } from '@/composables/queue/useJobList'
import { useJobMenu } from '@/composables/queue/useJobMenu'
import type { TaskItemImpl } from '@/stores/queueStore'
import { isActiveJobState } from '@/utils/queueUtil'

export type JobAction = {
  icon: string
  label: string
  variant: 'destructive' | 'secondary' | 'textonly'
}

export function useJobActions(
  job: MaybeRefOrGetter<JobListItem | null | undefined>
) {
  const { t } = useI18n()
  const { wrapWithErrorHandlingAsync } = useErrorHandling()
  const { cancelJob, removeFailedJob } = useJobMenu()

  const cancelAction: JobAction = {
    icon: 'icon-[lucide--x]',
    label: t('sideToolbar.queueProgressOverlay.cancelJobTooltip'),
    variant: 'destructive'
  }

  const deleteAction: JobAction = {
    icon: 'icon-[lucide--circle-minus]',
    label: t('queue.jobMenu.removeJob'),
    variant: 'destructive'
  }

  const jobRef = computed(() => toValue(job) ?? null)

  const canCancelJob = computed(() => {
    const currentJob = jobRef.value
    if (!currentJob) {
      return false
    }

    return currentJob.showClear !== false && isActiveJobState(currentJob.state)
  })

  const canDeleteJob = computed(() => {
    const currentJob = jobRef.value
    if (!currentJob) {
      return false
    }

    return currentJob.state === 'failed'
  })

  const runCancelJob = wrapWithErrorHandlingAsync(async () => {
    const currentJob = jobRef.value
    if (!currentJob) {
      return
    }

    await cancelJob(currentJob)
  })

  const runDeleteJob = wrapWithErrorHandlingAsync(async () => {
    const currentJob = jobRef.value
    const task = currentJob?.taskRef as TaskItemImpl | undefined
    if (!task) {
      return
    }

    await removeFailedJob(task)
  })

  return {
    cancelAction,
    canCancelJob,
    runCancelJob,
    deleteAction,
    canDeleteJob,
    runDeleteJob
  }
}
