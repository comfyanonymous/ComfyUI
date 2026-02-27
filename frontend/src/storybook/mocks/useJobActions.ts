import { computed, ref, toValue } from 'vue'
import type { MaybeRefOrGetter } from 'vue'

import type { JobAction } from '../../composables/queue/useJobActions'
import type { JobListItem } from '../../composables/queue/useJobList'
import { isActiveJobState } from '../../utils/queueUtil'

const actionsByJobId = ref<Record<string, JobAction[]>>({})
const cancelAction: JobAction = {
  icon: 'icon-[lucide--x]',
  label: 'Cancel',
  variant: 'destructive'
}

export function setMockJobActions(actions: Record<string, JobAction[]>) {
  actionsByJobId.value = actions
}

export function useJobActions(
  job?: MaybeRefOrGetter<JobListItem | null | undefined>
) {
  const jobRef = computed(() => (job ? (toValue(job) ?? null) : null))

  const canCancelJob = computed(() => {
    const currentJob = jobRef.value
    if (!currentJob) {
      return false
    }

    const configuredActions = actionsByJobId.value[currentJob.id]
    if (configuredActions) {
      return configuredActions.length > 0
    }

    return currentJob.showClear !== false && isActiveJobState(currentJob.state)
  })

  async function runCancelJob() {
    return undefined
  }

  return {
    cancelAction,
    canCancelJob,
    runCancelJob
  }
}
