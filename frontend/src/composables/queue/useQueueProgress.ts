import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { useExecutionStore } from '@/stores/executionStore'
import { clampPercentInt, formatPercent0 } from '@/utils/numberUtil'

/**
 * Queue progress composable exposing total/current node progress values and styles.
 */
export function useQueueProgress() {
  const { locale } = useI18n()
  const executionStore = useExecutionStore()

  const totalPercent = computed(() =>
    clampPercentInt(Math.round((executionStore.executionProgress ?? 0) * 100))
  )

  const totalPercentFormatted = computed(() =>
    formatPercent0(locale.value, totalPercent.value)
  )

  const currentNodePercent = computed(() =>
    clampPercentInt(
      Math.round((executionStore.executingNodeProgress ?? 0) * 100)
    )
  )

  const currentNodePercentFormatted = computed(() =>
    formatPercent0(locale.value, currentNodePercent.value)
  )

  const totalProgressStyle = computed(() => ({
    width: `${totalPercent.value}%`,
    background: 'var(--color-interface-panel-job-progress-primary)'
  }))

  const currentNodeProgressStyle = computed(() => ({
    width: `${currentNodePercent.value}%`,
    background: 'var(--color-interface-panel-job-progress-secondary)'
  }))

  return {
    totalPercent,
    totalPercentFormatted,
    currentNodePercent,
    currentNodePercentFormatted,
    totalProgressStyle,
    currentNodeProgressStyle
  }
}
