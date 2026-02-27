import { mount } from '@vue/test-utils'
import type { VueWrapper } from '@vue/test-utils'
import { nextTick, ref } from 'vue'
import type { Ref } from 'vue'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'

import { useQueueProgress } from '@/composables/queue/useQueueProgress'
import { formatPercent0 } from '@/utils/numberUtil'

type ProgressValue = number | null

const localeRef: Ref<string> = ref('en-US') as Ref<string>
const executionProgressRef: Ref<ProgressValue> = ref(null)
const executingNodeProgressRef: Ref<ProgressValue> = ref(null)

const createExecutionStoreMock = () => ({
  get executionProgress() {
    return executionProgressRef.value ?? undefined
  },
  get executingNodeProgress() {
    return executingNodeProgressRef.value ?? undefined
  }
})

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    locale: localeRef
  })
}))

vi.mock('@/stores/executionStore', () => ({
  useExecutionStore: () => createExecutionStoreMock()
}))

const mountedWrappers: VueWrapper[] = []

const mountUseQueueProgress = () => {
  let composable: ReturnType<typeof useQueueProgress>
  const wrapper = mount({
    template: '<div />',
    setup() {
      composable = useQueueProgress()
      return {}
    }
  })
  mountedWrappers.push(wrapper)
  return { wrapper, composable: composable! }
}

const setExecutionProgress = (value?: number | null) => {
  executionProgressRef.value = value ?? null
}

const setExecutingNodeProgress = (value?: number | null) => {
  executingNodeProgressRef.value = value ?? null
}

describe('useQueueProgress', () => {
  beforeEach(() => {
    localeRef.value = 'en-US'
    setExecutionProgress(null)
    setExecutingNodeProgress(null)
  })

  afterEach(() => {
    mountedWrappers.splice(0).forEach((wrapper) => wrapper.unmount())
  })

  it.each([
    {
      description: 'defaults to 0% when execution store values are missing',
      execution: undefined,
      node: undefined,
      expectedTotal: 0,
      expectedNode: 0
    },
    {
      description: 'rounds fractional progress to the nearest integer',
      execution: 0.324,
      node: 0.005,
      expectedTotal: 32,
      expectedNode: 1
    },
    {
      description: 'clamps values below 0 and above 100%',
      execution: 1.5,
      node: -0.25,
      expectedTotal: 100,
      expectedNode: 0
    },
    {
      description: 'caps near-complete totals at 100%',
      execution: 0.999,
      node: 0.731,
      expectedTotal: 100,
      expectedNode: 73
    }
  ])('$description', ({ execution, node, expectedTotal, expectedNode }) => {
    setExecutionProgress(execution ?? null)
    setExecutingNodeProgress(node ?? null)

    const { composable } = mountUseQueueProgress()

    expect(composable.totalPercent.value).toBe(expectedTotal)
    expect(composable.currentNodePercent.value).toBe(expectedNode)
    expect(composable.totalPercentFormatted.value).toBe(
      formatPercent0(localeRef.value, expectedTotal)
    )
    expect(composable.currentNodePercentFormatted.value).toBe(
      formatPercent0(localeRef.value, expectedNode)
    )
  })

  it('reformats output when the active locale changes', async () => {
    setExecutionProgress(0.32)
    setExecutingNodeProgress(0.58)

    const { composable } = mountUseQueueProgress()

    expect(composable.totalPercentFormatted.value).toBe(
      formatPercent0('en-US', composable.totalPercent.value)
    )
    expect(composable.currentNodePercentFormatted.value).toBe(
      formatPercent0('en-US', composable.currentNodePercent.value)
    )

    localeRef.value = 'fr-FR'
    await nextTick()

    expect(composable.totalPercentFormatted.value).toBe(
      formatPercent0('fr-FR', composable.totalPercent.value)
    )
    expect(composable.currentNodePercentFormatted.value).toBe(
      formatPercent0('fr-FR', composable.currentNodePercent.value)
    )
  })

  it('builds progress bar styles that track store updates', async () => {
    setExecutionProgress(0.1)
    setExecutingNodeProgress(0.25)

    const { composable } = mountUseQueueProgress()

    expect(composable.totalProgressStyle.value).toEqual({
      width: '10%',
      background: 'var(--color-interface-panel-job-progress-primary)'
    })
    expect(composable.currentNodeProgressStyle.value).toEqual({
      width: '25%',
      background: 'var(--color-interface-panel-job-progress-secondary)'
    })

    setExecutionProgress(0.755)
    setExecutingNodeProgress(0.02)
    await nextTick()

    expect(composable.totalProgressStyle.value.width).toBe('76%')
    expect(composable.currentNodeProgressStyle.value.width).toBe('2%')
  })
})
