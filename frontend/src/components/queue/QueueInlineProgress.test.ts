import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, ref } from 'vue'
import type { Ref } from 'vue'

import QueueInlineProgress from '@/components/queue/QueueInlineProgress.vue'

const mockProgress = vi.hoisted(() => ({
  totalPercent: null! as Ref<number>,
  currentNodePercent: null! as Ref<number>
}))

vi.mock('@/composables/queue/useQueueProgress', () => ({
  useQueueProgress: () => ({
    totalPercent: mockProgress.totalPercent,
    currentNodePercent: mockProgress.currentNodePercent
  })
}))

const createWrapper = (props: { hidden?: boolean } = {}) =>
  mount(QueueInlineProgress, { props })

describe('QueueInlineProgress', () => {
  beforeEach(() => {
    mockProgress.totalPercent = ref(0)
    mockProgress.currentNodePercent = ref(0)
  })

  it('renders when total progress is non-zero', () => {
    mockProgress.totalPercent.value = 12

    const wrapper = createWrapper()

    expect(wrapper.find('[aria-hidden="true"]').exists()).toBe(true)
  })

  it('renders when current node progress is non-zero', () => {
    mockProgress.currentNodePercent.value = 33

    const wrapper = createWrapper()

    expect(wrapper.find('[aria-hidden="true"]').exists()).toBe(true)
  })

  it('does not render when hidden', () => {
    mockProgress.totalPercent.value = 45

    const wrapper = createWrapper({ hidden: true })

    expect(wrapper.find('[aria-hidden="true"]').exists()).toBe(false)
  })

  it('shows when progress becomes non-zero', async () => {
    const wrapper = createWrapper()

    expect(wrapper.find('[aria-hidden="true"]').exists()).toBe(false)

    mockProgress.totalPercent.value = 10
    await nextTick()
    expect(wrapper.find('[aria-hidden="true"]').exists()).toBe(true)
  })

  it('hides when progress returns to zero', async () => {
    mockProgress.totalPercent.value = 10

    const wrapper = createWrapper()

    expect(wrapper.find('[aria-hidden="true"]').exists()).toBe(true)

    mockProgress.totalPercent.value = 0
    mockProgress.currentNodePercent.value = 0
    await nextTick()
    expect(wrapper.find('[aria-hidden="true"]').exists()).toBe(false)
  })
})
