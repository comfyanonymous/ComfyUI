import { mount } from '@vue/test-utils'
import { ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import ActiveJobCard from './ActiveMediaAssetCard.vue'

import type { JobListItem } from '@/composables/queue/useJobList'

const mockRunCancelJob = vi.fn()
const mockRunDeleteJob = vi.fn()
const mockCanCancelJob = ref(false)
const mockCanDeleteJob = ref(false)

vi.mock('@/composables/queue/useJobActions', () => ({
  useJobActions: () => ({
    cancelAction: {
      icon: 'icon-[lucide--x]',
      label: 'Cancel',
      variant: 'destructive'
    },
    canCancelJob: mockCanCancelJob,
    runCancelJob: mockRunCancelJob,
    deleteAction: {
      icon: 'icon-[lucide--circle-minus]',
      label: 'Remove job',
      variant: 'destructive'
    },
    canDeleteJob: mockCanDeleteJob,
    runDeleteJob: mockRunDeleteJob
  })
}))

vi.mock('@/composables/useProgressBarBackground', () => ({
  useProgressBarBackground: () => ({
    progressBarPrimaryClass: 'bg-blue-500',
    hasProgressPercent: (val: number | undefined) => typeof val === 'number',
    progressPercentStyle: (val: number) => ({ width: `${val}%` })
  })
}))

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      sideToolbar: {
        activeJobStatus: 'Active job: {status}'
      }
    }
  }
})

const createJob = (overrides: Partial<JobListItem> = {}): JobListItem => ({
  id: 'test-job-1',
  title: 'Running...',
  meta: 'Step 5/10',
  state: 'running',
  progressTotalPercent: 50,
  progressCurrentPercent: 75,
  ...overrides
})

const mountComponent = (job: JobListItem) =>
  mount(ActiveJobCard, {
    props: { job },
    global: {
      plugins: [i18n]
    }
  })

describe('ActiveJobCard', () => {
  beforeEach(() => {
    mockCanCancelJob.value = false
    mockCanDeleteJob.value = false
    mockRunCancelJob.mockReset()
    mockRunDeleteJob.mockReset()
  })

  it('displays percentage and progress bar when job is running', () => {
    const wrapper = mountComponent(
      createJob({ state: 'running', progressTotalPercent: 65 })
    )

    expect(wrapper.text()).toContain('65%')
    const progressBar = wrapper.find('.bg-blue-500')
    expect(progressBar.exists()).toBe(true)
    expect(progressBar.attributes('style')).toContain('width: 65%')
  })

  it('displays status text when job is pending', () => {
    const wrapper = mountComponent(
      createJob({
        state: 'pending',
        title: 'In queue...',
        progressTotalPercent: undefined
      })
    )

    expect(wrapper.text()).toContain('In queue...')
    const progressBar = wrapper.find('.bg-blue-500')
    expect(progressBar.exists()).toBe(false)
  })

  it('shows spinner for pending state', () => {
    const wrapper = mountComponent(createJob({ state: 'pending' }))

    const spinner = wrapper.find('.icon-\\[lucide--loader-circle\\]')
    expect(spinner.exists()).toBe(true)
    expect(spinner.classes()).toContain('animate-spin')
  })

  it('shows error icon for failed state', () => {
    const wrapper = mountComponent(
      createJob({ state: 'failed', title: 'Failed' })
    )

    const errorIcon = wrapper.find('.icon-\\[lucide--circle-alert\\]')
    expect(errorIcon.exists()).toBe(true)
    expect(wrapper.text()).toContain('Failed')
  })

  it('shows preview image when running with iconImageUrl', () => {
    const wrapper = mountComponent(
      createJob({
        state: 'running',
        iconImageUrl: 'https://example.com/preview.jpg'
      })
    )

    const img = wrapper.find('img')
    expect(img.exists()).toBe(true)
    expect(img.attributes('src')).toBe('https://example.com/preview.jpg')
  })

  it('has proper accessibility attributes', () => {
    const wrapper = mountComponent(createJob({ title: 'Generating...' }))

    const container = wrapper.find('[role="status"]')
    expect(container.exists()).toBe(true)
    expect(container.attributes('aria-label')).toBe('Active job: Generating...')
  })

  it('shows delete button on hover for failed jobs', async () => {
    mockCanDeleteJob.value = true

    const wrapper = mountComponent(
      createJob({ state: 'failed', title: 'Failed' })
    )

    expect(wrapper.findComponent({ name: 'Button' }).exists()).toBe(false)

    await wrapper.find('[role="status"]').trigger('mouseenter')

    const button = wrapper.findComponent({ name: 'Button' })
    expect(button.exists()).toBe(true)
    expect(button.attributes('aria-label')).toBe('Remove job')
  })

  it('calls runDeleteJob when delete button is clicked on a failed job', async () => {
    mockCanDeleteJob.value = true

    const wrapper = mountComponent(
      createJob({ state: 'failed', title: 'Failed' })
    )

    await wrapper.find('[role="status"]').trigger('mouseenter')

    const button = wrapper.findComponent({ name: 'Button' })
    await button.trigger('click')

    expect(mockRunDeleteJob).toHaveBeenCalledOnce()
  })

  it('does not show action button when job cannot be cancelled or deleted', async () => {
    const wrapper = mountComponent(
      createJob({ state: 'running', progressTotalPercent: 50 })
    )

    await wrapper.find('[role="status"]').trigger('mouseenter')

    expect(wrapper.findComponent({ name: 'Button' }).exists()).toBe(false)
  })

  it('shows cancel button on hover for cancellable jobs', async () => {
    mockCanCancelJob.value = true

    const wrapper = mountComponent(
      createJob({ state: 'running', progressTotalPercent: 50 })
    )

    await wrapper.find('[role="status"]').trigger('mouseenter')

    const button = wrapper.findComponent({ name: 'Button' })
    expect(button.exists()).toBe(true)
    expect(button.attributes('aria-label')).toBe('Cancel')
  })

  it('calls runCancelJob when cancel button is clicked', async () => {
    mockCanCancelJob.value = true

    const wrapper = mountComponent(
      createJob({ state: 'running', progressTotalPercent: 50 })
    )

    await wrapper.find('[role="status"]').trigger('mouseenter')

    const button = wrapper.findComponent({ name: 'Button' })
    await button.trigger('click')

    expect(mockRunCancelJob).toHaveBeenCalledOnce()
  })
})
