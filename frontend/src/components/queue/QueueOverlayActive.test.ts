import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import QueueOverlayActive from './QueueOverlayActive.vue'
import * as tooltipConfig from '@/composables/useTooltipConfig'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      sideToolbar: {
        queueProgressOverlay: {
          total: 'Total: {percent}',
          currentNode: 'Current node:',
          running: 'running',
          interruptAll: 'Interrupt all running jobs',
          queuedSuffix: 'queued',
          clearQueued: 'Clear queued',
          viewAllJobs: 'View all jobs',
          cancelJobTooltip: 'Cancel job',
          clearQueueTooltip: 'Clear queue'
        }
      }
    }
  }
})

const tooltipDirectiveStub = {
  mounted: vi.fn(),
  updated: vi.fn()
}

const SELECTORS = {
  interruptAllButton: 'button[aria-label="Interrupt all running jobs"]',
  clearQueuedButton: 'button[aria-label="Clear queued"]',
  summaryRow: '.flex.items-center.gap-2',
  currentNodeRow: '.flex.items-center.gap-1.text-text-secondary'
}

const COPY = {
  viewAllJobs: 'View all jobs'
}

const mountComponent = (props: Record<string, unknown> = {}) =>
  mount(QueueOverlayActive, {
    props: {
      totalProgressStyle: { width: '65%' },
      currentNodeProgressStyle: { width: '40%' },
      totalPercentFormatted: '65%',
      currentNodePercentFormatted: '40%',
      currentNodeName: 'Sampler',
      runningCount: 1,
      queuedCount: 2,
      bottomRowClass: 'flex custom-bottom-row',
      ...props
    },
    global: {
      plugins: [i18n],
      directives: {
        tooltip: tooltipDirectiveStub
      }
    }
  })

describe('QueueOverlayActive', () => {
  it('renders progress metrics and emits actions when buttons clicked', async () => {
    const wrapper = mountComponent({ runningCount: 2, queuedCount: 3 })

    const progressBars = wrapper.findAll('.absolute.inset-0')
    expect(progressBars[0].attributes('style')).toContain('width: 65%')
    expect(progressBars[1].attributes('style')).toContain('width: 40%')

    const content = wrapper.text().replace(/\s+/g, ' ')
    expect(content).toContain('Total: 65%')

    const [runningSection, queuedSection] = wrapper.findAll(
      SELECTORS.summaryRow
    )
    expect(runningSection.text()).toContain('2')
    expect(runningSection.text()).toContain('running')
    expect(queuedSection.text()).toContain('3')
    expect(queuedSection.text()).toContain('queued')

    const currentNodeSection = wrapper.find(SELECTORS.currentNodeRow)
    expect(currentNodeSection.text()).toContain('Current node:')
    expect(currentNodeSection.text()).toContain('Sampler')
    expect(currentNodeSection.text()).toContain('40%')

    const interruptButton = wrapper.get(SELECTORS.interruptAllButton)
    await interruptButton.trigger('click')
    expect(wrapper.emitted('interruptAll')).toHaveLength(1)

    const clearQueuedButton = wrapper.get(SELECTORS.clearQueuedButton)
    await clearQueuedButton.trigger('click')
    expect(wrapper.emitted('clearQueued')).toHaveLength(1)

    const buttons = wrapper.findAll('button')
    const viewAllButton = buttons.find((btn) =>
      btn.text().includes(COPY.viewAllJobs)
    )
    expect(viewAllButton).toBeDefined()
    await viewAllButton!.trigger('click')
    expect(wrapper.emitted('viewAllJobs')).toHaveLength(1)

    expect(wrapper.find('.custom-bottom-row').exists()).toBe(true)
  })

  it('hides action buttons when counts are zero', () => {
    const wrapper = mountComponent({ runningCount: 0, queuedCount: 0 })

    expect(wrapper.find(SELECTORS.interruptAllButton).exists()).toBe(false)
    expect(wrapper.find(SELECTORS.clearQueuedButton).exists()).toBe(false)
  })

  it('builds tooltip configs with translated strings', () => {
    const spy = vi.spyOn(tooltipConfig, 'buildTooltipConfig')

    mountComponent()

    expect(spy).toHaveBeenCalledWith('Cancel job')
    expect(spy).toHaveBeenCalledWith('Clear queue')
  })
})
