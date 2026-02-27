import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { defineComponent, h } from 'vue'

const popoverCloseSpy = vi.fn()

vi.mock('@/components/ui/Popover.vue', () => {
  const PopoverStub = defineComponent({
    name: 'Popover',
    setup(_, { slots }) {
      return () =>
        h('div', [
          slots.button?.(),
          slots.default?.({
            close: () => {
              popoverCloseSpy()
            }
          })
        ])
    }
  })
  return { default: PopoverStub }
})

const mockGetSetting = vi.fn<(key: string) => boolean | undefined>((key) =>
  key === 'Comfy.Queue.QPOV2' ? true : undefined
)
const mockSetSetting = vi.fn()
const mockSidebarTabStore = {
  activeSidebarTabId: null as string | null
}

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => ({
    get: mockGetSetting,
    set: mockSetSetting
  })
}))

vi.mock('@/stores/workspace/sidebarTabStore', () => ({
  useSidebarTabStore: () => mockSidebarTabStore
}))

import QueueOverlayHeader from './QueueOverlayHeader.vue'
import * as tooltipConfig from '@/composables/useTooltipConfig'

const tooltipDirectiveStub = {
  mounted: vi.fn(),
  updated: vi.fn()
}

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      g: { more: 'More' },
      sideToolbar: {
        queueProgressOverlay: {
          queuedSuffix: 'queued',
          clearQueued: 'Clear queued',
          clearQueueTooltip: 'Clear queue',
          clearAllJobsTooltip: 'Cancel all running jobs',
          moreOptions: 'More options',
          clearHistory: 'Clear history',
          dockedJobHistory: 'Docked Job History'
        }
      }
    }
  }
})

const mountHeader = (props = {}) =>
  mount(QueueOverlayHeader, {
    props: {
      headerTitle: 'Job queue',
      queuedCount: 3,
      ...props
    },
    global: {
      plugins: [i18n],
      directives: { tooltip: tooltipDirectiveStub }
    }
  })

describe('QueueOverlayHeader', () => {
  beforeEach(() => {
    popoverCloseSpy.mockClear()
    mockSetSetting.mockClear()
    mockSidebarTabStore.activeSidebarTabId = null
    mockGetSetting.mockImplementation((key: string) =>
      key === 'Comfy.Queue.QPOV2' ? true : undefined
    )
  })

  it('renders header title', () => {
    const wrapper = mountHeader()
    expect(wrapper.text()).toContain('Job queue')
  })

  it('shows clear queue text and emits clear queued', async () => {
    const wrapper = mountHeader({ queuedCount: 4 })

    expect(wrapper.text()).toContain('Clear queue')
    expect(wrapper.text()).not.toContain('4 queued')

    const clearQueuedButton = wrapper.get('button[aria-label="Clear queued"]')
    await clearQueuedButton.trigger('click')
    expect(wrapper.emitted('clearQueued')).toHaveLength(1)
  })

  it('disables clear queued button when queued count is zero', () => {
    const wrapper = mountHeader({ queuedCount: 0 })
    const clearQueuedButton = wrapper.get('button[aria-label="Clear queued"]')

    expect(clearQueuedButton.attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('Clear queue')
  })

  it('emits clear history from the menu', async () => {
    const spy = vi.spyOn(tooltipConfig, 'buildTooltipConfig')

    const wrapper = mountHeader()

    expect(wrapper.find('button[aria-label="More options"]').exists()).toBe(
      true
    )
    expect(spy).toHaveBeenCalledWith('More')

    const clearHistoryButton = wrapper.get(
      '[data-testid="clear-history-action"]'
    )
    await clearHistoryButton.trigger('click')
    expect(popoverCloseSpy).toHaveBeenCalledTimes(1)
    expect(wrapper.emitted('clearHistory')).toHaveLength(1)
  })

  it('opens floating queue progress overlay when disabling from the menu', async () => {
    const wrapper = mountHeader()

    const dockedJobHistoryButton = wrapper.get(
      '[data-testid="docked-job-history-action"]'
    )
    await dockedJobHistoryButton.trigger('click')

    expect(mockSetSetting).toHaveBeenCalledTimes(2)
    expect(mockSetSetting).toHaveBeenNthCalledWith(
      1,
      'Comfy.Queue.QPOV2',
      false
    )
    expect(mockSetSetting).toHaveBeenNthCalledWith(
      2,
      'Comfy.Queue.History.Expanded',
      true
    )
    expect(mockSidebarTabStore.activeSidebarTabId).toBe(null)
  })

  it('opens docked job history sidebar when enabling from the menu', async () => {
    mockGetSetting.mockImplementation((key: string) =>
      key === 'Comfy.Queue.QPOV2' ? false : undefined
    )
    const wrapper = mountHeader()

    const dockedJobHistoryButton = wrapper.get(
      '[data-testid="docked-job-history-action"]'
    )
    await dockedJobHistoryButton.trigger('click')

    expect(mockSetSetting).toHaveBeenCalledTimes(1)
    expect(mockSetSetting).toHaveBeenCalledWith('Comfy.Queue.QPOV2', true)
    expect(mockSidebarTabStore.activeSidebarTabId).toBe('job-history')
  })
})
