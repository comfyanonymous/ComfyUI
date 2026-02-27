import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { defineComponent } from 'vue'

vi.mock('primevue/popover', () => {
  const PopoverStub = defineComponent({
    name: 'Popover',
    setup(_, { slots, expose }) {
      expose({
        hide: () => undefined,
        toggle: (_event: Event) => undefined
      })
      return () => slots.default?.()
    }
  })
  return { default: PopoverStub }
})

vi.mock('@/platform/distribution/types', () => ({
  isCloud: false
}))

import JobFiltersBar from '@/components/queue/job/JobFiltersBar.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      g: {
        all: 'All',
        completed: 'Completed'
      },
      queue: {
        jobList: {
          sortMostRecent: 'Most recent',
          sortTotalGenerationTime: 'Total generation time'
        }
      },
      sideToolbar: {
        queueProgressOverlay: {
          filterJobs: 'Filter jobs',
          filterBy: 'Filter by',
          sortJobs: 'Sort jobs',
          sortBy: 'Sort by',
          showAssets: 'Show assets',
          showAssetsPanel: 'Show assets panel',
          filterAllWorkflows: 'All workflows',
          filterCurrentWorkflow: 'Current workflow'
        }
      }
    }
  }
})

describe('JobFiltersBar', () => {
  it('emits showAssets when the assets icon button is clicked', async () => {
    const wrapper = mount(JobFiltersBar, {
      props: {
        selectedJobTab: 'All',
        selectedWorkflowFilter: 'all',
        selectedSortMode: 'mostRecent',
        hasFailedJobs: false
      },
      global: {
        plugins: [i18n],
        directives: { tooltip: () => undefined }
      }
    })

    const showAssetsButton = wrapper.get(
      'button[aria-label="Show assets panel"]'
    )
    await showAssetsButton.trigger('click')

    expect(wrapper.emitted('showAssets')).toHaveLength(1)
  })

  it('hides the assets icon button when hideShowAssetsAction is true', () => {
    const wrapper = mount(JobFiltersBar, {
      props: {
        selectedJobTab: 'All',
        selectedWorkflowFilter: 'all',
        selectedSortMode: 'mostRecent',
        hasFailedJobs: false,
        hideShowAssetsAction: true
      },
      global: {
        plugins: [i18n],
        directives: { tooltip: () => undefined }
      }
    })

    expect(
      wrapper.find('button[aria-label="Show assets panel"]').exists()
    ).toBe(false)
  })
})
