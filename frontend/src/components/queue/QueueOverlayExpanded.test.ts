import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import type { JobListItem } from '@/composables/queue/useJobList'

vi.mock('@/composables/queue/useJobMenu', () => ({
  useJobMenu: () => ({ jobMenuEntries: [] })
}))

vi.mock('@/composables/useErrorHandling', () => ({
  useErrorHandling: () => ({
    wrapWithErrorHandlingAsync: <T extends (...args: never[]) => unknown>(
      fn: T
    ) => fn
  })
}))

import QueueOverlayExpanded from '@/components/queue/QueueOverlayExpanded.vue'

const QueueOverlayHeaderStub = {
  template: '<div />'
}

const JobFiltersBarStub = {
  template: '<div />'
}

const JobAssetsListStub = {
  name: 'JobAssetsList',
  template: '<div class="job-assets-list-stub" />'
}

const JobContextMenuStub = {
  template: '<div />'
}

const createJob = (): JobListItem => ({
  id: 'job-1',
  title: 'Job 1',
  meta: 'meta',
  state: 'pending'
})

const mountComponent = () =>
  mount(QueueOverlayExpanded, {
    props: {
      headerTitle: 'Jobs',
      queuedCount: 1,
      selectedJobTab: 'All',
      selectedWorkflowFilter: 'all',
      selectedSortMode: 'mostRecent',
      displayedJobGroups: [],
      hasFailedJobs: false
    },
    global: {
      stubs: {
        QueueOverlayHeader: QueueOverlayHeaderStub,
        JobFiltersBar: JobFiltersBarStub,
        JobAssetsList: JobAssetsListStub,
        JobContextMenu: JobContextMenuStub
      }
    }
  })

describe('QueueOverlayExpanded', () => {
  it('renders JobAssetsList', () => {
    const wrapper = mountComponent()
    expect(wrapper.find('.job-assets-list-stub').exists()).toBe(true)
  })

  it('re-emits list item actions from JobAssetsList', async () => {
    const wrapper = mountComponent()
    const job = createJob()
    const jobAssetsList = wrapper.findComponent({ name: 'JobAssetsList' })

    jobAssetsList.vm.$emit('cancel-item', job)
    jobAssetsList.vm.$emit('delete-item', job)
    jobAssetsList.vm.$emit('view-item', job)
    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('cancelItem')?.[0]).toEqual([job])
    expect(wrapper.emitted('deleteItem')?.[0]).toEqual([job])
    expect(wrapper.emitted('viewItem')?.[0]).toEqual([job])
  })
})
