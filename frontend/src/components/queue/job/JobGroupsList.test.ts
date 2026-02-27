import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, nextTick } from 'vue'

import JobGroupsList from '@/components/queue/job/JobGroupsList.vue'
import type { JobGroup, JobListItem } from '@/composables/queue/useJobList'
import type { TaskItemImpl } from '@/stores/queueStore'

const QueueJobItemStub = defineComponent({
  name: 'QueueJobItemStub',
  props: {
    jobId: { type: String, required: true },
    workflowId: { type: String, default: undefined },
    state: { type: String, required: true },
    title: { type: String, required: true },
    rightText: { type: String, default: '' },
    iconName: { type: String, default: undefined },
    iconImageUrl: { type: String, default: undefined },
    showClear: { type: Boolean, default: undefined },
    showMenu: { type: Boolean, default: undefined },
    progressTotalPercent: { type: Number, default: undefined },
    progressCurrentPercent: { type: Number, default: undefined },
    runningNodeName: { type: String, default: undefined },
    activeDetailsId: { type: String, default: null }
  },
  template: '<div class="queue-job-item-stub"></div>'
})

const createJobItem = (overrides: Partial<JobListItem> = {}): JobListItem => {
  const { taskRef, ...rest } = overrides
  return {
    id: 'job-id',
    title: 'Example job',
    meta: 'Meta text',
    state: 'running',
    iconName: 'icon',
    iconImageUrl: 'https://example.com/icon.png',
    showClear: true,
    taskRef: (taskRef ?? {
      workflow: { id: 'workflow-id' }
    }) as TaskItemImpl,
    progressTotalPercent: 60,
    progressCurrentPercent: 30,
    runningNodeName: 'Node A',
    ...rest
  }
}

const mountComponent = (groups: JobGroup[]) =>
  mount(JobGroupsList, {
    props: { displayedJobGroups: groups },
    global: {
      stubs: {
        QueueJobItem: QueueJobItemStub
      }
    }
  })

afterEach(() => {
  vi.useRealTimers()
})

describe('JobGroupsList hover behavior', () => {
  it('delays showing and hiding details while hovering over job rows', async () => {
    vi.useFakeTimers()
    const job = createJobItem({ id: 'job-d' })
    const wrapper = mountComponent([
      { key: 'today', label: 'Today', items: [job] }
    ])
    const jobItem = wrapper.findComponent(QueueJobItemStub)

    jobItem.vm.$emit('details-enter', job.id)
    vi.advanceTimersByTime(199)
    await nextTick()
    expect(
      wrapper.findComponent(QueueJobItemStub).props('activeDetailsId')
    ).toBeNull()

    vi.advanceTimersByTime(1)
    await nextTick()
    expect(
      wrapper.findComponent(QueueJobItemStub).props('activeDetailsId')
    ).toBe(job.id)

    wrapper.findComponent(QueueJobItemStub).vm.$emit('details-leave', job.id)
    vi.advanceTimersByTime(149)
    await nextTick()
    expect(
      wrapper.findComponent(QueueJobItemStub).props('activeDetailsId')
    ).toBe(job.id)

    vi.advanceTimersByTime(1)
    await nextTick()
    expect(
      wrapper.findComponent(QueueJobItemStub).props('activeDetailsId')
    ).toBeNull()
  })
})
