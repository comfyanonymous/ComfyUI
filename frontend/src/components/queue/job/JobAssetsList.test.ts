import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import type { JobGroup, JobListItem } from '@/composables/queue/useJobList'
import type { JobListItem as ApiJobListItem } from '@/platform/remote/comfyui/jobs/jobTypes'
import { ResultItemImpl, TaskItemImpl } from '@/stores/queueStore'

import JobAssetsList from './JobAssetsList.vue'

vi.mock('vue-i18n', () => {
  return {
    createI18n: () => ({
      global: {
        t: (key: string) => key,
        te: () => true,
        d: (value: string) => value
      }
    }),
    useI18n: () => ({
      t: (key: string) => key
    })
  }
})

const createResultItem = (
  filename: string,
  mediaType: string = 'images'
): ResultItemImpl => {
  const item = new ResultItemImpl({
    filename,
    subfolder: '',
    type: 'output',
    nodeId: 'node-1',
    mediaType
  })
  Object.defineProperty(item, 'url', {
    get: () => `/api/view/${filename}`
  })
  return item
}

const createTaskRef = (preview?: ResultItemImpl): TaskItemImpl => {
  const job: ApiJobListItem = {
    id: `task-${Math.random().toString(36).slice(2)}`,
    status: 'completed',
    create_time: Date.now(),
    preview_output: null,
    outputs_count: preview ? 1 : 0,
    priority: 0
  }
  const flatOutputs = preview ? [preview] : []
  return new TaskItemImpl(job, {}, flatOutputs)
}

const buildJob = (overrides: Partial<JobListItem> = {}): JobListItem => ({
  id: 'job-1',
  title: 'Job 1',
  meta: 'meta',
  state: 'completed',
  taskRef: createTaskRef(createResultItem('job-1.png')),
  ...overrides
})

const mountJobAssetsList = (jobs: JobListItem[]) => {
  const displayedJobGroups: JobGroup[] = [
    {
      key: 'group-1',
      label: 'Group 1',
      items: jobs
    }
  ]

  return mount(JobAssetsList, {
    props: { displayedJobGroups }
  })
}

describe('JobAssetsList', () => {
  it('emits viewItem on preview-click for completed jobs with preview', async () => {
    const job = buildJob()
    const wrapper = mountJobAssetsList([job])

    const listItem = wrapper.findComponent({ name: 'AssetsListItem' })
    listItem.vm.$emit('preview-click')
    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('viewItem')).toEqual([[job]])
  })

  it('emits viewItem on double-click for completed jobs with preview', async () => {
    const job = buildJob()
    const wrapper = mountJobAssetsList([job])

    const listItem = wrapper.findComponent({ name: 'AssetsListItem' })
    await listItem.trigger('dblclick')
    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('viewItem')).toEqual([[job]])
  })

  it('emits viewItem on double-click for completed video jobs without icon image', async () => {
    const job = buildJob({
      iconImageUrl: undefined,
      taskRef: createTaskRef(createResultItem('job-1.webm', 'video'))
    })
    const wrapper = mountJobAssetsList([job])

    const listItem = wrapper.findComponent({ name: 'AssetsListItem' })
    expect(listItem.props('previewUrl')).toBe('/api/view/job-1.webm')
    expect(listItem.props('isVideoPreview')).toBe(true)

    await listItem.trigger('dblclick')
    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('viewItem')).toEqual([[job]])
  })

  it('emits viewItem on icon click for completed 3D jobs without preview tile', async () => {
    const job = buildJob({
      iconImageUrl: undefined,
      taskRef: createTaskRef(createResultItem('job-1.glb', 'model'))
    })
    const wrapper = mountJobAssetsList([job])

    const listItem = wrapper.findComponent({ name: 'AssetsListItem' })

    await listItem.find('i').trigger('click')
    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('viewItem')).toEqual([[job]])
  })

  it('does not emit viewItem on double-click for non-completed jobs', async () => {
    const job = buildJob({
      state: 'running',
      taskRef: createTaskRef(createResultItem('job-1.png'))
    })
    const wrapper = mountJobAssetsList([job])

    const listItem = wrapper.findComponent({ name: 'AssetsListItem' })
    await listItem.trigger('dblclick')
    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('viewItem')).toBeUndefined()
  })
})
