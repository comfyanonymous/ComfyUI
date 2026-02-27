import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useResultGallery } from '@/composables/queue/useResultGallery'
import type { JobListItem as JobListViewItem } from '@/composables/queue/useJobList'
import type { JobListItem } from '@/platform/remote/comfyui/jobs/jobTypes'
import { ResultItemImpl, TaskItemImpl } from '@/stores/queueStore'

const createResultItem = (
  url: string,
  supportsPreview = true
): ResultItemImpl => {
  const item = new ResultItemImpl({
    filename: url,
    subfolder: '',
    type: 'output',
    nodeId: 'node-1',
    mediaType: supportsPreview ? 'images' : 'unknown'
  })
  // Override url getter for test matching
  Object.defineProperty(item, 'url', { get: () => url })
  Object.defineProperty(item, 'supportsPreview', { get: () => supportsPreview })
  return item
}

const createMockJob = (id: string, outputsCount = 1): JobListItem => ({
  id,
  status: 'completed',
  create_time: Date.now(),
  preview_output: null,
  outputs_count: outputsCount,
  priority: 0
})

const createTask = (
  preview?: ResultItemImpl,
  allOutputs?: ResultItemImpl[],
  outputsCount = 1
): TaskItemImpl => {
  const job = createMockJob(
    `task-${Math.random().toString(36).slice(2)}`,
    outputsCount
  )
  const flatOutputs = allOutputs ?? (preview ? [preview] : [])
  return new TaskItemImpl(job, {}, flatOutputs)
}

const createJobViewItem = (
  id: string,
  taskRef?: TaskItemImpl
): JobListViewItem =>
  ({
    id,
    title: `Job ${id}`,
    meta: '',
    state: 'completed',
    showClear: false,
    taskRef
  }) as JobListViewItem

describe('useResultGallery', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('collects only previewable outputs and preserves their order', async () => {
    const previewable = [createResultItem('p-1'), createResultItem('p-2')]
    const nonPreviewable = createResultItem('skip-me', false)
    const tasks = [
      createTask(previewable[0]),
      createTask(nonPreviewable),
      createTask(previewable[1]),
      createTask()
    ]

    const { galleryItems, galleryActiveIndex, onViewItem } = useResultGallery(
      () => tasks
    )

    await onViewItem(createJobViewItem('job-1', tasks[0]))

    expect(galleryItems.value).toEqual([previewable[0]])
    expect(galleryActiveIndex.value).toBe(0)
  })

  it('does not change state when there are no previewable tasks', async () => {
    const { galleryItems, galleryActiveIndex, onViewItem } = useResultGallery(
      () => []
    )

    await onViewItem(createJobViewItem('job-missing'))

    expect(galleryItems.value).toEqual([])
    expect(galleryActiveIndex.value).toBe(-1)
  })

  it('activates the index that matches the viewed preview URL', async () => {
    const previewable = [
      createResultItem('p-1'),
      createResultItem('p-2'),
      createResultItem('p-3')
    ]
    const tasks = previewable.map((preview) => createTask(preview))

    const { galleryItems, galleryActiveIndex, onViewItem } = useResultGallery(
      () => tasks
    )

    await onViewItem(createJobViewItem('job-2', tasks[1]))

    expect(galleryItems.value).toEqual([previewable[1]])
    expect(galleryActiveIndex.value).toBe(0)
  })

  it('defaults to the first entry when the clicked job lacks a preview', async () => {
    const previewable = [createResultItem('p-1'), createResultItem('p-2')]
    const tasks = previewable.map((preview) => createTask(preview))

    const { galleryItems, galleryActiveIndex, onViewItem } = useResultGallery(
      () => tasks
    )

    await onViewItem(createJobViewItem('job-no-preview'))

    expect(galleryItems.value).toEqual(previewable)
    expect(galleryActiveIndex.value).toBe(0)
  })

  it('defaults to the first entry when no gallery item matches the preview URL', async () => {
    const previewable = [createResultItem('p-1'), createResultItem('p-2')]
    const tasks = previewable.map((preview) => createTask(preview))

    const { galleryItems, galleryActiveIndex, onViewItem } = useResultGallery(
      () => tasks
    )

    const taskWithMismatchedPreview = createTask(createResultItem('missing'))
    await onViewItem(
      createJobViewItem('job-mismatch', taskWithMismatchedPreview)
    )

    expect(galleryItems.value).toEqual([createResultItem('missing')])
    expect(galleryActiveIndex.value).toBe(0)
  })

  it('loads full outputs when task has only preview outputs', async () => {
    const previewOutput = createResultItem('preview-1')
    const fullOutputs = [
      createResultItem('full-1'),
      createResultItem('full-2'),
      createResultItem('full-3')
    ]

    // Create a task with outputsCount > 1 to trigger lazy loading
    const job = createMockJob('task-1', 3)
    const task = new TaskItemImpl(job, {}, [previewOutput])

    // Mock loadFullOutputs to return full outputs
    const loadedTask = new TaskItemImpl(job, {}, fullOutputs)
    task.loadFullOutputs = async () => loadedTask

    const { galleryItems, galleryActiveIndex, onViewItem } = useResultGallery(
      () => [task]
    )

    await onViewItem(createJobViewItem('job-1', task))

    expect(galleryItems.value).toEqual(fullOutputs)
    expect(galleryActiveIndex.value).toBe(0)
  })
})
