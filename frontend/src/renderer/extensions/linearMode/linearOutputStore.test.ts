import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import { useLinearOutputStore } from '@/renderer/extensions/linearMode/linearOutputStore'
import type { ExecutedWsMessage } from '@/schemas/apiSchema'
import { ResultItemImpl } from '@/stores/queueStore'

const activeJobIdRef = ref<string | null>(null)
const previewsRef = ref<Record<string, string>>({})
const isAppModeRef = ref(true)

const { apiTarget } = vi.hoisted(() => ({
  apiTarget: new EventTarget()
}))

vi.mock('@/composables/useAppMode', () => ({
  useAppMode: () => ({
    isAppMode: isAppModeRef
  })
}))

vi.mock('@/stores/executionStore', () => ({
  useExecutionStore: () => ({
    get activeJobId() {
      return activeJobIdRef.value
    }
  })
}))

vi.mock('@/stores/jobPreviewStore', () => ({
  useJobPreviewStore: () => ({
    get previewsByPromptId() {
      return previewsRef.value
    }
  })
}))

vi.mock('@/scripts/api', () => ({
  api: Object.assign(apiTarget, {
    apiURL: (path: string) => path
  })
}))

vi.mock('@/renderer/extensions/linearMode/flattenNodeOutput', () => ({
  flattenNodeOutput: ([nodeId, output]: [
    string | number,
    Record<string, unknown>
  ]) => {
    if (!output.images) return []
    return (output.images as Array<Record<string, string>>).map(
      (img) =>
        new ResultItemImpl({
          ...img,
          nodeId: String(nodeId),
          mediaType: 'images'
        })
    )
  }
}))

function makeExecutedDetail(
  promptId: string,
  images: Array<Record<string, string>> = [
    { filename: 'out.png', subfolder: '', type: 'output' }
  ]
): ExecutedWsMessage {
  return {
    prompt_id: promptId,
    node: '1',
    display_node: '1',
    output: { images }
  } as ExecutedWsMessage
}

describe('linearOutputStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    activeJobIdRef.value = null
    previewsRef.value = {}
    isAppModeRef.value = true
  })

  afterEach(() => {
    activeJobIdRef.value = null
    previewsRef.value = {}
  })

  it('creates a skeleton item when a job starts', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    expect(store.inProgressItems).toHaveLength(1)
    expect(store.inProgressItems[0].state).toBe('skeleton')
    expect(store.inProgressItems[0].jobId).toBe('job-1')
  })

  it('auto-selects skeleton on first job start when no selection', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    expect(store.selectedId).toBe(`slot:${store.inProgressItems[0].id}`)
  })

  it('transitions to latent on preview', () => {
    vi.useFakeTimers()
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    const itemId = store.inProgressItems[0].id
    store.onLatentPreview('job-1', 'blob:preview')
    vi.advanceTimersByTime(16)

    expect(store.inProgressItems[0].state).toBe('latent')
    expect(store.inProgressItems[0].latentPreviewUrl).toBe('blob:preview')
    expect(store.selectedId).toBe(`slot:${itemId}`)
    vi.useRealTimers()
  })

  it('ignores latent preview for other jobs', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    store.onLatentPreview('job-other', 'blob:preview')

    expect(store.inProgressItems[0].state).toBe('skeleton')
  })

  it('transitions to image on executed event', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))

    const imageItems = store.inProgressItems.filter((i) => i.state === 'image')
    expect(imageItems).toHaveLength(1)
    expect(imageItems[0].output).toBeDefined()
  })

  it('does not create trailing skeleton after executed output', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))

    expect(store.inProgressItems).toHaveLength(1)
    expect(store.inProgressItems[0].state).toBe('image')
  })

  it('handles multi-output executed events', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    store.onNodeExecuted(
      'job-1',
      makeExecutedDetail('job-1', [
        { filename: 'a.png', subfolder: '', type: 'output' },
        { filename: 'b.png', subfolder: '', type: 'output' }
      ])
    )

    const imageItems = store.inProgressItems.filter((i) => i.state === 'image')
    expect(imageItems).toHaveLength(2)
  })

  it('removes slots when job ends without image outputs', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')
    expect(store.inProgressItems).toHaveLength(1)

    store.onJobComplete('job-1')

    expect(store.inProgressItems).toHaveLength(0)
    expect(store.pendingResolve.size).toBe(0)
  })

  it('adds to pendingResolve when job completes with images', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))

    store.onJobComplete('job-1')

    expect(store.inProgressItems.length).toBeGreaterThan(0)
    expect(store.pendingResolve.has('job-1')).toBe(true)
  })

  it('removes items when history resolves', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))

    store.onJobComplete('job-1')
    expect(store.inProgressItems.length).toBeGreaterThan(0)

    store.resolveIfReady('job-1', true)

    expect(store.inProgressItems).toHaveLength(0)
    expect(store.pendingResolve.has('job-1')).toBe(false)
  })

  it('does not resolve if history has not arrived', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))

    store.onJobComplete('job-1')

    store.resolveIfReady('job-1', false)

    expect(store.inProgressItems.length).toBeGreaterThan(0)
    expect(store.pendingResolve.has('job-1')).toBe(true)
  })

  it('does not auto-select when user is browsing history', () => {
    const store = useLinearOutputStore()

    // User manually selects a history item (browsing)
    store.select('history:asset-2:0')

    store.onJobStart('job-1')

    // Should NOT yank to in-progress — user is browsing
    expect(store.selectedId).toBe('history:asset-2:0')

    store.onLatentPreview('job-1', 'blob:preview')

    // Still should NOT yank
    expect(store.selectedId).toBe('history:asset-2:0')
  })

  it('auto-selects on new job when following latest', () => {
    const store = useLinearOutputStore()

    // selectAsLatest simulates "following the latest output"
    store.selectAsLatest('history:asset-1:0')

    store.onJobStart('job-1')

    // Following latest → auto-select new skeleton
    expect(store.selectedId?.startsWith('slot:')).toBe(true)
  })

  it('does not auto-select on new job when browsing history', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    // User manually browses to an older output
    store.select('history:asset-1:0')

    store.onJobStart('job-2')

    // Should NOT auto-select — user is browsing
    expect(store.selectedId).toBe('history:asset-1:0')
  })

  it('falls back selection when selected item is removed', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')
    const firstId = `slot:${store.inProgressItems[0].id}`
    expect(store.selectedId).toBe(firstId)

    store.onJobComplete('job-1')

    // Skeleton removed, no images, should clear selection
    expect(store.selectedId).toBeNull()
  })

  it('creates skeleton on-demand when latent arrives after execute', () => {
    vi.useFakeTimers()
    const store = useLinearOutputStore()
    store.onJobStart('job-1')
    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))

    // No skeleton after execute
    expect(
      store.inProgressItems.filter((i) => i.state === 'skeleton')
    ).toHaveLength(0)

    // Next node sends latent preview — skeleton created on demand
    store.onLatentPreview('job-1', 'blob:next')
    vi.advanceTimersByTime(16)

    expect(store.inProgressItems[0].state).toBe('latent')
    expect(store.inProgressItems[0].latentPreviewUrl).toBe('blob:next')
    expect(store.inProgressItems).toHaveLength(2)
    vi.useRealTimers()
  })

  it('handles execute without prior skeleton (no latent preview)', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    // First node executes (consumes initial skeleton)
    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))
    expect(store.inProgressItems).toHaveLength(1)

    // Second node executes directly (no latent, no skeleton)
    store.onNodeExecuted(
      'job-1',
      makeExecutedDetail('job-1', [
        { filename: 'b.png', subfolder: '', type: 'output' }
      ])
    )

    const images = store.inProgressItems.filter((i) => i.state === 'image')
    expect(images).toHaveLength(2)
  })

  it('does not fall back selection to stale items from other jobs', () => {
    const store = useLinearOutputStore()

    // Job 1 starts but is never completed (simulates watcher bug)
    store.onJobStart('job-1')
    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))

    // Job 2 starts directly
    store.onJobStart('job-2')
    store.onNodeExecuted('job-2', makeExecutedDetail('job-2'))
    store.onJobComplete('job-2')

    // Resolve job-2
    store.resolveIfReady('job-2', true)

    // Should clear to null for history takeover, NOT fall back to job-1
    expect(store.selectedId).toBeNull()
  })

  it('transitions to latent via previewsByPromptId watcher', async () => {
    vi.useFakeTimers()
    const { nextTick } = await import('vue')
    const store = useLinearOutputStore()

    activeJobIdRef.value = 'job-1'
    await nextTick()

    expect(store.inProgressItems).toHaveLength(1)
    expect(store.inProgressItems[0].state).toBe('skeleton')

    // Simulate jobPreviewStore update
    previewsRef.value = { 'job-1': 'blob:preview-1' }
    await nextTick()
    vi.advanceTimersByTime(16)

    expect(store.inProgressItems[0].state).toBe('latent')
    expect(store.inProgressItems[0].latentPreviewUrl).toBe('blob:preview-1')
    vi.useRealTimers()
  })

  it('completes previous job on direct job transition', async () => {
    const { nextTick } = await import('vue')
    const store = useLinearOutputStore()

    activeJobIdRef.value = 'job-1'
    await nextTick()

    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))

    // Direct transition: job-1 → job-2 (no null in between)
    activeJobIdRef.value = 'job-2'
    await nextTick()

    // job-1 should have been completed
    expect(store.pendingResolve.has('job-1')).toBe(true)
    // job-2 should have started
    expect(store.inProgressItems.some((i) => i.jobId === 'job-2')).toBe(true)
  })

  it('two sequential runs: selection clears after each resolve', () => {
    const store = useLinearOutputStore()

    // Run 1: 3 outputs
    store.onJobStart('job-1')
    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))
    store.onNodeExecuted(
      'job-1',
      makeExecutedDetail('job-1', [
        { filename: 'b.png', subfolder: '', type: 'output' }
      ])
    )
    store.onNodeExecuted(
      'job-1',
      makeExecutedDetail('job-1', [
        { filename: 'c.png', subfolder: '', type: 'output' }
      ])
    )

    const run1Images = store.inProgressItems.filter((i) => i.state === 'image')
    expect(run1Images).toHaveLength(3)

    store.onJobComplete('job-1')
    store.resolveIfReady('job-1', true)
    expect(store.selectedId).toBeNull()
    expect(store.inProgressItems).toHaveLength(0)

    // Simulate OutputHistory selecting run 1's first output (following latest)
    store.selectAsLatest('history:asset-run1:0')

    // Run 2: 3 outputs
    store.onJobStart('job-2')
    store.onNodeExecuted('job-2', makeExecutedDetail('job-2'))
    store.onNodeExecuted(
      'job-2',
      makeExecutedDetail('job-2', [
        { filename: 'e.png', subfolder: '', type: 'output' }
      ])
    )
    store.onNodeExecuted(
      'job-2',
      makeExecutedDetail('job-2', [
        { filename: 'f.png', subfolder: '', type: 'output' }
      ])
    )

    const run2Images = store.inProgressItems.filter((i) => i.state === 'image')
    expect(run2Images).toHaveLength(3)
    // Selection on run 2's latest output, not run 1's
    expect(store.selectedId).toBe(`slot:${run2Images[0].id}`)

    store.onJobComplete('job-2')
    store.resolveIfReady('job-2', true)

    // Must be null for history takeover — not a stale item
    expect(store.selectedId).toBeNull()
    expect(store.inProgressItems).toHaveLength(0)
  })

  it('keeps items visible across multiple resolveIfReady calls until loaded', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')
    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))
    store.onNodeExecuted(
      'job-1',
      makeExecutedDetail('job-1', [
        { filename: 'b.png', subfolder: '', type: 'output' }
      ])
    )
    store.onJobComplete('job-1')

    // History asset exists but outputs not loaded yet (async)
    store.resolveIfReady('job-1', false)
    expect(store.inProgressItems).toHaveLength(2)
    expect(store.pendingResolve.has('job-1')).toBe(true)

    // Still not loaded on next check
    store.resolveIfReady('job-1', false)
    expect(store.inProgressItems).toHaveLength(2)

    // Outputs finally loaded
    store.resolveIfReady('job-1', true)
    expect(store.inProgressItems).toHaveLength(0)
    expect(store.pendingResolve.has('job-1')).toBe(false)
  })

  it('does not remove in-progress items while history outputs are loading', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')
    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))
    store.onNodeExecuted(
      'job-1',
      makeExecutedDetail('job-1', [
        { filename: 'b.png', subfolder: '', type: 'output' }
      ])
    )
    store.onNodeExecuted(
      'job-1',
      makeExecutedDetail('job-1', [
        { filename: 'c.png', subfolder: '', type: 'output' }
      ])
    )
    store.onJobComplete('job-1')

    const itemCount = store.inProgressItems.length
    expect(itemCount).toBe(3)

    // History asset arrived but allOutputs() returns [] (still loading).
    // Caller passes false — items must stay visible to prevent a gap
    // where neither in-progress nor history items are rendered.
    store.resolveIfReady('job-1', false)
    expect(store.inProgressItems).toHaveLength(itemCount)

    // Once allOutputs() loads, caller passes true — safe to resolve
    store.resolveIfReady('job-1', true)
    expect(store.inProgressItems).toHaveLength(0)
  })

  it('ignores executed events for other jobs', () => {
    const store = useLinearOutputStore()
    store.onJobStart('job-1')

    store.onNodeExecuted('job-other', makeExecutedDetail('job-other'))

    expect(store.inProgressItems.every((i) => i.state === 'skeleton')).toBe(
      true
    )
  })

  it('resets state when leaving app mode', async () => {
    const { nextTick } = await import('vue')
    const store = useLinearOutputStore()

    store.onJobStart('job-1')
    store.onNodeExecuted('job-1', makeExecutedDetail('job-1'))
    store.select('slot:some-id')
    expect(store.inProgressItems.length).toBeGreaterThan(0)

    isAppModeRef.value = false
    await nextTick()

    expect(store.inProgressItems).toHaveLength(0)
    expect(store.selectedId).toBeNull()
    expect(store.trackedJobId).toBeNull()
    expect(store.pendingResolve.size).toBe(0)
  })

  it('ignores execution events when not in app mode', async () => {
    const { nextTick } = await import('vue')
    const store = useLinearOutputStore()

    isAppModeRef.value = false
    await nextTick()

    // Watcher-driven job start should be ignored
    activeJobIdRef.value = 'job-1'
    await nextTick()

    expect(store.inProgressItems).toHaveLength(0)
  })
})
