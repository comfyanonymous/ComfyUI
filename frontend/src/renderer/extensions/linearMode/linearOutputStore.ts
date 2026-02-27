import { defineStore } from 'pinia'
import { ref, shallowRef, watch } from 'vue'

import { flattenNodeOutput } from '@/renderer/extensions/linearMode/flattenNodeOutput'
import type { InProgressItem } from '@/renderer/extensions/linearMode/linearModeTypes'
import type { ExecutedWsMessage } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { useAppMode } from '@/composables/useAppMode'
import { useExecutionStore } from '@/stores/executionStore'
import { useJobPreviewStore } from '@/stores/jobPreviewStore'

export const useLinearOutputStore = defineStore('linearOutput', () => {
  const { isAppMode } = useAppMode()
  const executionStore = useExecutionStore()
  const jobPreviewStore = useJobPreviewStore()

  const inProgressItems = ref<InProgressItem[]>([])
  const selectedId = ref<string | null>(null)
  const isFollowing = ref(true)
  const trackedJobId = ref<string | null>(null)
  const pendingResolve = ref(new Set<string>())

  let nextSeq = 0

  function makeItemId(jobId: string): string {
    return `job-${jobId}-${nextSeq++}`
  }

  function replaceItem(
    id: string,
    updater: (item: InProgressItem) => InProgressItem
  ) {
    inProgressItems.value = inProgressItems.value.map((i) =>
      i.id === id ? updater(i) : i
    )
  }

  // --- Actions ---

  const currentSkeletonId = shallowRef<string | null>(null)

  function onJobStart(jobId: string) {
    const item: InProgressItem = {
      id: makeItemId(jobId),
      jobId,
      state: 'skeleton'
    }
    currentSkeletonId.value = item.id
    inProgressItems.value = [item, ...inProgressItems.value]

    trackedJobId.value = jobId
    autoSelect(`slot:${item.id}`)
  }

  let raf: number | null = null
  function onLatentPreview(jobId: string, url: string) {
    // Issue in Firefox where it doesnt seem to always re-render, wrapping in RAF fixes it
    if (raf) cancelAnimationFrame(raf)
    raf = requestAnimationFrame(() => {
      const existing = inProgressItems.value.find(
        (i) => i.id === currentSkeletonId.value && i.jobId === jobId
      )

      if (existing) {
        const wasEmpty = existing.state === 'skeleton'
        replaceItem(existing.id, (i) => ({
          ...i,
          state: 'latent',
          latentPreviewUrl: url
        }))
        if (wasEmpty) autoSelect(`slot:${existing.id}`)
        return
      }

      // Only create on-demand for the tracked job
      if (jobId !== trackedJobId.value) return

      const item: InProgressItem = {
        id: makeItemId(jobId),
        jobId,
        state: 'latent',
        latentPreviewUrl: url
      }
      currentSkeletonId.value = item.id
      inProgressItems.value = [item, ...inProgressItems.value]
      autoSelect(`slot:${item.id}`)
    })
  }

  function onNodeExecuted(jobId: string, detail: ExecutedWsMessage) {
    const nodeId = String(detail.display_node || detail.node)
    const newOutputs = flattenNodeOutput([nodeId, detail.output])
    if (newOutputs.length === 0) return

    const skeletonItem = inProgressItems.value.find(
      (i) => i.id === currentSkeletonId.value && i.jobId === jobId
    )

    if (skeletonItem) {
      const imageItem: InProgressItem = {
        ...skeletonItem,
        state: 'image',
        output: newOutputs[0],
        latentPreviewUrl: undefined
      }
      autoSelect(`slot:${imageItem.id}`)

      const extras: InProgressItem[] = newOutputs.slice(1).map((o) => ({
        id: makeItemId(jobId),
        jobId,
        state: 'image' as const,
        output: o
      }))

      const idx = inProgressItems.value.indexOf(skeletonItem)
      const arr = [...inProgressItems.value]
      arr.splice(idx, 1, imageItem, ...extras)
      currentSkeletonId.value = null
      inProgressItems.value = arr
      return
    }

    // No skeleton — create image items directly (only for tracked job)
    if (jobId !== trackedJobId.value) return

    const newItems: InProgressItem[] = newOutputs.map((o) => ({
      id: makeItemId(jobId),
      jobId,
      state: 'image' as const,
      output: o
    }))
    autoSelect(`slot:${newItems[0].id}`)
    inProgressItems.value = [...newItems, ...inProgressItems.value]
  }

  function onJobComplete(jobId: string) {
    currentSkeletonId.value = null

    const hasImages = inProgressItems.value.some(
      (i) => i.jobId === jobId && i.state === 'image'
    )

    if (hasImages) {
      // Remove non-image items (skeletons, latents), keep images for absorption
      inProgressItems.value = inProgressItems.value.filter(
        (i) => i.jobId !== jobId || i.state === 'image'
      )
      pendingResolve.value = new Set([...pendingResolve.value, jobId])
    } else {
      removeJobItems(jobId)
    }
  }

  function removeJobItems(jobId: string) {
    const removed = inProgressItems.value.filter((i) => i.jobId === jobId)
    inProgressItems.value = inProgressItems.value.filter(
      (i) => i.jobId !== jobId
    )

    if (
      selectedId.value &&
      removed.some((i) => `slot:${i.id}` === selectedId.value)
    ) {
      selectedId.value = null
    }
  }

  function resolveIfReady(jobId: string, historyLoaded: boolean) {
    if (!pendingResolve.value.has(jobId)) return
    if (!historyLoaded) return

    const next = new Set(pendingResolve.value)
    next.delete(jobId)
    pendingResolve.value = next

    removeJobItems(jobId)
  }

  function select(id: string | null) {
    selectedId.value = id
    isFollowing.value = false
  }

  function selectAsLatest(id: string | null) {
    selectedId.value = id
    isFollowing.value = true
  }

  function autoSelect(slotId: string) {
    const sel = selectedId.value
    if (!sel || sel.startsWith('slot:') || isFollowing.value) {
      selectedId.value = slotId
      isFollowing.value = true
      return
    }
    // User is browsing history — don't yank
  }

  // --- Event bindings (only active in app mode) ---

  function handleExecuted({ detail }: CustomEvent<ExecutedWsMessage>) {
    const jobId = detail.prompt_id
    if (jobId !== executionStore.activeJobId) return
    onNodeExecuted(jobId, detail)
  }

  function reset() {
    if (raf) {
      cancelAnimationFrame(raf)
      raf = null
    }
    inProgressItems.value = []
    selectedId.value = null
    isFollowing.value = true
    trackedJobId.value = null
    currentSkeletonId.value = null
    pendingResolve.value = new Set()
  }

  watch(
    () => executionStore.activeJobId,
    (jobId, oldJobId) => {
      if (!isAppMode.value) return
      if (oldJobId && oldJobId !== jobId) {
        onJobComplete(oldJobId)
      }
      if (jobId) {
        onJobStart(jobId)
      }
    }
  )

  watch(
    () => jobPreviewStore.previewsByPromptId,
    (previews) => {
      if (!isAppMode.value) return
      const jobId = executionStore.activeJobId
      if (!jobId) return
      const url = previews[jobId]
      if (url) onLatentPreview(jobId, url)
    },
    { deep: true }
  )

  watch(
    isAppMode,
    (active, wasActive) => {
      if (active) {
        api.addEventListener('executed', handleExecuted)
      } else if (wasActive) {
        api.removeEventListener('executed', handleExecuted)
        reset()
      }
    },
    { immediate: true }
  )

  return {
    inProgressItems,
    selectedId,
    trackedJobId,
    pendingResolve,
    select,
    selectAsLatest,
    resolveIfReady,

    onJobStart,
    onLatentPreview,
    onNodeExecuted,
    onJobComplete
  }
})
