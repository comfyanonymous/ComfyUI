<script setup lang="ts">
import { useEventListener, useInfiniteScroll } from '@vueuse/core'
import { ListboxContent, ListboxItem, ListboxRoot } from 'reka-ui'
import {
  computed,
  nextTick,
  toValue,
  useTemplateRef,
  watch,
  watchEffect
} from 'vue'

import { CanvasPointer } from '@/lib/litegraph/src/CanvasPointer'
import { getOutputAssetMetadata } from '@/platform/assets/schemas/assetMetadataSchema'
import OutputHistoryActiveQueueItem from '@/renderer/extensions/linearMode/OutputHistoryActiveQueueItem.vue'
import OutputHistoryItem from '@/renderer/extensions/linearMode/OutputHistoryItem.vue'
import { useLinearOutputStore } from '@/renderer/extensions/linearMode/linearOutputStore'
import type {
  OutputSelection,
  SelectionValue
} from '@/renderer/extensions/linearMode/linearModeTypes'
import OutputPreviewItem from '@/renderer/extensions/linearMode/OutputPreviewItem.vue'
import { useOutputHistory } from '@/renderer/extensions/linearMode/useOutputHistory'
import { useQueueStore } from '@/stores/queueStore'
import { cn } from '@/utils/tailwindUtil'

const { outputs, allOutputs } = useOutputHistory()
const queueStore = useQueueStore()
const store = useLinearOutputStore()

const emit = defineEmits<{
  updateSelection: [selection: OutputSelection]
}>()

const queueCount = computed(
  () => queueStore.runningTasks.length + queueStore.pendingTasks.length
)

const listboxRef = useTemplateRef<{
  highlightItem: (value: SelectionValue) => void
}>('listboxRef')

const itemClass = cn(
  'shrink-0 cursor-pointer p-1 rounded-lg border-2 border-transparent outline-none',
  'data-[state=checked]:border-interface-panel-job-progress-border'
)

const hasActiveContent = computed(() => store.inProgressItems.length > 0)

const visibleHistory = computed(() =>
  outputs.media.value.filter((a) => toValue(allOutputs(a)).length > 0)
)

const selectableItems = computed(() => {
  const items: SelectionValue[] = []
  for (const item of store.inProgressItems) {
    items.push({
      id: `slot:${item.id}`,
      kind: 'inProgress',
      itemId: item.id
    })
  }
  for (const asset of outputs.media.value) {
    const outs = toValue(allOutputs(asset))
    for (let k = 0; k < outs.length; k++) {
      items.push({
        id: `history:${asset.id}:${k}`,
        kind: 'history',
        assetId: asset.id,
        key: k
      })
    }
  }
  return items
})

const selectionMap = computed(
  () => new Map(selectableItems.value.map((v) => [v.id, v]))
)

const selectedValue = computed(() => {
  if (!store.selectedId) return undefined
  return selectionMap.value.get(store.selectedId)
})

function onSelectionChange(val: unknown) {
  const sv = val as SelectionValue | undefined
  store.select(sv?.id ?? null)
}

function doEmit() {
  const sel = selectedValue.value
  if (!sel) {
    emit('updateSelection', { canShowPreview: true })
    return
  }
  if (sel.kind === 'inProgress') {
    const item = store.inProgressItems.find((i) => i.id === sel.itemId)
    if (!item || item.state === 'skeleton') {
      emit('updateSelection', { canShowPreview: true })
    } else if (item.state === 'latent') {
      emit('updateSelection', {
        canShowPreview: true,
        latentPreviewUrl: item.latentPreviewUrl
      })
    } else {
      emit('updateSelection', {
        output: item.output,
        canShowPreview: true
      })
    }
    return
  }
  const asset = outputs.media.value.find((a) => a.id === sel.assetId)
  const output = asset ? toValue(allOutputs(asset))[sel.key] : undefined
  const isFirst = outputs.media.value[0]?.id === sel.assetId
  emit('updateSelection', {
    asset,
    output,
    canShowPreview: isFirst
  })
}

watchEffect(doEmit)

// Resolve in-progress items only when history outputs are loaded.
// Using watchEffect so it re-runs when allOutputs refs resolve (async).
watchEffect(() => {
  if (store.pendingResolve.size === 0) return
  for (const jobId of store.pendingResolve) {
    const asset = outputs.media.value.find((a) => {
      const m = getOutputAssetMetadata(a?.user_metadata)
      return m?.jobId === jobId
    })
    if (!asset) continue
    const loaded = toValue(allOutputs(asset)).length > 0
    if (loaded) {
      store.resolveIfReady(jobId, true)
      if (!store.selectedId) selectFirstHistory()
    }
  }
})

// Keep history selection stable on media changes
watch(
  () => outputs.media.value,
  (newAssets, oldAssets) => {
    if (
      newAssets.length === oldAssets.length ||
      (oldAssets.length === 0 && newAssets.length !== 1)
    )
      return

    if (store.selectedId?.startsWith('slot:')) return

    const sv = store.selectedId
      ? selectionMap.value.get(store.selectedId)
      : undefined

    if (!sv || sv.kind !== 'history') {
      selectFirstHistory()
      return
    }

    const wasFirst = sv.assetId === oldAssets[0]?.id
    if (wasFirst || !newAssets.some((a) => a.id === sv.assetId)) {
      selectFirstHistory()
    }
  }
)

function selectFirstHistory() {
  const first = outputs.media.value[0]
  if (first) {
    store.selectAsLatest(`history:${first.id}:0`)
  } else {
    store.selectAsLatest(null)
  }
}

const outputsRef = useTemplateRef('outputsRef')
useInfiniteScroll(outputsRef, outputs.loadMore, {
  canLoadMore: () => outputs.hasMore.value
})

// Reka UI's ListboxContent stops propagation on ALL Enter keydown events,
// which blocks modifier+Enter (Ctrl+Enter = run workflow) from reaching
// the global keybinding handler on window. Intercept in capture phase
// and re-dispatch from above the Listbox.
function onModifierEnter(e: KeyboardEvent) {
  if (e.key !== 'Enter' || !(e.ctrlKey || e.metaKey || e.shiftKey)) return
  e.stopImmediatePropagation()
  outputsRef.value?.parentElement?.dispatchEvent(
    new KeyboardEvent('keydown', {
      key: e.key,
      code: e.code,
      ctrlKey: e.ctrlKey,
      metaKey: e.metaKey,
      shiftKey: e.shiftKey,
      altKey: e.altKey,
      bubbles: true,
      cancelable: true
    })
  )
}

function navigateToAdjacent(direction: 1 | -1) {
  const items = selectableItems.value
  if (items.length === 0) return
  const currentId = store.selectedId
  const idx = currentId ? items.findIndex((i) => i.id === currentId) : -1
  const nextIdx =
    idx === -1 ? 0 : Math.max(0, Math.min(items.length - 1, idx + direction))
  const next = items[nextIdx]
  store.select(next.id)
  nextTick(() => listboxRef.value?.highlightItem(next))
}

const pointer = new CanvasPointer(document.body)
let scrollOffset = 0
useEventListener(
  document.body,
  'wheel',
  function (e: WheelEvent) {
    if (!e.ctrlKey && !e.metaKey) return
    e.preventDefault()
    e.stopPropagation()

    if (!pointer.isTrackpadGesture(e)) {
      if (e.deltaY > 0) navigateToAdjacent(1)
      else navigateToAdjacent(-1)
      return
    }
    scrollOffset += e.deltaY
    while (scrollOffset >= 60) {
      scrollOffset -= 60
      navigateToAdjacent(1)
    }
    while (scrollOffset <= -60) {
      scrollOffset += 60
      navigateToAdjacent(-1)
    }
  },
  { capture: true, passive: false }
)

useEventListener(
  outputsRef,
  'wheel',
  function (e: WheelEvent) {
    if (e.ctrlKey || e.metaKey || e.deltaY === 0) return
    e.preventDefault()
    if (outputsRef.value) outputsRef.value.scrollLeft += e.deltaY
  },
  { passive: false }
)

useEventListener(document.body, 'keydown', (e: KeyboardEvent) => {
  if (
    (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') ||
    e.target instanceof HTMLTextAreaElement ||
    e.target instanceof HTMLInputElement
  )
    return

  e.preventDefault()
  e.stopPropagation()
  if (e.key === 'ArrowDown') navigateToAdjacent(1)
  else navigateToAdjacent(-1)
})
</script>
<template>
  <ListboxRoot
    ref="listboxRef"
    :model-value="selectedValue"
    orientation="horizontal"
    selection-behavior="replace"
    by="id"
    class="min-w-0 px-4 pb-4"
    @update:model-value="onSelectionChange"
  >
    <ListboxContent as-child>
      <article
        ref="outputsRef"
        data-testid="linear-outputs"
        class="p-3 overflow-y-clip overflow-x-auto min-w-0"
        @keydown.capture="onModifierEnter"
      >
        <div class="flex items-center gap-0.5 mx-auto w-fit">
          <div v-if="queueCount > 0" class="shrink-0 flex items-center gap-0.5">
            <OutputHistoryActiveQueueItem :queue-count="queueCount" />
            <div
              v-if="hasActiveContent || visibleHistory.length > 0"
              class="border-l border-border-default h-12 shrink-0 mx-4"
            />
          </div>

          <ListboxItem
            v-for="item in store.inProgressItems"
            :key="`${item.id}-${item.state}`"
            :value="{
              id: `slot:${item.id}`,
              kind: 'inProgress',
              itemId: item.id
            }"
            :class="itemClass"
          >
            <OutputPreviewItem
              v-if="item.state !== 'image' || !item.output"
              :latent-preview="item.latentPreviewUrl"
            />
            <OutputHistoryItem v-else :output="item.output" />
          </ListboxItem>

          <div
            v-if="hasActiveContent && visibleHistory.length > 0"
            class="border-l border-border-default h-12 shrink-0 mx-4"
          />

          <template v-for="(asset, aIdx) in visibleHistory" :key="asset.id">
            <div
              v-if="aIdx > 0"
              class="border-l border-border-default h-12 shrink-0 mx-4"
            />
            <ListboxItem
              v-for="(output, key) in toValue(allOutputs(asset))"
              :key
              :value="{
                id: `history:${asset.id}:${key}`,
                kind: 'history',
                assetId: asset.id,
                key
              }"
              :class="itemClass"
            >
              <OutputHistoryItem :output="output" />
            </ListboxItem>
          </template>
        </div>
      </article>
    </ListboxContent>
  </ListboxRoot>
</template>
