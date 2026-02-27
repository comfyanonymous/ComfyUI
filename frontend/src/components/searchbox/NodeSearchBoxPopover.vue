<template>
  <div>
    <Dialog
      v-model:visible="visible"
      modal
      :dismissable-mask="dismissable"
      :pt="{
        root: {
          class: useSearchBoxV2
            ? 'w-4/5 min-w-[32rem] max-w-[56rem] border-0 bg-transparent mt-[10vh] max-md:w-[95%] max-md:min-w-0 overflow-visible'
            : 'invisible-dialog-root'
        },
        mask: {
          class: useSearchBoxV2 ? 'items-start' : 'node-search-box-dialog-mask'
        },
        transition: {
          enterFromClass: 'opacity-0 scale-75',
          // 100ms is the duration of the transition in the dialog component
          enterActiveClass: 'transition-all duration-100 ease-out',
          leaveActiveClass: 'transition-all duration-100 ease-in',
          leaveToClass: 'opacity-0 scale-75'
        }
      }"
      @hide="clearFilters"
    >
      <template #container>
        <div v-if="useSearchBoxV2" role="search" class="relative">
          <NodeSearchContent
            :filters="nodeFilters"
            @add-filter="addFilter"
            @remove-filter="removeFilter"
            @add-node="addNode"
            @hover-node="hoveredNodeDef = $event"
          />
          <NodePreviewCard
            v-if="hoveredNodeDef && enableNodePreview"
            :key="hoveredNodeDef.name"
            :node-def="hoveredNodeDef"
            show-category-path
            class="absolute top-0 left-full ml-3"
          />
        </div>
        <NodeSearchBox
          v-else
          :filters="nodeFilters"
          @add-filter="addFilter"
          @remove-filter="removeFilter"
          @add-node="addNode"
        />
      </template>
    </Dialog>
  </div>
</template>

<script setup lang="ts">
import { useEventListener, useWindowSize } from '@vueuse/core'
import { storeToRefs } from 'pinia'
import Dialog from 'primevue/dialog'
import { computed, ref, toRaw, watch, watchEffect } from 'vue'

import type { Point } from '@/lib/litegraph/src/interfaces'
import type { LiteGraphCanvasEvent } from '@/lib/litegraph/src/litegraph'
import { LGraphNode, LiteGraph } from '@/lib/litegraph/src/litegraph'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useLitegraphService } from '@/services/litegraphService'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { useSearchBoxStore } from '@/stores/workspace/searchBoxStore'
import { LinkReleaseTriggerAction } from '@/types/searchBoxTypes'
import type { FuseFilterWithValue } from '@/utils/fuseUtil'

import NodePreviewCard from '@/components/node/NodePreviewCard.vue'

import NodeSearchContent from './v2/NodeSearchContent.vue'
import NodeSearchBox from './NodeSearchBox.vue'

let triggerEvent: CanvasPointerEvent | null = null
let listenerController: AbortController | null = null
let disconnectOnReset = false

const settingStore = useSettingStore()
const searchBoxStore = useSearchBoxStore()
const litegraphService = useLitegraphService()

const { visible, newSearchBoxEnabled, useSearchBoxV2 } =
  storeToRefs(searchBoxStore)
const dismissable = ref(true)
const hoveredNodeDef = ref<ComfyNodeDefImpl | null>(null)
const { width: windowWidth } = useWindowSize()
// Minimum viewport width for the preview panel to fit beside the dialog
const MIN_WIDTH_FOR_PREVIEW = 1320
const enableNodePreview = computed(
  () =>
    useSearchBoxV2.value &&
    settingStore.get('Comfy.NodeSearchBoxImpl.NodePreview') &&
    windowWidth.value >= MIN_WIDTH_FOR_PREVIEW
)
function getNewNodeLocation(): Point {
  return triggerEvent
    ? [triggerEvent.canvasX, triggerEvent.canvasY]
    : litegraphService.getCanvasCenter()
}
const nodeFilters = ref<FuseFilterWithValue<ComfyNodeDefImpl, string>[]>([])
function addFilter(filter: FuseFilterWithValue<ComfyNodeDefImpl, string>) {
  const isDuplicate = nodeFilters.value.some(
    (f) => f.filterDef.id === filter.filterDef.id && f.value === filter.value
  )
  if (!isDuplicate) nodeFilters.value.push(filter)
}
function removeFilter(filter: FuseFilterWithValue<ComfyNodeDefImpl, string>) {
  nodeFilters.value = nodeFilters.value.filter(
    (f) => toRaw(f) !== toRaw(filter)
  )
}
function clearFilters() {
  nodeFilters.value = []
  hoveredNodeDef.value = null
}
function closeDialog() {
  visible.value = false
}
const canvasStore = useCanvasStore()

function addNode(nodeDef: ComfyNodeDefImpl, dragEvent?: MouseEvent) {
  const node = litegraphService.addNodeOnGraph(
    nodeDef,
    { pos: getNewNodeLocation() },
    { ghost: useSearchBoxV2.value, dragEvent }
  )
  if (!node) return

  if (disconnectOnReset && triggerEvent) {
    canvasStore.getCanvas().linkConnector.connectToNode(node, triggerEvent)
  } else if (!triggerEvent) {
    console.warn('The trigger event was undefined when addNode was called.')
  }

  disconnectOnReset = false

  // Notify changeTracker - new step should be added
  useWorkflowStore().activeWorkflow?.changeTracker?.checkState()
  window.requestAnimationFrame(closeDialog)
}

function showSearchBox(e: CanvasPointerEvent | null) {
  if (newSearchBoxEnabled.value) {
    if (e?.pointerType === 'touch') {
      setTimeout(() => {
        showNewSearchBox(e)
      }, 128)
    } else {
      showNewSearchBox(e)
    }
  } else {
    canvasStore.getCanvas().showSearchBox(e)
  }
}

function getFirstLink() {
  return canvasStore.getCanvas().linkConnector.renderLinks.at(0)
}

const nodeDefStore = useNodeDefStore()
function showNewSearchBox(e: CanvasPointerEvent | null) {
  const firstLink = getFirstLink()
  if (firstLink) {
    const filter =
      firstLink.toType === 'input'
        ? nodeDefStore.nodeSearchService.inputTypeFilter
        : nodeDefStore.nodeSearchService.outputTypeFilter

    const dataType = firstLink.fromSlot.type?.toString() ?? ''
    addFilter({
      filterDef: filter,
      value: dataType
    })
  }

  visible.value = true
  triggerEvent = e

  // Prevent the dialog from being dismissed immediately
  dismissable.value = false
  setTimeout(() => {
    dismissable.value = true
  }, 300)
}

function showContextMenu(e: CanvasPointerEvent) {
  const firstLink = getFirstLink()
  if (!firstLink) return

  const { node, fromSlot, toType } = firstLink
  const commonOptions = {
    e,
    allow_searchbox: true,
    showSearchBox: () => {
      cancelResetOnContextClose()
      showSearchBox(e)
    }
  }
  const afterRerouteId = firstLink.fromReroute?.id
  const connectionOptions =
    toType === 'input'
      ? { nodeFrom: node, slotFrom: fromSlot, afterRerouteId }
      : { nodeTo: node, slotTo: fromSlot, afterRerouteId }

  const canvas = canvasStore.getCanvas()
  const menu = canvas.showConnectionMenu({
    ...connectionOptions,
    ...commonOptions
  })

  if (!menu) {
    console.warn('No menu was returned from showConnectionMenu')
    return
  }

  triggerEvent = e
  listenerController = new AbortController()
  const { signal } = listenerController
  const options = { once: true, signal }

  // Connect the node after it is created via context menu
  useEventListener(
    canvas.canvas,
    'connect-new-default-node',
    (createEvent) => {
      if (!(createEvent instanceof CustomEvent))
        throw new Error('Invalid event')

      const node: unknown = createEvent.detail?.node
      if (!(node instanceof LGraphNode)) throw new Error('Invalid node')

      disconnectOnReset = false
      createEvent.preventDefault()
      canvas.linkConnector.connectToNode(node, e)
    },
    options
  )

  // Reset when the context menu is closed
  const cancelResetOnContextClose = useEventListener(
    menu.controller.signal,
    'abort',
    reset,
    options
  )
}

// Disable litegraph's default behavior of release link and search box.
watchEffect(() => {
  const { canvas } = canvasStore
  if (!canvas) return

  LiteGraph.release_link_on_empty_shows_menu = false
  canvas.allow_searchbox = false

  useEventListener(
    canvas.linkConnector.events,
    'dropped-on-canvas',
    handleDroppedOnCanvas
  )
})

function canvasEventHandler(e: LiteGraphCanvasEvent) {
  if (e.detail.subType === 'empty-double-click') {
    showSearchBox(e.detail.originalEvent)
  } else if (e.detail.subType === 'group-double-click') {
    const group = e.detail.group
    const [_, y] = group.pos
    const relativeY = e.detail.originalEvent.canvasY - y
    // Show search box if the click is NOT on the title bar
    if (relativeY > group.titleHeight) {
      showSearchBox(e.detail.originalEvent)
    }
  }
}

const linkReleaseAction = computed(() =>
  settingStore.get('Comfy.LinkRelease.Action')
)

const linkReleaseActionShift = computed(() =>
  settingStore.get('Comfy.LinkRelease.ActionShift')
)

// Prevent normal LinkConnector reset (called by CanvasPointer.finally)
function preventDefault(e: Event) {
  return e.preventDefault()
}
function cancelNextReset(e: CustomEvent<CanvasPointerEvent>) {
  e.preventDefault()

  const canvas = canvasStore.getCanvas()
  canvas.linkConnector.state.snapLinksPos = [e.detail.canvasX, e.detail.canvasY]
  useEventListener(canvas.linkConnector.events, 'reset', preventDefault, {
    once: true
  })
}

function handleDroppedOnCanvas(e: CustomEvent<CanvasPointerEvent>) {
  disconnectOnReset = true
  const action = e.detail.shiftKey
    ? linkReleaseActionShift.value
    : linkReleaseAction.value
  switch (action) {
    case LinkReleaseTriggerAction.SEARCH_BOX:
      cancelNextReset(e)
      showSearchBox(e.detail)
      break
    case LinkReleaseTriggerAction.CONTEXT_MENU:
      cancelNextReset(e)
      showContextMenu(e.detail)
      break
    case LinkReleaseTriggerAction.NO_ACTION:
    default:
      break
  }
}

// Resets litegraph state
function reset() {
  listenerController?.abort()
  listenerController = null
  triggerEvent = null

  const canvas = canvasStore.getCanvas()
  canvas.linkConnector.events.removeEventListener('reset', preventDefault)
  if (disconnectOnReset) canvas.linkConnector.disconnectLinks()

  canvas.linkConnector.reset()
  canvas.setDirty(true, true)
}

// Reset connecting links when the search box is closed
watch(visible, () => {
  if (!visible.value) reset()
})

useEventListener(document, 'litegraph:canvas', canvasEventHandler)
defineExpose({ showSearchBox })
</script>

<style>
.invisible-dialog-root {
  width: 60%;
  min-width: 24rem;
  max-width: 48rem;
  border: 0 !important;
  background-color: transparent !important;
  margin-top: 25vh;
  margin-left: 400px;
}
@media all and (max-width: 768px) {
  .invisible-dialog-root {
    margin-left: 0;
  }
}

.node-search-box-dialog-mask {
  align-items: flex-start !important;
}
</style>
