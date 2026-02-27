<template>
  <div class="flex h-full items-center" :class="cn(!isDocked && '-ml-2')">
    <div
      v-if="isDragging && !isDocked"
      :class="actionbarClass"
      @mouseenter="onMouseEnterDropZone"
      @mouseleave="onMouseLeaveDropZone"
    >
      {{ t('actionbar.dockToTop') }}
    </div>

    <Panel
      ref="panelRef"
      class="pointer-events-auto"
      :style="style"
      :class="panelClass"
      :pt="{
        header: { class: 'hidden' },
        content: { class: isDocked ? 'p-0' : 'p-1' }
      }"
    >
      <div class="relative flex items-center select-none gap-2">
        <span
          ref="dragHandleRef"
          :class="
            cn(
              'drag-handle cursor-grab w-3 h-max',
              isDragging && 'cursor-grabbing'
            )
          "
        />
        <Suspense @resolve="comfyRunButtonResolved">
          <ComfyRunButton />
        </Suspense>
        <Button
          v-tooltip.bottom="cancelJobTooltipConfig"
          variant="destructive"
          size="icon"
          :disabled="isExecutionIdle"
          :aria-label="t('menu.interrupt')"
          @click="cancelCurrentJob"
        >
          <i class="icon-[lucide--x] size-4" />
        </Button>
        <Button
          v-tooltip.bottom="queueHistoryTooltipConfig"
          variant="secondary"
          size="md"
          :aria-pressed="
            isQueuePanelV2Enabled
              ? activeSidebarTabId === 'job-history'
              : queueOverlayExpanded
          "
          class="relative px-3"
          data-testid="queue-overlay-toggle"
          @click="toggleQueueOverlay"
          @contextmenu.stop.prevent="showQueueContextMenu"
        >
          <span class="text-sm font-normal tabular-nums">
            {{ activeJobsLabel }}
          </span>
          <StatusBadge
            v-if="activeJobsCount > 0"
            data-testid="active-jobs-indicator"
            variant="dot"
            class="pointer-events-none absolute -top-0.5 -right-0.5 animate-pulse"
          />
          <span class="sr-only">
            {{
              isQueuePanelV2Enabled
                ? t('sideToolbar.queueProgressOverlay.viewJobHistory')
                : t('sideToolbar.queueProgressOverlay.expandCollapsedQueue')
            }}
          </span>
        </Button>
        <ContextMenu ref="queueContextMenu" :model="queueContextMenuItems" />
      </div>
    </Panel>

    <Teleport v-if="inlineProgressTarget" :to="inlineProgressTarget">
      <QueueInlineProgress
        :hidden="shouldHideInlineProgress"
        :radius-class="cn(isDocked ? 'rounded-[7px]' : 'rounded-[5px]')"
        data-testid="queue-inline-progress"
      />
    </Teleport>
  </div>
</template>

<script lang="ts" setup>
import {
  useDraggable,
  useEventListener,
  useLocalStorage,
  unrefElement,
  watchDebounced
} from '@vueuse/core'
import { clamp } from 'es-toolkit/compat'
import { storeToRefs } from 'pinia'
import ContextMenu from 'primevue/contextmenu'
import type { MenuItem } from 'primevue/menuitem'
import Panel from 'primevue/panel'
import { computed, nextTick, ref, watch } from 'vue'
import type { ComponentPublicInstance } from 'vue'
import { useI18n } from 'vue-i18n'

import StatusBadge from '@/components/common/StatusBadge.vue'
import QueueInlineProgress from '@/components/queue/QueueInlineProgress.vue'
import Button from '@/components/ui/button/Button.vue'
import { buildTooltipConfig } from '@/composables/useTooltipConfig'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useTelemetry } from '@/platform/telemetry'
import { useCommandStore } from '@/stores/commandStore'
import { useExecutionStore } from '@/stores/executionStore'
import { useQueueStore } from '@/stores/queueStore'
import { useSidebarTabStore } from '@/stores/workspace/sidebarTabStore'
import { cn } from '@/utils/tailwindUtil'

import ComfyRunButton from './ComfyRunButton'

const { topMenuContainer, queueOverlayExpanded = false } = defineProps<{
  topMenuContainer?: HTMLElement | null
  queueOverlayExpanded?: boolean
}>()

const emit = defineEmits<{
  (event: 'update:progressTarget', target: HTMLElement | null): void
}>()

const settingsStore = useSettingStore()
const commandStore = useCommandStore()
const executionStore = useExecutionStore()
const queueStore = useQueueStore()
const sidebarTabStore = useSidebarTabStore()
const { t, n } = useI18n()
const { isIdle: isExecutionIdle } = storeToRefs(executionStore)
const { activeJobsCount } = storeToRefs(queueStore)
const { activeSidebarTabId } = storeToRefs(sidebarTabStore)

const position = computed(() => settingsStore.get('Comfy.UseNewMenu'))
const visible = computed(() => position.value !== 'Disabled')
const isQueuePanelV2Enabled = computed(() =>
  settingsStore.get('Comfy.Queue.QPOV2')
)

const panelRef = ref<ComponentPublicInstance | null>(null)
const panelElement = computed<HTMLElement | null>(() => {
  const element = unrefElement(panelRef)
  return element instanceof HTMLElement ? element : null
})
const dragHandleRef = ref<HTMLElement | null>(null)
const isDocked = useLocalStorage('Comfy.MenuPosition.Docked', true)
const storedPosition = useLocalStorage('Comfy.MenuPosition.Floating', {
  x: 0,
  y: 0
})
const { x, y, style, isDragging } = useDraggable(panelElement, {
  initialValue: { x: 0, y: 0 },
  handle: dragHandleRef,
  containerElement: document.body
})

// Update storedPosition when x or y changes
watchDebounced(
  [x, y],
  ([newX, newY]) => {
    storedPosition.value = { x: newX, y: newY }
  },
  { debounce: 300 }
)

// Set initial position to bottom center
const setInitialPosition = () => {
  const panel = panelElement.value
  if (panel) {
    const screenWidth = window.innerWidth
    const screenHeight = window.innerHeight
    const menuWidth = panel.offsetWidth
    const menuHeight = panel.offsetHeight

    if (menuWidth === 0 || menuHeight === 0) {
      return
    }

    // Check if stored position exists and is within bounds
    if (storedPosition.value.x !== 0 || storedPosition.value.y !== 0) {
      // Ensure stored position is within screen bounds
      x.value = clamp(storedPosition.value.x, 0, screenWidth - menuWidth)
      y.value = clamp(storedPosition.value.y, 0, screenHeight - menuHeight)
      captureLastDragState()
      return
    }

    // If no stored position or current position, set to bottom center
    if (x.value === 0 && y.value === 0) {
      x.value = clamp((screenWidth - menuWidth) / 2, 0, screenWidth - menuWidth)
      y.value = clamp(
        screenHeight - menuHeight - 10,
        0,
        screenHeight - menuHeight
      )
      captureLastDragState()
    }
  }
}

//The ComfyRunButton is a dynamic import. Which means it will not be loaded onMount in this component.
//So we must use suspense resolve to ensure that is has loaded and updated the DOM before calling setInitialPosition()
async function comfyRunButtonResolved() {
  await nextTick()
  setInitialPosition()
}

watch(visible, async (newVisible) => {
  if (newVisible) {
    await nextTick(setInitialPosition)
  }
})

/**
 * Track run button handle drag start using mousedown on the drag handle.
 */
useEventListener(dragHandleRef, 'mousedown', () => {
  useTelemetry()?.trackUiButtonClicked({
    button_id: 'actionbar_run_handle_drag_start'
  })
})

const lastDragState = ref({
  x: x.value,
  y: y.value,
  windowWidth: window.innerWidth,
  windowHeight: window.innerHeight
})
const captureLastDragState = () => {
  lastDragState.value = {
    x: x.value,
    y: y.value,
    windowWidth: window.innerWidth,
    windowHeight: window.innerHeight
  }
}
watch(
  isDragging,
  (newIsDragging) => {
    if (!newIsDragging) {
      // Stop dragging
      captureLastDragState()
    }
  },
  { immediate: true }
)

const adjustMenuPosition = () => {
  const panel = panelElement.value
  if (panel) {
    const screenWidth = window.innerWidth
    const screenHeight = window.innerHeight
    const menuWidth = panel.offsetWidth
    const menuHeight = panel.offsetHeight

    // Calculate distances to all edges
    const distanceLeft = lastDragState.value.x
    const distanceRight =
      lastDragState.value.windowWidth - (lastDragState.value.x + menuWidth)
    const distanceTop = lastDragState.value.y
    const distanceBottom =
      lastDragState.value.windowHeight - (lastDragState.value.y + menuHeight)

    // Find the smallest distance to determine which edge to anchor to
    const distances = [
      { edge: 'left', distance: distanceLeft },
      { edge: 'right', distance: distanceRight },
      { edge: 'top', distance: distanceTop },
      { edge: 'bottom', distance: distanceBottom }
    ]
    const closestEdge = distances.reduce((min, curr) =>
      curr.distance < min.distance ? curr : min
    )

    // Calculate vertical position as a percentage of screen height
    const verticalRatio =
      lastDragState.value.y / lastDragState.value.windowHeight
    const horizontalRatio =
      lastDragState.value.x / lastDragState.value.windowWidth

    // Apply positioning based on closest edge
    if (closestEdge.edge === 'left') {
      x.value = closestEdge.distance // Maintain exact distance from left
      y.value = verticalRatio * screenHeight
    } else if (closestEdge.edge === 'right') {
      x.value = screenWidth - menuWidth - closestEdge.distance // Maintain exact distance from right
      y.value = verticalRatio * screenHeight
    } else if (closestEdge.edge === 'top') {
      x.value = horizontalRatio * screenWidth
      y.value = closestEdge.distance // Maintain exact distance from top
    } else {
      // bottom
      x.value = horizontalRatio * screenWidth
      y.value = screenHeight - menuHeight - closestEdge.distance // Maintain exact distance from bottom
    }

    // Ensure the menu stays within the screen bounds
    x.value = clamp(x.value, 0, screenWidth - menuWidth)
    y.value = clamp(y.value, 0, screenHeight - menuHeight)
  }
}

useEventListener(window, 'resize', adjustMenuPosition)

// Drop zone state
const isMouseOverDropZone = ref(false)

// Mouse event handlers for self-contained drop zone
const onMouseEnterDropZone = () => {
  if (isDragging.value) {
    isMouseOverDropZone.value = true
  }
}

const onMouseLeaveDropZone = () => {
  if (isDragging.value) {
    isMouseOverDropZone.value = false
  }
}

const inlineProgressTarget = computed(() => {
  if (!visible.value || !isQueuePanelV2Enabled.value) return null
  if (isDocked.value) return topMenuContainer ?? null
  return panelElement.value
})
const shouldHideInlineProgress = computed(
  () => !isQueuePanelV2Enabled.value && queueOverlayExpanded
)
watch(
  panelElement,
  (target) => {
    emit('update:progressTarget', target)
  },
  { immediate: true }
)

// Handle drag state changes
watch(isDragging, (dragging) => {
  if (dragging) {
    // Starting to drag - undock if docked
    if (isDocked.value) {
      isDocked.value = false
    }
  } else {
    // Stopped dragging - dock if mouse is over drop zone
    if (isMouseOverDropZone.value) {
      isDocked.value = true
    }
    // Reset drop zone state
    isMouseOverDropZone.value = false
  }
})

const cancelJobTooltipConfig = computed(() =>
  buildTooltipConfig(t('menu.interrupt'))
)
const queueHistoryTooltipConfig = computed(() =>
  buildTooltipConfig(t('sideToolbar.queueProgressOverlay.viewJobHistory'))
)
const activeJobsLabel = computed(() => {
  const count = activeJobsCount.value
  return t(
    'sideToolbar.queueProgressOverlay.activeJobsShort',
    { count: n(count) },
    count
  )
})
const queueContextMenu = ref<InstanceType<typeof ContextMenu> | null>(null)
const queueContextMenuItems = computed<MenuItem[]>(() => [
  {
    label: t('sideToolbar.queueProgressOverlay.clearQueueTooltip'),
    icon: 'icon-[lucide--list-x] text-destructive-background',
    class: '*:text-destructive-background',
    disabled: queueStore.pendingTasks.length === 0,
    command: () => {
      void handleClearQueue()
    }
  }
])

const cancelCurrentJob = async () => {
  if (isExecutionIdle.value) return
  await commandStore.execute('Comfy.Interrupt')
}
const toggleQueueOverlay = () => {
  if (isQueuePanelV2Enabled.value) {
    sidebarTabStore.toggleSidebarTab('job-history')
    return
  }
  commandStore.execute('Comfy.Queue.ToggleOverlay')
}
const showQueueContextMenu = (event: MouseEvent) => {
  queueContextMenu.value?.show(event)
}
const handleClearQueue = async () => {
  const pendingJobIds = queueStore.pendingTasks
    .map((task) => task.jobId)
    .filter((id): id is string => typeof id === 'string' && id.length > 0)

  await commandStore.execute('Comfy.ClearPendingTasks')
  executionStore.clearInitializationByJobIds(pendingJobIds)
}

const actionbarClass = computed(() =>
  cn(
    'w-[200px] border-dashed border-blue-500 opacity-80',
    'm-1.5 flex items-center justify-center self-stretch',
    'rounded-md before:w-50 before:-ml-50 before:h-full',
    'pointer-events-auto',
    isMouseOverDropZone.value &&
      'border-[3px] opacity-100 scale-105 shadow-[0_0_20px] shadow-blue-500'
  )
)
const panelClass = computed(() =>
  cn(
    'actionbar pointer-events-auto z-1300',
    isDragging.value && 'select-none pointer-events-none',
    isDocked.value
      ? 'p-0 static border-none bg-transparent'
      : 'fixed shadow-interface'
  )
)
</script>
