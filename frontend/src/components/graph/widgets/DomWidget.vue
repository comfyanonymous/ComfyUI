<template>
  <div
    v-show="widgetState.visible"
    ref="widgetElement"
    class="dom-widget h-full w-full"
    :title="tooltip"
    :style="style"
  >
    <component
      :is="widget.component"
      v-if="isComponentWidget(widget)"
      class="h-full w-full"
      :model-value="widget.value"
      :widget="widget"
      v-bind="widget.props"
      @update:model-value="emit('update:widgetValue', $event)"
    />
  </div>
</template>

<script setup lang="ts">
import { useElementBounding, useEventListener, whenever } from '@vueuse/core'
import type { CSSProperties } from 'vue'
import { computed, nextTick, onMounted, ref, watch } from 'vue'

import { useAbsolutePosition } from '@/composables/element/useAbsolutePosition'
import { useDomClipping } from '@/composables/element/useDomClipping'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { isComponentWidget, isDOMWidget } from '@/scripts/domWidget'
import type { DomWidgetState } from '@/stores/domWidgetStore'

const { widgetState } = defineProps<{
  widgetState: DomWidgetState
}>()
const widget = widgetState.widget

const emit = defineEmits<{
  'update:widgetValue': [value: string | object]
}>()

const widgetElement = ref<HTMLElement | undefined>()

/**
 * @note Do NOT convert style to a computed value, as it will cause lag when
 * updating the style on different animation frames. Vue's computed value is
 * evaluated asynchronously.
 */
const style = ref<CSSProperties>({})
const { style: positionStyle, updatePosition } = useAbsolutePosition({
  useTransform: true
})
const { style: clippingStyle, updateClipPath } = useDomClipping()

const canvasStore = useCanvasStore()
const settingStore = useSettingStore()
const enableDomClipping = computed(() =>
  settingStore.get('Comfy.DOMClippingEnabled')
)

const updateDomClipping = () => {
  const lgCanvas = canvasStore.canvas
  if (!lgCanvas || !widgetElement.value) return

  const selectedNode = Object.values(lgCanvas.selected_nodes ?? {})[0]
  if (!selectedNode) {
    // Clear clipping when no node is selected
    updateClipPath(widgetElement.value, lgCanvas.canvas, false, undefined)
    return
  }

  const override = widgetState.positionOverride
  const overrideInGraph =
    override && lgCanvas.graph?.getNodeById(override.node.id)
  const ownerNode = overrideInGraph ? override.node : widgetState.widget.node
  const isSelected = selectedNode === ownerNode
  const renderArea = selectedNode?.renderArea
  const offset = lgCanvas.ds.offset
  const scale = lgCanvas.ds.scale
  const selectedAreaConfig = renderArea
    ? {
        x: renderArea[0],
        y: renderArea[1],
        width: renderArea[2],
        height: renderArea[3],
        scale,
        offset: [offset[0], offset[1]] as [number, number]
      }
    : undefined

  updateClipPath(
    widgetElement.value,
    lgCanvas.canvas,
    isSelected,
    selectedAreaConfig
  )
}

/**
 * @note mapping between canvas position and client position depends on the
 * canvas element's position, so we need to watch the canvas element's position
 * and update the position of the widget accordingly.
 */
const { left, top } = useElementBounding(canvasStore.getCanvas().canvas)
watch(
  [() => widgetState, left, top],
  ([widgetState, _, __]) => {
    updatePosition(widgetState)
    if (enableDomClipping.value) {
      updateDomClipping()
    }

    style.value = {
      ...positionStyle.value,
      ...(enableDomClipping.value ? clippingStyle.value : {}),
      zIndex: widgetState.zIndex,
      pointerEvents:
        widgetState.readonly || widget.computedDisabled ? 'none' : 'auto',
      opacity: widget.computedDisabled ? 0.5 : 1
    }
  },
  { deep: true }
)

watch(
  () => widgetState.visible,
  (newVisible, oldVisible) => {
    if (!newVisible && oldVisible) {
      widget.options.onHide?.(widget)
    }
  }
)
useEventListener(document, 'mousedown', (event) => {
  if (!isDOMWidget(widget) || !widgetState.visible || !widget.element.blur) {
    return
  }
  if (!widget.element.contains(event.target as HTMLElement)) {
    widget.element.blur()
  }
})

onMounted(() => {
  if (!isDOMWidget(widget)) {
    return
  }
  useEventListener(
    widget.element,
    widget.options.selectOn ?? ['focus', 'click'],
    () => {
      const lgCanvas = canvasStore.canvas
      if (!lgCanvas) return

      const override = widgetState.positionOverride
      const overrideInGraph =
        override && lgCanvas.graph?.getNodeById(override.node.id)
      const ownerNode = overrideInGraph
        ? override.node
        : widgetState.widget.node

      lgCanvas.selectNode(ownerNode)
      lgCanvas.bringToFront(ownerNode)
    }
  )
})

const inputSpec = widget.node.constructor.nodeData
const tooltip = inputSpec?.inputs?.[widget.name]?.tooltip

// Mount DOM element when widget is or becomes visible
const mountElementIfVisible = () => {
  if (!(widgetState.visible && isDOMWidget(widget) && widgetElement.value)) {
    return
  }
  // Only append if not already a child
  if (widgetElement.value.contains(widget.element)) {
    return
  }

  widget.element.classList.add('h-full', 'w-full')
  widgetElement.value.appendChild(widget.element)
}

// Check on mount - but only after next tick to ensure visibility is calculated
onMounted(() => {
  nextTick(() => {
    mountElementIfVisible()
  }).catch((error) => {
    console.error('Error mounting DOM widget element:', error)
  })
})

// And watch for visibility changes
watch(
  () => widgetState.visible,
  () => {
    mountElementIfVisible()
  }
)

whenever(() => !canvasStore.linearMode, mountElementIfVisible)
</script>
