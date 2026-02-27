<template>
  <div v-if="renderError" class="node-error p-2 text-sm text-red-500">
    {{ st('nodeErrors.widgets', 'Node Widgets Error') }}
  </div>
  <div
    v-else
    :class="
      cn(
        'lg-node-widgets grid grid-cols-[min-content_minmax(80px,min-content)_minmax(125px,1fr)] gap-y-1 pr-3',
        shouldHandleNodePointerEvents
          ? 'pointer-events-auto'
          : 'pointer-events-none'
      )
    "
    :style="{
      'grid-template-rows': gridTemplateRows,
      flex: gridTemplateRows.includes('auto') ? 1 : undefined
    }"
    @pointerdown.capture="handleBringToFront"
    @pointerdown="handleWidgetPointerEvent"
    @pointermove="handleWidgetPointerEvent"
    @pointerup="handleWidgetPointerEvent"
  >
    <template
      v-for="(widget, index) in processedWidgets"
      :key="`widget-${index}-${widget.name}`"
    >
      <div
        v-if="!widget.hidden && (!widget.advanced || showAdvanced)"
        class="lg-node-widget group col-span-full grid grid-cols-subgrid items-stretch"
        :data-widget-name="widget.name"
      >
        <!-- Widget Input Slot Dot -->
        <div
          :class="
            cn(
              'z-10 w-3 opacity-0 transition-opacity duration-150 group-hover:opacity-100 flex items-stretch',
              widget.slotMetadata?.linked && 'opacity-100'
            )
          "
        >
          <InputSlot
            v-if="widget.slotMetadata"
            :slot-data="{
              name: widget.name,
              type: widget.type,
              boundingRect: [0, 0, 0, 0]
            }"
            :node-id="nodeData?.id != null ? String(nodeData.id) : ''"
            :has-error="widget.hasError"
            :index="widget.slotMetadata.index"
            :socketless="widget.simplified.spec?.socketless"
            dot-only
          />
        </div>
        <!-- Widget Component -->
        <component
          :is="widget.vueComponent"
          v-model="widget.value"
          v-tooltip.left="widget.tooltipConfig"
          :widget="widget.simplified"
          :node-id="nodeData?.id != null ? String(nodeData.id) : ''"
          :node-type="nodeType"
          :class="
            cn(
              'col-span-2',
              widget.hasError && 'text-node-stroke-error font-bold'
            )
          "
          @update:model-value="widget.updateHandler"
        />
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import type { TooltipOptions } from 'primevue'
import { computed, onErrorCaptured, ref, toValue } from 'vue'
import type { Component } from 'vue'

import type {
  VueNodeData,
  WidgetSlotMetadata
} from '@/composables/graph/useGraphNodeManager'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useErrorHandling } from '@/composables/useErrorHandling'
import { st } from '@/i18n'
import { useCanvasInteractions } from '@/renderer/core/canvas/useCanvasInteractions'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useNodeTooltips } from '@/renderer/extensions/vueNodes/composables/useNodeTooltips'
import { useNodeZIndex } from '@/renderer/extensions/vueNodes/composables/useNodeZIndex'
import WidgetDOM from '@/renderer/extensions/vueNodes/widgets/components/WidgetDOM.vue'
// Import widget components directly
import WidgetLegacy from '@/renderer/extensions/vueNodes/widgets/components/WidgetLegacy.vue'
import {
  getComponent,
  shouldExpand,
  shouldRenderAsVue
} from '@/renderer/extensions/vueNodes/widgets/registry/widgetRegistry'
import {
  stripGraphPrefix,
  useWidgetValueStore
} from '@/stores/widgetValueStore'
import { usePromotionStore } from '@/stores/promotionStore'
import { useExecutionErrorStore } from '@/stores/executionErrorStore'
import type { SimplifiedWidget, WidgetValue } from '@/types/simplifiedWidget'
import { cn } from '@/utils/tailwindUtil'

import InputSlot from './InputSlot.vue'

interface NodeWidgetsProps {
  nodeData?: VueNodeData
}

const { nodeData } = defineProps<NodeWidgetsProps>()

const { shouldHandleNodePointerEvents, forwardEventToCanvas } =
  useCanvasInteractions()
const canvasStore = useCanvasStore()
const { bringNodeToFront } = useNodeZIndex()
const promotionStore = usePromotionStore()
const executionErrorStore = useExecutionErrorStore()

function handleWidgetPointerEvent(event: PointerEvent) {
  if (shouldHandleNodePointerEvents.value) return
  event.stopPropagation()
  forwardEventToCanvas(event)
}

function handleBringToFront() {
  if (nodeData?.id != null) {
    bringNodeToFront(String(nodeData.id))
  }
}

// Error boundary implementation
const renderError = ref<string | null>(null)

const { toastErrorHandler } = useErrorHandling()

onErrorCaptured((error) => {
  renderError.value = error.message
  toastErrorHandler(error)
  return false
})

const nodeType = computed(() => nodeData?.type || '')
const settingStore = useSettingStore()
const showAdvanced = computed(
  () =>
    nodeData?.showAdvanced ||
    settingStore.get('Comfy.Node.AlwaysShowAdvancedWidgets')
)
const { getWidgetTooltip, createTooltipConfig } = useNodeTooltips(
  nodeType.value
)
const widgetValueStore = useWidgetValueStore()

interface ProcessedWidget {
  advanced: boolean
  hasLayoutSize: boolean
  hasError: boolean
  hidden: boolean
  name: string
  simplified: SimplifiedWidget
  tooltipConfig: TooltipOptions
  type: string
  updateHandler: (value: WidgetValue) => void
  value: WidgetValue
  vueComponent: Component
  slotMetadata?: WidgetSlotMetadata
}

const processedWidgets = computed((): ProcessedWidget[] => {
  if (!nodeData?.widgets) return []
  const nodeErrors = executionErrorStore.lastNodeErrors?.[nodeData.id ?? '']
  const graphId = canvasStore.canvas?.graph?.rootGraph.id

  const nodeId = nodeData.id
  const { widgets } = nodeData
  const result: ProcessedWidget[] = []

  for (const widget of widgets) {
    if (!shouldRenderAsVue(widget)) continue

    const vueComponent =
      getComponent(widget.type) ||
      (widget.isDOMWidget ? WidgetDOM : WidgetLegacy)

    const { slotMetadata } = widget

    // Get metadata from store (registered during BaseWidget.setNodeId)
    const bareWidgetId = stripGraphPrefix(widget.nodeId ?? nodeId)
    const widgetState = graphId
      ? widgetValueStore.getWidget(graphId, bareWidgetId, widget.name)
      : undefined

    // Get value from store (falls back to undefined if not registered)
    const value = widgetState?.value as WidgetValue

    // Build options from store state, with slot-linked override for disabled
    const storeOptions = widgetState?.options ?? {}
    const widgetOptions = slotMetadata?.linked
      ? { ...storeOptions, disabled: true }
      : storeOptions

    const isPromotedView = !!widget.nodeId
    const borderStyle =
      graphId &&
      !isPromotedView &&
      promotionStore.isPromotedByAny(graphId, String(bareWidgetId), widget.name)
        ? 'ring ring-component-node-widget-promoted'
        : widget.options?.advanced
          ? 'ring ring-component-node-widget-advanced'
          : undefined

    const simplified: SimplifiedWidget = {
      name: widget.name,
      type: widget.type,
      value,
      borderStyle,
      callback: widget.callback,
      controlWidget: widget.controlWidget,
      label: widgetState?.label,
      options: widgetOptions,
      spec: widget.spec
    }

    function updateHandler(newValue: WidgetValue) {
      // Update value in store
      if (widgetState) widgetState.value = newValue
      // Invoke LiteGraph callback wrapper (handles triggerDraw, etc.)
      widget.callback?.(newValue)
    }

    const tooltipText = getWidgetTooltip(widget)
    const tooltipConfig = createTooltipConfig(tooltipText)

    result.push({
      advanced: widget.options?.advanced ?? false,
      hasLayoutSize: widget.hasLayoutSize ?? false,
      hasError:
        nodeErrors?.errors?.some(
          (error) => error.extra_info?.input_name === widget.name
        ) ?? false,
      hidden: widget.options?.hidden ?? false,
      name: widget.name,
      type: widget.type,
      vueComponent,
      simplified,
      value,
      updateHandler,
      tooltipConfig,
      slotMetadata
    })
  }

  return result
})

const gridTemplateRows = computed((): string => {
  // Use processedWidgets directly since it already has store-based hidden/advanced
  return toValue(processedWidgets)
    .filter((w) => !w.hidden && (!w.advanced || showAdvanced.value))
    .map((w) =>
      shouldExpand(w.type) || w.hasLayoutSize ? 'auto' : 'min-content'
    )
    .join(' ')
})
</script>
