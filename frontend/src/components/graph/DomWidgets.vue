<template>
  <!-- Create a new stacking context for widgets to avoid z-index issues -->
  <div class="isolate">
    <DomWidget
      v-for="widgetState in widgetStates"
      :key="widgetState.widget.id"
      :widget-state="widgetState"
      @update:widget-value="widgetState.widget.value = $event"
    />
  </div>
</template>

<script setup lang="ts">
import { whenever } from '@vueuse/core'
import { computed } from 'vue'

import DomWidget from '@/components/graph/widgets/DomWidget.vue'
import { getDomWidgetZIndex } from '@/components/graph/widgets/domWidgetZIndex'
import { useChainCallback } from '@/composables/functional/useChainCallback'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useDomWidgetStore } from '@/stores/domWidgetStore'

const domWidgetStore = useDomWidgetStore()
const overrideTransitionGrace = new Set<string>()

const widgetStates = computed(() => [...domWidgetStore.widgetStates.values()])

const updateWidgets = () => {
  const lgCanvas = canvasStore.canvas
  if (!lgCanvas) return

  const lowQuality = lgCanvas.low_quality
  const currentGraph = lgCanvas.graph
  const seenWidgetIds = new Set<string>()

  for (const widgetState of widgetStates.value) {
    const widget = widgetState.widget
    seenWidgetIds.add(widget.id)

    // Use position override only when the override node (SubgraphNode) is
    // in the current graph. When the user enters the subgraph, the override
    // node is no longer visible — fall back to the widget's own node.
    // Use graph reference equality (IDs are not unique across graphs).
    const override = widgetState.positionOverride
    const useOverride = !!override && currentGraph === override.node.graph
    const inOverrideTransitionGap =
      !!override && !useOverride && !widgetState.active
    const useTransitionGrace =
      inOverrideTransitionGap && !overrideTransitionGrace.has(widget.id)

    if (useTransitionGrace) {
      overrideTransitionGrace.add(widget.id)
    } else if (!inOverrideTransitionGap) {
      overrideTransitionGrace.delete(widget.id)
    }

    // Early exit for non-visible widgets.
    // When a position override is active (widget promoted to SubgraphNode),
    // the interior widget's `active` flag is false (its node is in the
    // subgraph, not the current graph) — bypass that check.
    if (
      !widget.isVisible() ||
      (!widgetState.active && !useOverride && !useTransitionGrace)
    ) {
      widgetState.visible = false
      continue
    }

    // During graph transitions, hold the previous position for one frame
    // so promoted widgets don't briefly disappear before activation flips.
    if (useTransitionGrace) continue

    const posNode = useOverride ? override.node : widget.node
    const posWidget = useOverride ? override.widget : widget

    const isInCorrectGraph = posNode.graph === currentGraph
    const nodeVisible = lgCanvas.isNodeVisible(posNode)

    widgetState.visible =
      isInCorrectGraph &&
      nodeVisible &&
      !(widget.options.hideOnZoom && lowQuality)

    if (widgetState.visible) {
      const margin = widget.margin
      widgetState.pos = [
        posNode.pos[0] + margin,
        posNode.pos[1] + margin + posWidget.y
      ]
      widgetState.size = [
        (posWidget.width ?? posNode.width) - margin * 2,
        (posWidget.computedHeight ?? 50) - margin * 2
      ]
      widgetState.zIndex = getDomWidgetZIndex(posNode, currentGraph)
      widgetState.readonly = lgCanvas.read_only
    }
  }

  for (const widgetId of overrideTransitionGrace) {
    if (!seenWidgetIds.has(widgetId)) {
      overrideTransitionGrace.delete(widgetId)
    }
  }
}

const canvasStore = useCanvasStore()
whenever(
  () => canvasStore.canvas,
  (canvas) =>
    (canvas.onDrawForeground = useChainCallback(
      canvas.onDrawForeground,
      updateWidgets
    )),
  { immediate: true }
)
</script>
