/**
 * Stores all DOM widgets that are used in the canvas.
 */
import { defineStore } from 'pinia'
import { computed, markRaw, ref } from 'vue'
import type { Raw } from 'vue'

import type { PositionConfig } from '@/composables/element/useAbsolutePosition'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import type { BaseDOMWidget } from '@/scripts/domWidget'

interface PositionOverride {
  node: Raw<LGraphNode>
  widget: Raw<IBaseWidget>
}

export interface DomWidgetState extends PositionConfig {
  // Raw widget instance
  widget: Raw<BaseDOMWidget<object | string>>
  visible: boolean
  readonly: boolean
  zIndex: number
  /** If the widget belongs to the current graph/subgraph. */
  active: boolean
  /** Override positioning to render on a different node (e.g. SubgraphNode). */
  positionOverride?: PositionOverride
}

export const useDomWidgetStore = defineStore('domWidget', () => {
  const widgetStates = ref<Map<string, DomWidgetState>>(new Map())

  const activeWidgetStates = computed(() =>
    [...widgetStates.value.values()].filter((state) => state.active)
  )
  const inactiveWidgetStates = computed(() =>
    [...widgetStates.value.values()].filter((state) => !state.active)
  )

  function registerWidget<V extends object | string>(widget: BaseDOMWidget<V>) {
    widgetStates.value.set(widget.id, {
      widget: markRaw(widget) as unknown as Raw<BaseDOMWidget<object | string>>,
      visible: true,
      readonly: false,
      zIndex: 0,
      pos: [0, 0],
      size: [0, 0],
      active: true
    })
  }

  function unregisterWidget(widgetId: string) {
    widgetStates.value.delete(widgetId)
  }

  function activateWidget(widgetId: string) {
    const state = widgetStates.value.get(widgetId)
    if (state) state.active = true
  }

  function deactivateWidget(widgetId: string) {
    const state = widgetStates.value.get(widgetId)
    if (state) state.active = false
  }

  function setWidget(widget: BaseDOMWidget) {
    const state = widgetStates.value.get(widget.id)
    if (!state) return
    state.active = true
    state.widget = widget
  }

  function setPositionOverride(widgetId: string, override: PositionOverride) {
    const state = widgetStates.value.get(widgetId)
    if (!state) return
    const current = state.positionOverride
    if (
      current &&
      current.node === override.node &&
      current.widget === override.widget
    )
      return
    state.positionOverride = {
      node: markRaw(override.node),
      widget: markRaw(override.widget)
    }
  }

  function clearPositionOverride(widgetId: string) {
    const state = widgetStates.value.get(widgetId)
    if (state) state.positionOverride = undefined
  }

  function clear() {
    widgetStates.value.clear()
  }

  return {
    widgetStates,
    activeWidgetStates,
    inactiveWidgetStates,
    registerWidget,
    unregisterWidget,
    activateWidget,
    deactivateWidget,
    setWidget,
    setPositionOverride,
    clearPositionOverride,
    clear
  }
})
