import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import type { InputSpec as InputSpecV2 } from '@/schemas/nodeDef/nodeDefSchemaV2'
import { getInputSpecType } from '@/schemas/nodeDefSchema'
import type { InputSpec as InputSpecV1 } from '@/schemas/nodeDefSchema'
import type { ComfyWidgetConstructor } from '@/scripts/widgets'
import { ComfyWidgets } from '@/scripts/widgets'

export const useWidgetStore = defineStore('widget', () => {
  const coreWidgets = ComfyWidgets
  const customWidgets = ref<Map<string, ComfyWidgetConstructor>>(new Map())
  const widgets = computed<Map<string, ComfyWidgetConstructor>>(
    () => new Map([...customWidgets.value, ...Object.entries(coreWidgets)])
  )

  function inputIsWidget(spec: InputSpecV2 | InputSpecV1) {
    const type = Array.isArray(spec) ? getInputSpecType(spec) : spec.type
    return widgets.value.has(type)
  }

  function registerCustomWidgets(
    newWidgets: Record<string, ComfyWidgetConstructor>
  ) {
    for (const [type, widget] of Object.entries(newWidgets)) {
      customWidgets.value.set(type, widget)
    }
  }

  return {
    widgets,
    inputIsWidget,
    registerCustomWidgets
  }
})
