<template>
  <div
    v-if="tooltipText"
    ref="tooltipRef"
    class="node-tooltip"
    :style="{ left, top }"
  >
    {{ tooltipText }}
  </div>
</template>

<script setup lang="ts">
import { useEventListener } from '@vueuse/core'
import { nextTick, ref } from 'vue'

import { st } from '@/i18n'
import {
  LiteGraph,
  isOverNodeInput,
  isOverNodeOutput
} from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'
import { app as comfyApp } from '@/scripts/app'
import { isDOMWidget } from '@/scripts/domWidget'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { normalizeI18nKey } from '@/utils/formatUtil'

let idleTimeout: number
const nodeDefStore = useNodeDefStore()
const settingStore = useSettingStore()
const tooltipRef = ref<HTMLDivElement | undefined>()
const tooltipText = ref('')
const left = ref<string>()
const top = ref<string>()

function hideTooltip() {
  return (tooltipText.value = '')
}

async function showTooltip(tooltip: string | null | undefined) {
  if (!tooltip) return

  left.value = comfyApp.canvas.mouse[0] + 'px'
  top.value = comfyApp.canvas.mouse[1] + 'px'
  tooltipText.value = tooltip

  await nextTick()

  const rect = tooltipRef.value?.getBoundingClientRect()
  if (!rect) return

  if (rect.right > window.innerWidth) {
    left.value = comfyApp.canvas.mouse[0] - rect.width + 'px'
  }

  if (rect.top < 0) {
    top.value = comfyApp.canvas.mouse[1] + rect.height + 'px'
  }
}

function onIdle() {
  const { canvas } = comfyApp
  const node = canvas?.node_over
  if (!node || node.flags?.ghost) return

  const ctor = node.constructor as { title_mode?: 0 | 1 | 2 | 3 }
  const nodeDef = nodeDefStore.nodeDefsByName[node.type ?? '']

  if (
    ctor.title_mode !== LiteGraph.NO_TITLE &&
    canvas.graph_mouse[1] < node.pos[1] // If we are over a node, but not within the node then we are on its title
  ) {
    return showTooltip(nodeDef?.description)
  }

  if (node.flags?.collapsed) return

  const inputSlot = isOverNodeInput(
    node,
    canvas.graph_mouse[0],
    canvas.graph_mouse[1],
    [0, 0]
  )
  if (inputSlot !== -1) {
    const inputName = node.inputs[inputSlot].name
    const translatedTooltip = st(
      `nodeDefs.${normalizeI18nKey(node.type ?? '')}.inputs.${normalizeI18nKey(inputName)}.tooltip`,
      nodeDef?.inputs[inputName]?.tooltip ?? ''
    )
    return showTooltip(translatedTooltip)
  }

  const outputSlot = isOverNodeOutput(
    node,
    canvas.graph_mouse[0],
    canvas.graph_mouse[1],
    [0, 0]
  )
  if (outputSlot !== -1) {
    const translatedTooltip = st(
      `nodeDefs.${normalizeI18nKey(node.type ?? '')}.outputs.${outputSlot}.tooltip`,
      nodeDef?.outputs[outputSlot]?.tooltip ?? ''
    )
    return showTooltip(translatedTooltip)
  }

  const widget = comfyApp.canvas.getWidgetAtCursor()
  // Dont show for DOM widgets, these use native browser tooltips as we dont get proper mouse events on these
  if (widget && !isDOMWidget(widget)) {
    const translatedTooltip = st(
      `nodeDefs.${normalizeI18nKey(node.type ?? '')}.inputs.${normalizeI18nKey(widget.name)}.tooltip`,
      nodeDef?.inputs[widget.name]?.tooltip ?? ''
    )
    // Widget tooltip can be set dynamically, current translation collection does not support this.
    return showTooltip(widget.tooltip ?? translatedTooltip)
  }
}

const onMouseMove = (e: MouseEvent) => {
  hideTooltip()
  clearTimeout(idleTimeout)

  if ((e.target as Node).nodeName !== 'CANVAS') return
  idleTimeout = window.setTimeout(
    onIdle,
    settingStore.get('LiteGraph.Node.TooltipDelay')
  )
}

useEventListener(window, 'mousemove', onMouseMove)
useEventListener(window, 'click', hideTooltip)
</script>

<style lang="css" scoped>
.node-tooltip {
  pointer-events: none;
  background: var(--comfy-input-bg);
  border-radius: 5px;
  box-shadow: 0 0 5px rgb(0 0 0 / 0.4);
  color: var(--input-text);
  left: 0;
  max-width: 30vw;
  padding: 4px 8px;
  position: absolute;
  top: 0;
  transform: translate(5px, calc(-100% - 5px));
  white-space: pre-wrap;
  z-index: 99999;
}
</style>
