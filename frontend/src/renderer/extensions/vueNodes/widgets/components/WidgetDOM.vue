<script setup lang="ts">
import { whenever } from '@vueuse/core'
import { onMounted, ref } from 'vue'

import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import type { DOMWidget } from '@/scripts/domWidget'
import { isDOMWidget } from '@/scripts/domWidget'
import { resolveWidgetFromHostNode } from '@/renderer/extensions/vueNodes/widgets/utils/resolvePromotedWidget'
import type { SimplifiedWidget } from '@/types/simplifiedWidget'

// Button widgets don't have a v-model value, they trigger actions
const props = defineProps<{
  widget: SimplifiedWidget<void>
  nodeId: string
}>()

const domEl = ref<HTMLElement>()

const canvasStore = useCanvasStore()
const { canvas } = canvasStore

function findDOMWidget(): DOMWidget<HTMLElement, object | string> | undefined {
  const hostNode = canvas?.graph?.getNodeById(props.nodeId) ?? undefined
  const resolved = resolveWidgetFromHostNode(hostNode, props.widget.name)
  if (resolved && isDOMWidget(resolved.widget)) return resolved.widget

  return undefined
}

function mountWidgetElement() {
  if (!domEl.value) return
  const widget = findDOMWidget()
  if (!widget) return
  if (domEl.value.contains(widget.element)) return
  domEl.value.replaceChildren(widget.element)
}

onMounted(() => {
  mountWidgetElement()
})

whenever(() => !canvasStore.linearMode, mountWidgetElement)
</script>
<template>
  <div
    ref="domEl"
    class="flex flex-col [&>*]:flex-1"
    @pointerdown.stop
    @pointermove.stop
    @pointerup.stop
  />
</template>
