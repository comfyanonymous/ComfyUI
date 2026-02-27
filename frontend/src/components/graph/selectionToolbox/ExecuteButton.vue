<template>
  <Button
    v-tooltip.top="{
      value: t('selectionToolbox.executeButton.tooltip'),
      showDelay: 1000
    }"
    variant="primary"
    :aria-label="t('selectionToolbox.executeButton.tooltip')"
    @mouseenter="() => handleMouseEnter()"
    @mouseleave="() => handleMouseLeave()"
    @click="handleClick"
  >
    <i class="icon-[lucide--play]" />
  </Button>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useSelectionState } from '@/composables/graph/useSelectionState'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useCommandStore } from '@/stores/commandStore'
import { isLGraphNode } from '@/utils/litegraphUtil'
import { isOutputNode } from '@/utils/nodeFilterUtil'

const { t } = useI18n()
const commandStore = useCommandStore()
const canvasStore = useCanvasStore()
const { selectedNodes } = useSelectionState()

const canvas = canvasStore.getCanvas()
const buttonHovered = ref(false)
const selectedOutputNodes = computed(() =>
  selectedNodes.value.filter(isLGraphNode).filter(isOutputNode)
)

function outputNodeStokeStyle(this: LGraphNode) {
  if (
    this.selected &&
    this.constructor.nodeData?.output_node &&
    buttonHovered.value
  ) {
    return { color: 'orange', lineWidth: 2, padding: 10 }
  }
}

const handleMouseEnter = () => {
  buttonHovered.value = true
  for (const node of selectedOutputNodes.value) {
    node.strokeStyles['outputNode'] = outputNodeStokeStyle
  }
  canvas.setDirty(true)
}

const handleMouseLeave = () => {
  buttonHovered.value = false
  canvas.setDirty(true)
}

const handleClick = async () => {
  await commandStore.execute('Comfy.QueueSelectedOutputNodes')
}
</script>
