<template>
  <div
    class="group relative flex flex-col items-center justify-center py-4 px-2 rounded-2xl cursor-pointer select-none transition-colors duration-150 box-content bg-component-node-background hover:bg-secondary-background-hover border border-component-node-border aspect-square"
    :data-node-name="node.label"
    draggable="true"
    @click="handleClick"
    @dragstart="handleDragStart"
    @dragend="handleDragEnd"
    @mouseenter="handleMouseEnter"
    @mouseleave="handleMouseLeave"
  >
    <div class="flex flex-1 items-center justify-center">
      <i :class="cn(nodeIcon, 'size-14 text-muted-foreground')" />
    </div>

    <TextTickerMultiLine
      class="shrink-0 h-8 w-full text-xs font-bold text-foreground leading-4"
    >
      {{ node.label }}
    </TextTickerMultiLine>
  </div>

  <Teleport v-if="showPreview" to="body">
    <div
      :ref="(el) => (previewRef = el as HTMLElement)"
      :style="nodePreviewStyle"
    >
      <NodePreviewCard
        :node-def="node.data!"
        :show-inputs-and-outputs="false"
      />
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { kebabCase } from 'es-toolkit/string'
import { computed, inject } from 'vue'

import TextTickerMultiLine from '@/components/common/TextTickerMultiLine.vue'
import NodePreviewCard from '@/components/node/NodePreviewCard.vue'
import { SidebarContainerKey } from '@/components/sidebar/tabs/SidebarTabTemplate.vue'
import { useNodePreviewAndDrag } from '@/composables/node/useNodePreviewAndDrag'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import type { RenderedTreeExplorerNode } from '@/types/treeExplorerTypes'
import { cn } from '@/utils/tailwindUtil'

const { node } = defineProps<{
  node: RenderedTreeExplorerNode<ComfyNodeDefImpl>
}>()

const emit = defineEmits<{
  click: [node: RenderedTreeExplorerNode<ComfyNodeDefImpl>]
}>()

const panelRef = inject(SidebarContainerKey, undefined)
const nodeDef = computed(() => node.data)

const {
  previewRef,
  showPreview,
  nodePreviewStyle,
  handleMouseEnter,
  handleMouseLeave,
  handleDragStart,
  handleDragEnd
} = useNodePreviewAndDrag(nodeDef, { panelRef })

const nodeIcon = computed(() => {
  const nodeName = node.data?.name
  const iconName = nodeName ? kebabCase(nodeName) : 'node'
  return `icon-[comfy--${iconName}]`
})

function handleClick() {
  if (!node.data) return
  emit('click', node)
}
</script>
