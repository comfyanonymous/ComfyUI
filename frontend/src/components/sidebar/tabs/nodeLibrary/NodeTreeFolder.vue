<template>
  <div
    ref="container"
    class="node-lib-node-container"
    data-testid="node-tree-folder"
    :data-folder-name="node.label"
  >
    <TreeExplorerTreeNode :node="node" @item-dropped="handleItemDrop" />
  </div>
</template>

<script setup lang="ts">
import { computed, inject, onMounted, onUnmounted, ref, watch } from 'vue'

import TreeExplorerTreeNode from '@/components/common/TreeExplorerTreeNode.vue'
import type { BookmarkCustomization } from '@/schemas/apiSchema'
import { useNodeBookmarkStore } from '@/stores/nodeBookmarkStore'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { InjectKeyExpandedKeys } from '@/types/treeExplorerTypes'
import type { RenderedTreeExplorerNode } from '@/types/treeExplorerTypes'

const { node } = defineProps<{
  node: RenderedTreeExplorerNode<ComfyNodeDefImpl>
}>()

const nodeBookmarkStore = useNodeBookmarkStore()
const customization = computed<BookmarkCustomization | undefined>(() => {
  const nodeDef = node.data
  return nodeDef
    ? nodeBookmarkStore.bookmarksCustomization[nodeDef.nodePath]
    : undefined
})

const treeNodeElement = ref<HTMLElement | null>(null)
const iconElement = ref<HTMLElement | null>(null)

let stopWatchCustomization: (() => void) | null = null

const container = ref<HTMLElement | null>(null)
onMounted(() => {
  treeNodeElement.value = container.value?.closest(
    '.p-tree-node-content'
  ) as HTMLElement
  iconElement.value = treeNodeElement.value.querySelector(
    ':scope > .p-tree-node-icon'
  ) as HTMLElement
  updateIconColor()

  // Start watching after the component is mounted
  stopWatchCustomization = watch(customization, updateIconColor, { deep: true })
})

const updateIconColor = () => {
  if (iconElement.value && customization.value) {
    iconElement.value.style.color = customization.value.color ?? ''
  }
}

onUnmounted(() => {
  if (stopWatchCustomization) {
    stopWatchCustomization()
  }
})

const expandedKeys = inject(InjectKeyExpandedKeys)
const handleItemDrop = (node: RenderedTreeExplorerNode<ComfyNodeDefImpl>) => {
  if (!expandedKeys) return
  expandedKeys.value[node.key] = true
}
</script>
