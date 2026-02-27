<template>
  <div
    ref="container"
    :class="[
      'tree-node',
      {
        'can-drop': canDrop,
        'tree-folder': !props.node.leaf,
        'tree-leaf': props.node.leaf
      }
    ]"
    :data-testid="`tree-node-${node.key}`"
  >
    <div class="node-content">
      <span class="node-label">
        <slot name="before-label" :node="props.node" />
        <EditableText
          :model-value="node.label"
          :is-editing="isEditing"
          @edit="handleRename"
        />
        <slot name="after-label" :node="props.node" />
      </span>
      <Badge
        v-if="showNodeBadgeText"
        :value="nodeBadgeText"
        severity="secondary"
        class="leaf-count-badge"
      />
    </div>
    <div
      class="node-actions flex gap-1 touch:opacity-100 motion-safe:opacity-0 motion-safe:group-hover/tree-node:opacity-100"
    >
      <slot name="actions" :node="props.node" />
    </div>
  </div>
</template>

<script setup lang="ts" generic="T">
import { setCustomNativeDragPreview } from '@atlaskit/pragmatic-drag-and-drop/element/set-custom-native-drag-preview'
import Badge from 'primevue/badge'
import { computed, inject, ref } from 'vue'

import EditableText from '@/components/common/EditableText.vue'
import {
  usePragmaticDraggable,
  usePragmaticDroppable
} from '@/composables/usePragmaticDragAndDrop'
import { InjectKeyHandleEditLabelFunction } from '@/types/treeExplorerTypes'
import type {
  RenderedTreeExplorerNode,
  TreeExplorerDragAndDropData
} from '@/types/treeExplorerTypes'

const props = defineProps<{
  node: RenderedTreeExplorerNode<T>
}>()

const emit = defineEmits<{
  (
    e: 'itemDropped',
    node: RenderedTreeExplorerNode<T>,
    data: RenderedTreeExplorerNode<T>
  ): void
  (e: 'dragStart', node: RenderedTreeExplorerNode<T>): void
  (e: 'dragEnd', node: RenderedTreeExplorerNode<T>): void
}>()

const nodeBadgeText = computed<string>(() => {
  if (props.node.leaf) {
    return ''
  }
  if (props.node.badgeText !== undefined && props.node.badgeText !== null) {
    return props.node.badgeText
  }
  return props.node.totalLeaves.toString()
})
const showNodeBadgeText = computed<boolean>(() => nodeBadgeText.value !== '')

const isEditing = computed<boolean>(() => props.node.isEditingLabel ?? false)
const handleEditLabel = inject(InjectKeyHandleEditLabelFunction)
const handleRename = (newName: string) => {
  handleEditLabel?.(props.node as RenderedTreeExplorerNode, newName)
}

const container = ref<HTMLElement | null>(null)
const canDrop = ref(false)

const treeNodeElementGetter = () =>
  container.value?.closest('.p-tree-node-content') as HTMLElement

if (props.node.draggable) {
  usePragmaticDraggable(treeNodeElementGetter, {
    getInitialData: () => {
      return {
        type: 'tree-explorer-node',
        data: props.node
      }
    },
    onDragStart: () => emit('dragStart', props.node),
    onDrop: () => emit('dragEnd', props.node),
    onGenerateDragPreview: props.node.renderDragPreview
      ? ({ nativeSetDragImage }) => {
          setCustomNativeDragPreview({
            render: ({ container }) => {
              return props.node.renderDragPreview?.(container)
            },
            nativeSetDragImage
          })
        }
      : undefined
  })
}

if (props.node.droppable) {
  usePragmaticDroppable(treeNodeElementGetter, {
    onDrop: async (event) => {
      const dndData = event.source.data as TreeExplorerDragAndDropData
      if (dndData.type === 'tree-explorer-node') {
        await props.node.handleDrop?.(dndData as TreeExplorerDragAndDropData<T>)
        canDrop.value = false
        emit(
          'itemDropped',
          props.node,
          dndData.data as RenderedTreeExplorerNode<T>
        )
      }
    },
    onDragEnter: (event) => {
      const dndData = event.source.data as TreeExplorerDragAndDropData
      if (dndData.type === 'tree-explorer-node') {
        canDrop.value = true
      }
    },
    onDragLeave: () => {
      canDrop.value = false
    }
  })
}
</script>

<style scoped>
.tree-node {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.leaf-count-badge {
  margin-left: 0.5rem;
}
.node-content {
  display: flex;
  align-items: center;
  flex-grow: 1;
}
.leaf-label {
  margin-left: 0.5rem;
}
:deep(.editable-text span) {
  word-break: break-all;
}
</style>
