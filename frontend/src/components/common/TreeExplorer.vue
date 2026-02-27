<template>
  <Tree
    v-bind="$attrs"
    v-model:expanded-keys="expandedKeys"
    v-model:selection-keys="selectionKeys"
    class="tree-explorer px-2 py-0 2xl:px-4 bg-transparent"
    :class="props.class"
    :value="renderedRoot.children"
    selection-mode="single"
    :pt="{
      nodeLabel: 'tree-explorer-node-label',
      nodeContent: ({ context }) => ({
        class: 'group/tree-node',
        onClick: (e: MouseEvent) =>
          onNodeContentClick(e, context.node as RenderedTreeExplorerNode<T>),
        onContextmenu: (e: MouseEvent) =>
          handleContextMenu(e, context.node as RenderedTreeExplorerNode<T>)
      }),
      nodeToggleButton: () => ({
        onClick: (e: MouseEvent) => {
          e.stopImmediatePropagation()
        }
      })
    }"
  >
    <template #folder="{ node }">
      <slot name="folder" :node="node">
        <TreeExplorerTreeNode :node="node" />
      </slot>
    </template>
    <template #node="{ node }">
      <slot name="node" :node="node">
        <TreeExplorerTreeNode :node="node" />
      </slot>
    </template>
  </Tree>
  <ContextMenu ref="menu" :model="menuItems" />
</template>
<script setup lang="ts" generic="T">
import ContextMenu from 'primevue/contextmenu'
import type { MenuItem, MenuItemCommandEvent } from 'primevue/menuitem'
import Tree from 'primevue/tree'
import { computed, provide, ref, shallowRef } from 'vue'
import { useI18n } from 'vue-i18n'

import TreeExplorerTreeNode from '@/components/common/TreeExplorerTreeNode.vue'
import { useTreeFolderOperations } from '@/composables/tree/useTreeFolderOperations'
import { useErrorHandling } from '@/composables/useErrorHandling'
import {
  InjectKeyExpandedKeys,
  InjectKeyHandleEditLabelFunction
} from '@/types/treeExplorerTypes'
import type {
  RenderedTreeExplorerNode,
  TreeExplorerNode
} from '@/types/treeExplorerTypes'
import { combineTrees, findNodeByKey } from '@/utils/treeUtil'

defineOptions({
  inheritAttrs: false
})

const expandedKeys = defineModel<Record<string, boolean>>('expandedKeys', {
  required: true
})
provide(InjectKeyExpandedKeys, expandedKeys)
const selectionKeys = defineModel<Record<string, boolean>>('selectionKeys')
// Tracks whether the caller has set the selectionKeys model.
const storeSelectionKeys = selectionKeys.value !== undefined

const props = defineProps<{
  root: TreeExplorerNode<T>
  class?: string
}>()
const emit = defineEmits<{
  (e: 'nodeClick', node: RenderedTreeExplorerNode<T>, event: MouseEvent): void
  (e: 'nodeDelete', node: RenderedTreeExplorerNode<T>): void
  (e: 'contextMenu', node: RenderedTreeExplorerNode<T>, event: MouseEvent): void
}>()

const {
  newFolderNode,
  getAddFolderMenuItem,
  handleFolderCreation,
  addFolderCommand
} = useTreeFolderOperations<T>(
  /* expandNode */ (node: TreeExplorerNode<T>) => {
    expandedKeys.value[node.key] = true
  }
)

const renderedRoot = computed<RenderedTreeExplorerNode<T>>(() => {
  const renderedRoot = fillNodeInfo(props.root)
  return newFolderNode.value
    ? combineTrees(renderedRoot, newFolderNode.value)
    : renderedRoot
})
const getTreeNodeIcon = (node: TreeExplorerNode<T>) => {
  if (node.getIcon) {
    const icon = node.getIcon()
    if (icon) {
      return icon
    }
  } else if (node.icon) {
    return node.icon
  }
  // node.icon is undefined
  if (node.leaf) {
    return 'pi pi-file'
  }
  const isExpanded = expandedKeys.value?.[node.key] ?? false
  return isExpanded ? 'pi pi-folder-open' : 'pi pi-folder'
}
const fillNodeInfo = (
  node: TreeExplorerNode<T>
): RenderedTreeExplorerNode<T> => {
  const children = node.children?.map(fillNodeInfo) ?? []
  const totalLeaves = node.leaf
    ? 1
    : children.reduce((acc, child) => acc + child.totalLeaves, 0)
  return {
    ...node,
    icon: getTreeNodeIcon(node),
    children,
    type: node.leaf ? 'node' : 'folder',
    totalLeaves,
    badgeText: node.getBadgeText ? node.getBadgeText() : undefined,
    isEditingLabel: node.key === renameEditingNode.value?.key
  }
}
const onNodeContentClick = async (
  e: MouseEvent,
  node: RenderedTreeExplorerNode<T>
) => {
  if (!storeSelectionKeys) {
    selectionKeys.value = {}
  }
  if (node.handleClick) {
    await node.handleClick(e)
  }
  emit('nodeClick', node, e)
}
const menu = ref<InstanceType<typeof ContextMenu> | null>(null)
const menuTargetNode = shallowRef<RenderedTreeExplorerNode<T> | null>(null)
const extraMenuItems = computed(() => {
  const node = menuTargetNode.value
  return node?.contextMenuItems
    ? typeof node.contextMenuItems === 'function'
      ? node.contextMenuItems(node)
      : node.contextMenuItems
    : []
})
const renameEditingNode = shallowRef<RenderedTreeExplorerNode<T> | null>(null)
const errorHandling = useErrorHandling()
const handleNodeLabelEdit = async (
  n: RenderedTreeExplorerNode,
  newName: string
) => {
  const node = n as RenderedTreeExplorerNode<T>
  await errorHandling.wrapWithErrorHandlingAsync(
    async () => {
      if (node.key === newFolderNode.value?.key) {
        await handleFolderCreation(newName)
      } else {
        await node.handleRename?.(newName)
      }
    },
    node.handleError,
    () => {
      renameEditingNode.value = null
    }
  )()
}
provide(InjectKeyHandleEditLabelFunction, handleNodeLabelEdit)

const { t } = useI18n()
const renameCommand = (node: RenderedTreeExplorerNode<T>) => {
  renameEditingNode.value = node
}
const deleteCommand = async (node: RenderedTreeExplorerNode<T>) => {
  await node.handleDelete?.()
  emit('nodeDelete', node)
}
const menuItems = computed<MenuItem[]>(() => {
  const node = menuTargetNode.value
  return [
    getAddFolderMenuItem(node),
    {
      label: t('g.rename'),
      icon: 'pi pi-file-edit',
      command: () => {
        if (node) {
          renameCommand(node)
        }
      },
      visible: node?.handleRename !== undefined
    },
    {
      label: t('g.delete'),
      icon: 'pi pi-trash',
      command: async () => {
        if (node) {
          await deleteCommand(node)
        }
      },
      visible: node?.handleDelete !== undefined,
      isAsync: true // The delete command can be async
    },
    ...extraMenuItems.value
  ].map((menuItem: MenuItem) => ({
    ...menuItem,
    command: menuItem.command
      ? wrapCommandWithErrorHandler(menuItem.command, {
          isAsync: menuItem.isAsync ?? false
        })
      : undefined
  }))
})

const handleContextMenu = (
  e: MouseEvent,
  node: RenderedTreeExplorerNode<T>
) => {
  menuTargetNode.value = node
  emit('contextMenu', node, e)
  if (menuItems.value.filter((item) => item.visible).length > 0) {
    menu.value?.show(e)
  }
}

const wrapCommandWithErrorHandler = (
  command: (event: MenuItemCommandEvent) => void,
  { isAsync = false }: { isAsync: boolean }
) => {
  const node = menuTargetNode.value
  return isAsync
    ? errorHandling.wrapWithErrorHandlingAsync(
        command as (event: MenuItemCommandEvent) => Promise<void>,
        node?.handleError
      )
    : errorHandling.wrapWithErrorHandling(command, node?.handleError)
}

defineExpose({
  /**
   * The command to add a folder to a node via the context menu
   * @param targetNodeKey - The key of the node where the folder will be added under
   */
  addFolderCommand: (targetNodeKey: string) => {
    const targetNode = findNodeByKey(renderedRoot.value, targetNodeKey)
    if (targetNode) {
      addFolderCommand(targetNode)
    }
  }
})
</script>

<style scoped>
:deep(.tree-explorer-node-label) {
  width: 100%;
  display: flex;
  align-items: center;
  margin-left: var(--p-tree-node-gap);
  flex-grow: 1;
}

/*
 * The following styles are necessary to avoid layout shift when dragging nodes over folders.
 * By setting the position to relative on the parent and using an absolutely positioned pseudo-element,
 * we can create a visual indicator for the drop target without affecting the layout of other elements.
 */
:deep(.p-tree-node-content:has(.tree-folder)) {
  position: relative;
}

:deep(.p-tree-node-content:has(.tree-folder.can-drop))::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  border: 1px solid var(--p-content-color);
  pointer-events: none;
}
</style>
