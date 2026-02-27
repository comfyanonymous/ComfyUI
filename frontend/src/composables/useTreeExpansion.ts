import type { Ref } from 'vue'

import type { TreeNode } from '@/types/treeExplorerTypes'

export function useTreeExpansion(expandedKeys: Ref<Record<string, boolean>>) {
  const toggleNode = (node: TreeNode) => {
    if (node.key && typeof node.key === 'string') {
      if (node.key in expandedKeys.value) {
        delete expandedKeys.value[node.key]
      } else {
        expandedKeys.value[node.key] = true
      }
    }
  }

  const toggleNodeRecursive = (node: TreeNode) => {
    if (node.key && typeof node.key === 'string') {
      if (node.key in expandedKeys.value) {
        collapseNode(node)
      } else {
        expandNode(node)
      }
    }
  }

  const expandNode = (node: TreeNode) => {
    if (node.key && typeof node.key === 'string' && !node.leaf) {
      expandedKeys.value[node.key] = true

      for (const child of node.children ?? []) {
        expandNode(child)
      }
    }
  }

  const collapseNode = (node: TreeNode) => {
    if (node.key && typeof node.key === 'string' && !node.leaf) {
      delete expandedKeys.value[node.key]

      for (const child of node.children ?? []) {
        collapseNode(child)
      }
    }
  }

  const toggleNodeOnEvent = (e: MouseEvent | KeyboardEvent, node: TreeNode) => {
    if (e.ctrlKey) {
      toggleNodeRecursive(node)
    } else {
      toggleNode(node)
    }
  }

  return {
    toggleNode,
    toggleNodeRecursive,
    expandNode,
    collapseNode,
    toggleNodeOnEvent
  }
}
