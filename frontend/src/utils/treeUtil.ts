import type { TreeNode } from '@/types/treeExplorerTypes'

export function buildTree<T>(items: T[], key: (item: T) => string[]): TreeNode {
  const root: TreeNode = {
    key: 'root',
    label: 'root',
    children: []
  }

  const map: Record<string, TreeNode> = {
    root: root
  }

  for (const item of items) {
    const keys = key(item)
    let parent = root
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i]
      // 'a/b/c/' represents an empty folder 'c' in folder 'b' in folder 'a'
      // 'a/b/c/' is split into ['a', 'b', 'c', '']
      if (k === '' && i === keys.length - 1) break

      const id = parent.key + '/' + k
      if (!map[id]) {
        const node: TreeNode = {
          key: id,
          label: k,
          leaf: false,
          children: []
        }
        map[id] = node
        parent.children?.push(node)
      }
      parent = map[id]
    }
    parent.leaf = keys[keys.length - 1] !== ''
    parent.data = item
  }
  return root
}

export function flattenTree<T>(tree: TreeNode): T[] {
  const result: T[] = []
  const stack: TreeNode[] = [tree]
  while (stack.length) {
    const node = stack.pop()!
    if (node.leaf && node.data) result.push(node.data)
    stack.push(...(node.children ?? []))
  }
  return result
}

/**
 * Sort the children of the node recursively.
 * @param node - The node to sort.
 * @param options - The options for sorting.
 * @param options.groupLeaf - Whether to group leaf nodes together.
 * @returns The sorted node.
 */
export function sortedTree(
  node: TreeNode,
  {
    groupLeaf = false
  }: {
    groupLeaf?: boolean
  } = {}
): TreeNode {
  const newNode: TreeNode = {
    ...node
  }

  if (node.children) {
    if (groupLeaf) {
      // Split children into folders and files
      const folders = node.children.filter((child) => !child.leaf)
      const files = node.children.filter((child) => child.leaf)

      // Sort folders and files separately by label
      const sortedFolders = folders.sort((a, b) =>
        (a.label ?? '').localeCompare(b.label ?? '')
      )
      const sortedFiles = files.sort((a, b) =>
        (a.label ?? '').localeCompare(b.label ?? '')
      )

      // Recursively sort folder children
      newNode.children = [
        ...sortedFolders.map((folder) =>
          sortedTree(folder, { groupLeaf: true })
        ),
        ...sortedFiles
      ]
    } else {
      const sortedChildren = [...node.children].sort((a, b) =>
        (a.label ?? '').localeCompare(b.label ?? '')
      )
      newNode.children = [
        ...sortedChildren.map((child) =>
          sortedTree(child, { groupLeaf: false })
        )
      ]
    }
  }

  return newNode
}

export const findNodeByKey = <T extends TreeNode>(
  root: T,
  key: string
): T | null => {
  if (root.key === key) {
    return root
  }
  if (!root.children) {
    return null
  }
  for (const child of root.children) {
    const result = findNodeByKey(child, key)
    if (result) {
      return result
    }
  }
  return null
}

/**
 * Deep clone a tree node and its children.
 * @param node - The node to clone.
 * @returns A deep clone of the node.
 */
function cloneTree<T extends TreeNode>(node: T): T {
  const clone = { ...node }

  // Clone children recursively
  if (node.children && node.children.length > 0) {
    clone.children = node.children.map((child) => cloneTree(child))
  }

  return clone
}

/**
 * Merge a subtree into the tree.
 * @param root - The root of the tree.
 * @param subtree - The subtree to merge.
 * @returns A new tree with the subtree merged.
 */
export const combineTrees = <T extends TreeNode>(root: T, subtree: T): T => {
  const newRoot = cloneTree(root)

  const parentKey = subtree.key.slice(0, subtree.key.lastIndexOf('/'))
  const parent = findNodeByKey(newRoot, parentKey)

  if (parent) {
    parent.children ??= []
    parent.children.push(cloneTree(subtree))
  }

  return newRoot
}
