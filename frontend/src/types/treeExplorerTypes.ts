import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import type { MenuItem } from 'primevue/menuitem'
import type { TreeNode as PrimeVueTreeNode } from 'primevue/treenode'
import type { InjectionKey, ModelRef, Ref } from 'vue'

export interface TreeNode extends PrimeVueTreeNode {
  label: string
  children?: this[]
}

export interface NodeLibrarySection {
  title?: string
  root: RenderedTreeExplorerNode<ComfyNodeDefImpl>
}

export interface TreeExplorerNode<T = unknown> extends TreeNode {
  data?: T
  children?: this[]
  icon?: string
  /**
   * Function to override what icon to use for the node.
   * Return undefined to fallback to {@link icon} property.
   */
  getIcon?: (this: TreeExplorerNode<T>) => string | undefined
  /**
   * Function to override what text to use for the leaf-count badge on a folder node.
   * Return undefined to fallback to default badge text, which is the subtree's leaf count.
   * Return empty string to hide the badge.
   */
  getBadgeText?: (this: TreeExplorerNode<T>) => string | undefined
  /** Function to handle renaming the node */
  handleRename?: (
    this: TreeExplorerNode<T>,
    newName: string
  ) => void | Promise<void>
  /** Function to handle deleting the node */
  handleDelete?: (this: TreeExplorerNode<T>) => void | Promise<void>
  /** Function to handle adding a folder */
  handleAddFolder?: (
    this: TreeExplorerNode<T>,
    folderName: string
  ) => void | Promise<void>
  /** Whether the node is draggable */
  draggable?: boolean
  /** Function to render a drag preview */
  renderDragPreview?: (
    this: TreeExplorerNode<T>,
    container: HTMLElement
  ) => void | (() => void)
  /** Whether the node is droppable */
  droppable?: boolean
  /** Function to handle dropping a node */
  handleDrop?: (
    this: TreeExplorerNode<T>,
    data: TreeExplorerDragAndDropData<T>
  ) => void | Promise<void>
  /** Function to handle clicking a node */
  handleClick?: (
    this: TreeExplorerNode<T>,
    event: MouseEvent
  ) => void | Promise<void>
  /** Function to handle errors */
  handleError?: (this: TreeExplorerNode<T>, error: unknown) => void
  /** Extra context menu items */
  contextMenuItems?:
    | MenuItem[]
    | ((targetNode: RenderedTreeExplorerNode<T>) => MenuItem[])
}

export interface RenderedTreeExplorerNode<
  T = unknown
> extends TreeExplorerNode<T> {
  children?: this[]
  icon: string
  type: 'folder' | 'node'
  /** Total number of leaves in the subtree */
  totalLeaves: number
  /** Text to display on the leaf-count badge. Empty string means no badge. */
  badgeText?: string
  /** Whether the node label is currently being edited */
  isEditingLabel?: boolean
}

export type TreeExplorerDragAndDropData<T = unknown> = {
  type: 'tree-explorer-node'
  data: RenderedTreeExplorerNode<T>
}

export const InjectKeyHandleEditLabelFunction: InjectionKey<
  (node: RenderedTreeExplorerNode, newName: string) => void
> = Symbol()

export const InjectKeyExpandedKeys: InjectionKey<
  ModelRef<Record<string, boolean>>
> = Symbol()

export const InjectKeyContextMenuNode: InjectionKey<
  Ref<RenderedTreeExplorerNode<ComfyNodeDefImpl> | null>
> = Symbol()
