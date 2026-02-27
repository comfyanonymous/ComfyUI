import _ from 'es-toolkit/compat'
import { defineStore } from 'pinia'
import { computed } from 'vue'

import { useSettingStore } from '@/platform/settings/settingStore'
import type { BookmarkCustomization } from '@/schemas/apiSchema'
import type { TreeNode } from '@/types/treeExplorerTypes'

import {
  buildNodeDefTree,
  createDummyFolderNodeDef,
  useNodeDefStore
} from './nodeDefStore'
import type { ComfyNodeDefImpl } from './nodeDefStore'

const BOOKMARK_SETTING_ID = 'Comfy.NodeLibrary.Bookmarks.V2'

export const useNodeBookmarkStore = defineStore('nodeBookmark', () => {
  const settingStore = useSettingStore()
  const nodeDefStore = useNodeDefStore()
  const bookmarks = computed<string[]>(() =>
    settingStore.get(BOOKMARK_SETTING_ID)
  )

  const bookmarksSet = computed<Set<string>>(() => new Set(bookmarks.value))

  const bookmarkedRoot = computed<TreeNode>(() =>
    buildBookmarkTree(bookmarks.value)
  )

  // For a node in custom bookmark folders, check if its nodePath is in bookmarksSet
  // For a node in the nodeDefStore, check if its name is bookmarked at top level
  const isBookmarked = (node: ComfyNodeDefImpl) =>
    bookmarksSet.value.has(node.nodePath) || bookmarksSet.value.has(node.name)

  const toggleBookmark = async (node: ComfyNodeDefImpl) => {
    if (isBookmarked(node)) {
      await deleteBookmark(node.nodePath)
      // Delete the bookmark at the top level if it exists
      // This is used for clicking the bookmark button in the node library, i.e.
      // the node is inside original/standard node library tree node
      await deleteBookmark(node.name)
    } else {
      await addBookmark(node.name)
    }
  }

  const buildBookmarkTree = (bookmarks: string[]) => {
    const bookmarkNodes = bookmarks
      .map((bookmark: string) => {
        if (bookmark.endsWith('/')) return createDummyFolderNodeDef(bookmark)

        const parts = bookmark.split('/')
        const name = parts.pop() ?? ''
        const category = parts.join('/')
        const srcNodeDef = nodeDefStore.allNodeDefsByName[name]
        if (!srcNodeDef) {
          return null
        }
        const nodeDef = _.clone(srcNodeDef)
        nodeDef.category = category
        return nodeDef
      })
      .filter((nodeDef) => nodeDef !== null)
    return buildNodeDefTree(bookmarkNodes)
  }

  const addBookmark = async (nodePath: string) => {
    await settingStore.set(BOOKMARK_SETTING_ID, [...bookmarks.value, nodePath])
  }

  const deleteBookmark = async (nodePath: string) => {
    await settingStore.set(
      BOOKMARK_SETTING_ID,
      bookmarks.value.filter((b: string) => b !== nodePath)
    )
  }

  const addNewBookmarkFolder = async (
    parent: ComfyNodeDefImpl | undefined,
    folderName: string
  ) => {
    const parentPath = parent ? parent.nodePath : ''
    const newFolderPath = parentPath + folderName + '/'
    await addBookmark(newFolderPath)
    return newFolderPath
  }

  const renameBookmarkFolder = async (
    folderNode: ComfyNodeDefImpl,
    newName: string
  ) => {
    if (!folderNode.isDummyFolder) {
      throw new Error('Cannot rename non-folder node')
    }

    if (newName.includes('/')) {
      throw new Error('Folder name cannot contain "/"')
    }

    const newNodePath =
      folderNode.category.split('/').slice(0, -1).concat(newName).join('/') +
      '/'

    if (newNodePath === folderNode.nodePath) {
      return
    }

    if (bookmarks.value.some((b: string) => b.startsWith(newNodePath))) {
      throw new Error(`Folder name "${newNodePath}" already exists`)
    }

    await settingStore.set(
      BOOKMARK_SETTING_ID,
      bookmarks.value.map((b: string) =>
        b.startsWith(folderNode.nodePath)
          ? b.replace(folderNode.nodePath, newNodePath)
          : b
      )
    )
    await renameBookmarkCustomization(folderNode.nodePath, newNodePath)
  }

  const deleteBookmarkFolder = async (folderNode: ComfyNodeDefImpl) => {
    if (!folderNode.isDummyFolder) {
      throw new Error('Cannot delete non-folder node')
    }
    await settingStore.set(
      BOOKMARK_SETTING_ID,
      bookmarks.value.filter(
        (b: string) =>
          b !== folderNode.nodePath && !b.startsWith(folderNode.nodePath)
      )
    )
    await deleteBookmarkCustomization(folderNode.nodePath)
  }

  const bookmarksCustomization = computed<
    Record<string, BookmarkCustomization>
  >(() => settingStore.get('Comfy.NodeLibrary.BookmarksCustomization'))

  const updateBookmarkCustomization = async (
    nodePath: string,
    customization: BookmarkCustomization
  ) => {
    const currentCustomization = bookmarksCustomization.value[nodePath] || {}
    const newCustomization = { ...currentCustomization, ...customization }

    // Remove attributes that are set to default values
    if (newCustomization.icon === defaultBookmarkIcon) {
      delete newCustomization.icon
    }
    if (newCustomization.color === defaultBookmarkColor) {
      delete newCustomization.color
    }

    // If the customization is empty, remove it entirely
    if (Object.keys(newCustomization).length === 0) {
      await deleteBookmarkCustomization(nodePath)
    } else {
      await settingStore.set('Comfy.NodeLibrary.BookmarksCustomization', {
        ...bookmarksCustomization.value,
        [nodePath]: newCustomization
      })
    }
  }

  const deleteBookmarkCustomization = async (nodePath: string) => {
    await settingStore.set('Comfy.NodeLibrary.BookmarksCustomization', {
      ...bookmarksCustomization.value,
      [nodePath]: undefined
    } as Record<string, BookmarkCustomization>)
  }

  const renameBookmarkCustomization = async (
    oldNodePath: string,
    newNodePath: string
  ) => {
    const updatedCustomization = { ...bookmarksCustomization.value }
    if (updatedCustomization[oldNodePath]) {
      updatedCustomization[newNodePath] = updatedCustomization[oldNodePath]
      delete updatedCustomization[oldNodePath]
    }
    await settingStore.set(
      'Comfy.NodeLibrary.BookmarksCustomization',
      updatedCustomization
    )
  }

  const defaultBookmarkIcon = 'pi-bookmark-fill'
  const defaultBookmarkColor = '#a1a1aa'

  return {
    bookmarks,
    bookmarkedRoot,
    isBookmarked,
    toggleBookmark,
    addBookmark,
    addNewBookmarkFolder,
    renameBookmarkFolder,
    deleteBookmarkFolder,

    bookmarksCustomization,
    updateBookmarkCustomization,
    deleteBookmarkCustomization,
    renameBookmarkCustomization,
    defaultBookmarkIcon,
    defaultBookmarkColor
  }
})
