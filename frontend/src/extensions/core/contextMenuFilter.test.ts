import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { IContextMenuValue } from '@/lib/litegraph/src/interfaces'
import type { LGraphCanvas, LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useExtensionService } from '@/services/extensionService'
import { useExtensionStore } from '@/stores/extensionStore'
import type { ComfyExtension } from '@/types/comfy'
import {
  createMockCanvas,
  createMockLGraphNode
} from '@/utils/__tests__/litegraphTestUtils'

describe('Context Menu Extension API', () => {
  let mockCanvas: LGraphCanvas
  let mockNode: LGraphNode
  let extensionStore: ReturnType<typeof useExtensionStore>
  let extensionService: ReturnType<typeof useExtensionService>

  // Mock menu items
  const canvasMenuItem1: IContextMenuValue = {
    content: 'Canvas Item 1',
    callback: () => {}
  }
  const canvasMenuItem2: IContextMenuValue = {
    content: 'Canvas Item 2',
    callback: () => {}
  }
  const nodeMenuItem1: IContextMenuValue = {
    content: 'Node Item 1',
    callback: () => {}
  }
  const nodeMenuItem2: IContextMenuValue = {
    content: 'Node Item 2',
    callback: () => {}
  }

  // Mock extensions
  const createCanvasMenuExtension = (
    name: string,
    items: (IContextMenuValue | null)[]
  ): ComfyExtension => ({
    name,
    getCanvasMenuItems: () => items
  })

  const createNodeMenuExtension = (
    name: string,
    items: IContextMenuValue[]
  ): ComfyExtension => ({
    name,
    getNodeMenuItems: () => items
  })

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    extensionStore = useExtensionStore()
    extensionService = useExtensionService()

    mockCanvas = createMockCanvas({
      graph_mouse: [100, 100],
      selectedItems: new Set()
    })

    mockNode = createMockLGraphNode({
      id: 1,
      type: 'TestNode',
      pos: [0, 0]
    })
  })

  describe('collectCanvasMenuItems', () => {
    it('should call getCanvasMenuItems and collect into flat array', () => {
      const ext1 = createCanvasMenuExtension('Extension 1', [canvasMenuItem1])
      const ext2 = createCanvasMenuExtension('Extension 2', [
        canvasMenuItem2,
        { content: 'Item 3', callback: () => {} }
      ])

      extensionStore.registerExtension(ext1)
      extensionStore.registerExtension(ext2)

      const items: IContextMenuValue[] = extensionService
        .invokeExtensions('getCanvasMenuItems', mockCanvas)
        .flat() as IContextMenuValue[]

      expect(items).toHaveLength(3)
      expect(items[0]).toMatchObject({ content: 'Canvas Item 1' })
      expect(items[1]).toMatchObject({ content: 'Canvas Item 2' })
      expect(items[2]).toMatchObject({ content: 'Item 3' })
    })

    it('should support submenus and separators', () => {
      const extension = createCanvasMenuExtension('Test Extension', [
        {
          content: 'Menu with Submenu',
          has_submenu: true,
          submenu: {
            options: [
              { content: 'Submenu Item 1', callback: () => {} },
              { content: 'Submenu Item 2', callback: () => {} }
            ]
          }
        },
        null,
        { content: 'After Separator', callback: () => {} }
      ])

      extensionStore.registerExtension(extension)

      const items: IContextMenuValue[] = extensionService
        .invokeExtensions('getCanvasMenuItems', mockCanvas)
        .flat() as IContextMenuValue[]

      expect(items).toHaveLength(3)
      expect(items[0].content).toBe('Menu with Submenu')
      expect(items[0].submenu?.options).toHaveLength(2)
      expect(items[1]).toBeNull()
      expect(items[2].content).toBe('After Separator')
    })

    it('should skip extensions without getCanvasMenuItems', () => {
      const canvasExtension = createCanvasMenuExtension('Canvas Ext', [
        canvasMenuItem1
      ])
      const extensionWithoutCanvasMenu: ComfyExtension = {
        name: 'No Canvas Menu'
      }

      extensionStore.registerExtension(canvasExtension)
      extensionStore.registerExtension(extensionWithoutCanvasMenu)

      const items: IContextMenuValue[] = extensionService
        .invokeExtensions('getCanvasMenuItems', mockCanvas)
        .flat() as IContextMenuValue[]

      expect(items).toHaveLength(1)
      expect(items[0].content).toBe('Canvas Item 1')
    })

    it('should not duplicate menu items when collected multiple times', () => {
      const extension = createCanvasMenuExtension('Test Extension', [
        canvasMenuItem1,
        canvasMenuItem2
      ])

      extensionStore.registerExtension(extension)

      // Collect items multiple times (simulating repeated menu opens)
      const items1: IContextMenuValue[] = extensionService
        .invokeExtensions('getCanvasMenuItems', mockCanvas)
        .flat() as IContextMenuValue[]

      const items2: IContextMenuValue[] = extensionService
        .invokeExtensions('getCanvasMenuItems', mockCanvas)
        .flat() as IContextMenuValue[]

      // Both collections should have the same items (no duplication)
      expect(items1).toHaveLength(2)
      expect(items2).toHaveLength(2)

      // Verify items are unique by checking their content
      const contents1 = items1.map((item) => item.content)
      const uniqueContents1 = new Set(contents1)
      expect(uniqueContents1.size).toBe(contents1.length)

      const contents2 = items2.map((item) => item.content)
      const uniqueContents2 = new Set(contents2)
      expect(uniqueContents2.size).toBe(contents2.length)
    })
  })

  describe('collectNodeMenuItems', () => {
    it('should call getNodeMenuItems and collect into flat array', () => {
      const ext1 = createNodeMenuExtension('Extension 1', [nodeMenuItem1])
      const ext2 = createNodeMenuExtension('Extension 2', [
        nodeMenuItem2,
        { content: 'Item 3', callback: () => {} }
      ])

      extensionStore.registerExtension(ext1)
      extensionStore.registerExtension(ext2)

      const items: IContextMenuValue[] = extensionService
        .invokeExtensions('getNodeMenuItems', mockNode)
        .flat() as IContextMenuValue[]

      expect(items).toHaveLength(3)
      expect(items[0]).toMatchObject({ content: 'Node Item 1' })
      expect(items[1]).toMatchObject({ content: 'Node Item 2' })
    })

    it('should support submenus', () => {
      const extension = createNodeMenuExtension('Submenu Extension', [
        {
          content: 'Node Menu with Submenu',
          has_submenu: true,
          submenu: {
            options: [
              { content: 'Node Submenu 1', callback: () => {} },
              { content: 'Node Submenu 2', callback: () => {} }
            ]
          }
        }
      ])

      extensionStore.registerExtension(extension)

      const items: IContextMenuValue[] = extensionService
        .invokeExtensions('getNodeMenuItems', mockNode)
        .flat() as IContextMenuValue[]

      expect(items[0].content).toBe('Node Menu with Submenu')
      expect(items[0].submenu?.options).toHaveLength(2)
    })

    it('should skip extensions without getNodeMenuItems', () => {
      const nodeExtension = createNodeMenuExtension('Node Ext', [nodeMenuItem1])
      const extensionWithoutNodeMenu: ComfyExtension = {
        name: 'No Node Menu'
      }

      extensionStore.registerExtension(nodeExtension)
      extensionStore.registerExtension(extensionWithoutNodeMenu)

      const items: IContextMenuValue[] = extensionService
        .invokeExtensions('getNodeMenuItems', mockNode)
        .flat() as IContextMenuValue[]

      expect(items).toHaveLength(1)
      expect(items[0].content).toBe('Node Item 1')
    })
  })
})
