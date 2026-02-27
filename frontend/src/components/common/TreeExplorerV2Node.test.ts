import { mount } from '@vue/test-utils'
import type { FlattenedItem } from 'reka-ui'
import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import type { RenderedTreeExplorerNode } from '@/types/treeExplorerTypes'
import { InjectKeyContextMenuNode } from '@/types/treeExplorerTypes'

import TreeExplorerV2Node from './TreeExplorerV2Node.vue'

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => ({
    get: vi.fn().mockReturnValue('left')
  })
}))

vi.mock('@/components/node/NodePreviewCard.vue', () => ({
  default: { template: '<div />' }
}))

const mockStartDrag = vi.fn()
const mockHandleNativeDrop = vi.fn()

vi.mock('@/composables/node/useNodeDragToCanvas', () => ({
  useNodeDragToCanvas: () => ({
    startDrag: mockStartDrag,
    handleNativeDrop: mockHandleNativeDrop
  })
}))

describe('TreeExplorerV2Node', () => {
  function createMockItem(
    type: 'node' | 'folder',
    overrides: Record<string, unknown> = {}
  ): FlattenedItem<RenderedTreeExplorerNode<ComfyNodeDefImpl>> {
    const value = {
      key: 'test-key',
      label: 'Test Label',
      type,
      icon: 'pi pi-folder',
      totalLeaves: 5,
      ...overrides
    } as RenderedTreeExplorerNode<ComfyNodeDefImpl>
    return {
      _id: 'test-id',
      index: 0,
      value,
      level: 1,
      hasChildren: type === 'folder',
      bind: { value, level: 1 }
    }
  }

  function createTreeItemStub() {
    const handleToggle = vi.fn()
    const handleSelect = vi.fn()
    return {
      handleToggle,
      handleSelect,
      stub: {
        template: `<div data-testid="tree-item"><slot :isExpanded="false" :isSelected="false" :handleToggle="handleToggle" :handleSelect="handleSelect" /></div>`,
        setup() {
          return { handleToggle, handleSelect }
        }
      }
    }
  }

  function mountComponent(
    props: Record<string, unknown> = {},
    options: {
      provide?: Record<string, unknown>
      treeItemStub?: ReturnType<typeof createTreeItemStub>
    } = {}
  ) {
    const treeItemStub = options.treeItemStub ?? createTreeItemStub()
    return {
      wrapper: mount(TreeExplorerV2Node, {
        global: {
          stubs: {
            TreeItem: treeItemStub.stub,
            Teleport: { template: '<div />' }
          },
          provide: {
            ...options.provide
          }
        },
        props: {
          item: createMockItem('node'),
          ...props
        }
      }),
      treeItemStub
    }
  }

  describe('handleClick', () => {
    it('emits nodeClick event when clicked', async () => {
      const { wrapper } = mountComponent({
        item: createMockItem('node')
      })

      const nodeDiv = wrapper.find('div.group\\/tree-node')
      await nodeDiv.trigger('click')

      expect(wrapper.emitted('nodeClick')).toBeTruthy()
      expect(wrapper.emitted('nodeClick')?.[0]?.[0]).toMatchObject({
        type: 'node',
        label: 'Test Label'
      })
    })

    it('calls handleToggle for folder items', async () => {
      const treeItemStub = createTreeItemStub()
      const { wrapper } = mountComponent(
        { item: createMockItem('folder') },
        { treeItemStub }
      )

      const folderDiv = wrapper.find('div.group\\/tree-node')
      await folderDiv.trigger('click')

      expect(wrapper.emitted('nodeClick')).toBeTruthy()
      expect(treeItemStub.handleToggle).toHaveBeenCalled()
    })

    it('does not call handleToggle for node items', async () => {
      const treeItemStub = createTreeItemStub()
      const { wrapper } = mountComponent(
        { item: createMockItem('node') },
        { treeItemStub }
      )

      const nodeDiv = wrapper.find('div.group\\/tree-node')
      await nodeDiv.trigger('click')

      expect(wrapper.emitted('nodeClick')).toBeTruthy()
      expect(treeItemStub.handleToggle).not.toHaveBeenCalled()
    })
  })

  describe('context menu', () => {
    it('sets contextMenuNode when contextmenu event is triggered on node', async () => {
      const contextMenuNode = ref<RenderedTreeExplorerNode | null>(null)
      const nodeItem = createMockItem('node')

      const { wrapper } = mountComponent(
        { item: nodeItem },
        {
          provide: {
            [InjectKeyContextMenuNode as symbol]: contextMenuNode
          }
        }
      )

      const nodeDiv = wrapper.find('div.group\\/tree-node')
      await nodeDiv.trigger('contextmenu')

      expect(contextMenuNode.value).toEqual(nodeItem.value)
    })

    it('does not set contextMenuNode for folder items', async () => {
      const contextMenuNode = ref<RenderedTreeExplorerNode | null>(null)

      const { wrapper } = mountComponent(
        { item: createMockItem('folder') },
        {
          provide: {
            [InjectKeyContextMenuNode as symbol]: contextMenuNode
          }
        }
      )

      const folderDiv = wrapper.find('div.group\\/tree-node')
      await folderDiv.trigger('contextmenu')

      expect(contextMenuNode.value).toBeNull()
    })
  })

  describe('rendering', () => {
    it('renders node icon for node type', () => {
      const { wrapper } = mountComponent({
        item: createMockItem('node')
      })

      expect(wrapper.find('i.icon-\\[comfy--node\\]').exists()).toBe(true)
    })

    it('renders folder icon for folder type', () => {
      const { wrapper } = mountComponent({
        item: createMockItem('folder', { icon: 'icon-[lucide--folder]' })
      })

      expect(wrapper.find('i.icon-\\[lucide--folder\\]').exists()).toBe(true)
    })

    it('renders label text', () => {
      const { wrapper } = mountComponent({
        item: createMockItem('node', { label: 'My Node' })
      })

      expect(wrapper.text()).toContain('My Node')
    })

    it('renders chevron for folder with children', () => {
      const { wrapper } = mountComponent({
        item: {
          ...createMockItem('folder'),
          hasChildren: true
        }
      })

      expect(wrapper.find('i.icon-\\[lucide--chevron-down\\]').exists()).toBe(
        true
      )
    })
  })

  describe('drag and drop', () => {
    beforeEach(() => {
      vi.clearAllMocks()
    })

    it('sets draggable attribute on node items', () => {
      const { wrapper } = mountComponent({
        item: createMockItem('node')
      })

      const nodeDiv = wrapper.find('div.group\\/tree-node')
      expect(nodeDiv.attributes('draggable')).toBe('true')
    })

    it('does not set draggable on folder items', () => {
      const { wrapper } = mountComponent({
        item: createMockItem('folder')
      })

      const folderDiv = wrapper.find('div.group\\/tree-node')
      expect(folderDiv.attributes('draggable')).toBeUndefined()
    })

    it('calls startDrag with native mode on dragstart', async () => {
      const mockData = { name: 'TestNode' }
      const { wrapper } = mountComponent({
        item: createMockItem('node', { data: mockData })
      })

      const nodeDiv = wrapper.find('div.group\\/tree-node')
      await nodeDiv.trigger('dragstart')

      expect(mockStartDrag).toHaveBeenCalledWith(mockData, 'native')
    })

    it('does not call startDrag for folder items on dragstart', async () => {
      const { wrapper } = mountComponent({
        item: createMockItem('folder')
      })

      const folderDiv = wrapper.find('div.group\\/tree-node')
      await folderDiv.trigger('dragstart')

      expect(mockStartDrag).not.toHaveBeenCalled()
    })

    it('calls handleNativeDrop on dragend with drop coordinates', async () => {
      const mockData = { name: 'TestNode' }
      const { wrapper } = mountComponent({
        item: createMockItem('node', { data: mockData })
      })

      const nodeDiv = wrapper.find('div.group\\/tree-node')

      await nodeDiv.trigger('dragstart')

      const dragEndEvent = new DragEvent('dragend', { bubbles: true })
      Object.defineProperty(dragEndEvent, 'clientX', { value: 100 })
      Object.defineProperty(dragEndEvent, 'clientY', { value: 200 })

      await nodeDiv.element.dispatchEvent(dragEndEvent)
      await wrapper.vm.$nextTick()

      expect(mockHandleNativeDrop).toHaveBeenCalledWith(100, 200)
    })

    it('calls handleNativeDrop regardless of dropEffect', async () => {
      const mockData = { name: 'TestNode' }
      const { wrapper } = mountComponent({
        item: createMockItem('node', { data: mockData })
      })

      const nodeDiv = wrapper.find('div.group\\/tree-node')

      await nodeDiv.trigger('dragstart')
      mockHandleNativeDrop.mockClear()

      const dragEndEvent = new DragEvent('dragend', { bubbles: true })
      Object.defineProperty(dragEndEvent, 'clientX', { value: 300 })
      Object.defineProperty(dragEndEvent, 'clientY', { value: 400 })
      Object.defineProperty(dragEndEvent, 'dataTransfer', {
        value: { dropEffect: 'none' }
      })

      await nodeDiv.element.dispatchEvent(dragEndEvent)
      await wrapper.vm.$nextTick()

      expect(mockHandleNativeDrop).toHaveBeenCalledWith(300, 400)
    })
  })
})
