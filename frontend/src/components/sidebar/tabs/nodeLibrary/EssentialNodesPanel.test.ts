import { flushPromises, mount } from '@vue/test-utils'
import { nextTick, ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import type { RenderedTreeExplorerNode } from '@/types/treeExplorerTypes'

import EssentialNodesPanel from './EssentialNodesPanel.vue'

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => ({
    get: vi.fn().mockReturnValue('left')
  })
}))

vi.mock('@/composables/node/useNodeDragToCanvas', () => ({
  useNodeDragToCanvas: () => ({
    startDrag: vi.fn(),
    handleNativeDrop: vi.fn(),
    cancelDrag: vi.fn()
  })
}))

vi.mock('@/components/node/NodePreviewCard.vue', () => ({
  default: { template: '<div />' }
}))

describe('EssentialNodesPanel', () => {
  function createMockNode(
    name: string
  ): RenderedTreeExplorerNode<ComfyNodeDefImpl> {
    return {
      key: `node-${name}`,
      label: name,
      icon: 'icon-[comfy--node]',
      type: 'node',
      totalLeaves: 1,
      data: {
        name,
        display_name: name
      } as ComfyNodeDefImpl
    }
  }

  function createMockFolder(
    name: string,
    children: RenderedTreeExplorerNode<ComfyNodeDefImpl>[]
  ): RenderedTreeExplorerNode<ComfyNodeDefImpl> {
    return {
      key: `folder-${name}`,
      label: name,
      icon: 'icon-[lucide--folder]',
      type: 'folder',
      totalLeaves: children.length,
      children
    }
  }

  function createMockRoot(): RenderedTreeExplorerNode<ComfyNodeDefImpl> {
    return {
      key: 'root',
      label: 'Root',
      icon: '',
      type: 'folder',
      totalLeaves: 6,
      children: [
        createMockFolder('images', [
          createMockNode('LoadImage'),
          createMockNode('SaveImage')
        ]),
        createMockFolder('video', [
          createMockNode('LoadVideo'),
          createMockNode('SaveVideo')
        ]),
        createMockFolder('audio', [
          createMockNode('LoadAudio'),
          createMockNode('SaveAudio')
        ])
      ]
    }
  }

  function mountComponent(
    root = createMockRoot(),
    expandedKeys: string[] = []
  ) {
    const WrapperComponent = {
      template: `<EssentialNodesPanel :root="root" v-model:expandedKeys="keys" />`,
      components: { EssentialNodesPanel },
      setup() {
        const keys = ref(expandedKeys)
        return { root, keys }
      }
    }
    return mount(WrapperComponent, {
      global: {
        stubs: {
          Teleport: true,
          TabsContent: {
            template: '<div class="tabs-content"><slot /></div>'
          },
          CollapsibleRoot: {
            template:
              '<div class="collapsible-root" :data-state="open ? \'open\' : \'closed\'"><slot /></div>',
            props: ['open'],
            emits: ['update:open']
          },
          CollapsibleTrigger: {
            template:
              '<button class="collapsible-trigger" @click="$emit(\'click\')"><slot /></button>'
          },
          CollapsibleContent: {
            template: '<div class="collapsible-content"><slot /></div>'
          }
        }
      }
    })
  }

  describe('folder rendering', () => {
    it('should render all top-level folders', () => {
      const wrapper = mountComponent()
      const triggers = wrapper.findAll('.collapsible-trigger')
      expect(triggers).toHaveLength(3)
    })

    it('should display folder labels', () => {
      const wrapper = mountComponent()
      expect(wrapper.text()).toContain('images')
      expect(wrapper.text()).toContain('video')
      expect(wrapper.text()).toContain('audio')
    })
  })

  describe('default expansion', () => {
    it('should expand all folders by default when expandedKeys is empty', async () => {
      const wrapper = mountComponent(createMockRoot(), [])
      await nextTick()
      await flushPromises()
      await nextTick()

      const roots = wrapper.findAll('.collapsible-root')
      expect(roots[0].attributes('data-state')).toBe('open')
      expect(roots[1].attributes('data-state')).toBe('open')
      expect(roots[2].attributes('data-state')).toBe('open')
    })

    it('should respect provided expandedKeys', async () => {
      const wrapper = mountComponent(createMockRoot(), ['folder-audio'])
      await nextTick()

      const roots = wrapper.findAll('.collapsible-root')
      expect(roots[0].attributes('data-state')).toBe('closed')
      expect(roots[1].attributes('data-state')).toBe('closed')
      expect(roots[2].attributes('data-state')).toBe('open')
    })

    it('should expand all provided keys', async () => {
      const wrapper = mountComponent(createMockRoot(), [
        'folder-images',
        'folder-video',
        'folder-audio'
      ])
      await nextTick()

      const roots = wrapper.findAll('.collapsible-root')
      expect(roots[0].attributes('data-state')).toBe('open')
      expect(roots[1].attributes('data-state')).toBe('open')
      expect(roots[2].attributes('data-state')).toBe('open')
    })
  })

  describe('with single folder', () => {
    it('should expand only one folder when there is only one', async () => {
      const root: RenderedTreeExplorerNode<ComfyNodeDefImpl> = {
        key: 'root',
        label: 'Root',
        icon: '',
        type: 'folder',
        totalLeaves: 2,
        children: [
          createMockFolder('images', [
            createMockNode('LoadImage'),
            createMockNode('SaveImage')
          ])
        ]
      }

      const wrapper = mountComponent(root, [])
      await nextTick()
      await flushPromises()
      await nextTick()

      const roots = wrapper.findAll('.collapsible-root')
      expect(roots).toHaveLength(1)
      expect(roots[0].attributes('data-state')).toBe('open')
    })
  })

  describe('node cards', () => {
    it('should render node cards for each node in expanded folders', () => {
      const wrapper = mountComponent(createMockRoot(), ['folder-images'])
      const cards = wrapper.findAllComponents({ name: 'EssentialNodeCard' })
      expect(cards.length).toBeGreaterThanOrEqual(2)
    })
  })
})
