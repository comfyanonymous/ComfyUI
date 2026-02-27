import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import Badge from 'primevue/badge'
import PrimeVue from 'primevue/config'
import InputText from 'primevue/inputtext'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { createApp } from 'vue'
import { createI18n } from 'vue-i18n'

import EditableText from '@/components/common/EditableText.vue'
import TreeExplorerTreeNode from '@/components/common/TreeExplorerTreeNode.vue'
import type { RenderedTreeExplorerNode } from '@/types/treeExplorerTypes'
import { InjectKeyHandleEditLabelFunction } from '@/types/treeExplorerTypes'

// Create a mock i18n instance
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {}
})

describe('TreeExplorerTreeNode', () => {
  const mockNode = {
    key: '1',
    label: 'Test Node',
    leaf: false,
    totalLeaves: 3,
    icon: 'pi pi-folder',
    type: 'folder',
    handleRename: () => {}
  } as RenderedTreeExplorerNode

  const mockHandleEditLabel = vi.fn()

  beforeAll(() => {
    // Create a Vue app instance for PrimeVuePrimeVue
    const app = createApp({})
    app.use(PrimeVue)
    vi.useFakeTimers()
  })

  afterAll(() => {
    vi.useRealTimers()
  })

  it('renders correctly', () => {
    const wrapper = mount(TreeExplorerTreeNode, {
      props: { node: mockNode },
      global: {
        components: { EditableText, Badge },
        plugins: [createTestingPinia(), i18n],
        provide: {
          [InjectKeyHandleEditLabelFunction]: mockHandleEditLabel
        }
      }
    })

    expect(wrapper.find('.tree-node').exists()).toBe(true)
    expect(wrapper.find('.tree-folder').exists()).toBe(true)
    expect(wrapper.find('.tree-leaf').exists()).toBe(false)
    expect(wrapper.findComponent(EditableText).props('modelValue')).toBe(
      'Test Node'
    )
    // @ts-expect-error fixme ts strict error
    expect(wrapper.findComponent(Badge).props()['value'].toString()).toBe('3')
  })

  it('makes node label editable when renamingEditingNode matches', async () => {
    const wrapper = mount(TreeExplorerTreeNode, {
      props: {
        node: {
          ...mockNode,
          isEditingLabel: true
        }
      },
      global: {
        components: { EditableText, Badge, InputText },
        plugins: [createTestingPinia(), i18n, PrimeVue],
        provide: {
          [InjectKeyHandleEditLabelFunction]: mockHandleEditLabel
        }
      }
    })

    const editableText = wrapper.findComponent(EditableText)
    expect(editableText.props('isEditing')).toBe(true)
  })

  it('triggers handleEditLabel callback when editing is finished', async () => {
    const handleEditLabelMock = vi.fn()

    const wrapper = mount(TreeExplorerTreeNode, {
      props: {
        node: {
          ...mockNode,
          isEditingLabel: true
        }
      },
      global: {
        components: { EditableText, Badge, InputText },
        provide: { [InjectKeyHandleEditLabelFunction]: handleEditLabelMock },
        plugins: [createTestingPinia(), i18n, PrimeVue]
      }
    })

    const editableText = wrapper.findComponent(EditableText)
    editableText.vm.$emit('edit', 'New Node Name')
    expect(handleEditLabelMock).toHaveBeenCalledOnce()
  })
})
