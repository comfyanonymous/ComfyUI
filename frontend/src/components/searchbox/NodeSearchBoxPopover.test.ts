import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import PrimeVue from 'primevue/config'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent } from 'vue'
import { createI18n } from 'vue-i18n'

import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import type { FuseFilter, FuseFilterWithValue } from '@/utils/fuseUtil'

import NodeSearchBoxPopover from './NodeSearchBoxPopover.vue'

const mockStoreRefs = vi.hoisted(() => ({
  visible: { value: false },
  newSearchBoxEnabled: { value: true }
}))

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => ({
    get: vi.fn()
  })
}))

vi.mock('pinia', async () => {
  const actual = await vi.importActual('pinia')
  return {
    ...(actual as Record<string, unknown>),
    storeToRefs: () => mockStoreRefs
  }
})

vi.mock('@/stores/workspace/searchBoxStore', () => ({
  useSearchBoxStore: () => ({})
}))

vi.mock('@/services/litegraphService', () => ({
  useLitegraphService: () => ({
    getCanvasCenter: vi.fn(() => [0, 0]),
    addNodeOnGraph: vi.fn()
  })
}))

vi.mock('@/platform/workflow/management/stores/workflowStore', () => ({
  useWorkflowStore: () => ({
    activeWorkflow: null
  })
}))

vi.mock('@/renderer/core/canvas/canvasStore', () => ({
  useCanvasStore: () => ({
    canvas: null,
    getCanvas: vi.fn(() => ({
      linkConnector: {
        events: new EventTarget(),
        renderLinks: []
      }
    }))
  })
}))

vi.mock('@/stores/nodeDefStore', () => ({
  useNodeDefStore: () => ({
    nodeSearchService: {
      nodeFilters: [],
      inputTypeFilter: {},
      outputTypeFilter: {}
    }
  })
}))

const NodeSearchBoxStub = defineComponent({
  name: 'NodeSearchBox',
  props: {
    filters: { type: Array, default: () => [] }
  },
  template: '<div class="node-search-box" />'
})

function createFilter(
  id: string,
  value: string
): FuseFilterWithValue<ComfyNodeDefImpl, string> {
  return {
    filterDef: { id } as FuseFilter<ComfyNodeDefImpl, string>,
    value
  }
}

describe('NodeSearchBoxPopover', () => {
  const i18n = createI18n({
    legacy: false,
    locale: 'en',
    messages: { en: {} }
  })

  beforeEach(() => {
    setActivePinia(createPinia())
    mockStoreRefs.visible.value = false
  })

  const mountComponent = () => {
    return mount(NodeSearchBoxPopover, {
      global: {
        plugins: [i18n, PrimeVue],
        stubs: {
          NodeSearchBox: NodeSearchBoxStub,
          Dialog: {
            template: '<div><slot name="container" /></div>',
            props: ['visible', 'modal', 'dismissableMask', 'pt']
          }
        }
      }
    })
  }

  describe('addFilter duplicate prevention', () => {
    it('should add a filter when no duplicates exist', async () => {
      const wrapper = mountComponent()
      const searchBox = wrapper.findComponent(NodeSearchBoxStub)

      searchBox.vm.$emit('addFilter', createFilter('outputType', 'IMAGE'))
      await wrapper.vm.$nextTick()

      const filters = searchBox.props('filters') as FuseFilterWithValue<
        ComfyNodeDefImpl,
        string
      >[]
      expect(filters).toHaveLength(1)
      expect(filters[0]).toEqual(
        expect.objectContaining({
          filterDef: expect.objectContaining({ id: 'outputType' }),
          value: 'IMAGE'
        })
      )
    })

    it('should not add a duplicate filter with same id and value', async () => {
      const wrapper = mountComponent()
      const searchBox = wrapper.findComponent(NodeSearchBoxStub)

      searchBox.vm.$emit('addFilter', createFilter('outputType', 'IMAGE'))
      await wrapper.vm.$nextTick()
      searchBox.vm.$emit('addFilter', createFilter('outputType', 'IMAGE'))
      await wrapper.vm.$nextTick()

      expect(searchBox.props('filters')).toHaveLength(1)
    })

    it('should allow filters with same id but different values', async () => {
      const wrapper = mountComponent()
      const searchBox = wrapper.findComponent(NodeSearchBoxStub)

      searchBox.vm.$emit('addFilter', createFilter('outputType', 'IMAGE'))
      await wrapper.vm.$nextTick()
      searchBox.vm.$emit('addFilter', createFilter('outputType', 'MASK'))
      await wrapper.vm.$nextTick()

      expect(searchBox.props('filters')).toHaveLength(2)
    })

    it('should allow filters with different ids but same value', async () => {
      const wrapper = mountComponent()
      const searchBox = wrapper.findComponent(NodeSearchBoxStub)

      searchBox.vm.$emit('addFilter', createFilter('outputType', 'IMAGE'))
      await wrapper.vm.$nextTick()
      searchBox.vm.$emit('addFilter', createFilter('inputType', 'IMAGE'))
      await wrapper.vm.$nextTick()

      expect(searchBox.props('filters')).toHaveLength(2)
    })
  })
})
