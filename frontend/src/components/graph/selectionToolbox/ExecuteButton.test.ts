import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import PrimeVue from 'primevue/config'
import Tooltip from 'primevue/tooltip'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import ExecuteButton from '@/components/graph/selectionToolbox/ExecuteButton.vue'
import { useSelectionState } from '@/composables/graph/useSelectionState'
import type { LGraphCanvas, LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useCommandStore } from '@/stores/commandStore'

// Mock the utils
vi.mock('@/utils/litegraphUtil', () => ({
  isLGraphNode: vi.fn((node) => !!node?.type)
}))

vi.mock('@/utils/nodeFilterUtil', () => ({
  isOutputNode: vi.fn((node) => !!node?.constructor?.nodeData?.output_node)
}))

// Mock the composables
vi.mock('@/composables/graph/useSelectionState', () => ({
  useSelectionState: vi.fn(() => ({
    selectedNodes: {
      value: []
    }
  }))
}))

describe('ExecuteButton', () => {
  let mockCanvas: LGraphCanvas
  let mockSelectedNodes: LGraphNode[]

  const i18n = createI18n({
    legacy: false,
    locale: 'en',
    messages: {
      en: {
        selectionToolbox: {
          executeButton: {
            tooltip: 'Execute selected nodes'
          }
        }
      }
    }
  })

  beforeEach(() => {
    // Set up Pinia with testing utilities
    setActivePinia(
      createTestingPinia({
        createSpy: vi.fn
      })
    )

    // Reset mocks
    const partialCanvas: Partial<LGraphCanvas> = {
      setDirty: vi.fn()
    }
    mockCanvas = partialCanvas as Partial<LGraphCanvas> as LGraphCanvas

    mockSelectedNodes = []

    // Get store instances and mock methods
    const canvasStore = useCanvasStore()
    const commandStore = useCommandStore()

    vi.spyOn(canvasStore, 'getCanvas').mockReturnValue(mockCanvas)
    vi.spyOn(commandStore, 'execute').mockResolvedValue()

    // Update the useSelectionState mock
    vi.mocked(useSelectionState).mockReturnValue({
      selectedNodes: {
        value: mockSelectedNodes
      }
    } as ReturnType<typeof useSelectionState>)

    vi.clearAllMocks()
  })

  const mountComponent = () => {
    return mount(ExecuteButton, {
      global: {
        plugins: [i18n, PrimeVue],
        directives: { tooltip: Tooltip },
        stubs: {
          'i-lucide:play': { template: '<div class="play-icon" />' }
        }
      }
    })
  }

  describe('Rendering', () => {
    it('should be able to render', () => {
      const wrapper = mountComponent()
      const button = wrapper.find('button')
      expect(button.exists()).toBe(true)
    })
  })

  describe('Click Handler', () => {
    it('should execute Comfy.QueueSelectedOutputNodes command on click', async () => {
      const commandStore = useCommandStore()
      const wrapper = mountComponent()
      const button = wrapper.find('button')

      await button.trigger('click')

      expect(commandStore.execute).toHaveBeenCalledWith(
        'Comfy.QueueSelectedOutputNodes'
      )
      expect(commandStore.execute).toHaveBeenCalledTimes(1)
    })
  })
})
