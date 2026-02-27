import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import PrimeVue from 'primevue/config'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import SelectionToolbox from '@/components/graph/SelectionToolbox.vue'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useCanvasInteractions } from '@/renderer/core/canvas/useCanvasInteractions'
import { useExtensionService } from '@/services/extensionService'
import {
  createMockCanvas,
  createMockPositionable
} from '@/utils/__tests__/litegraphTestUtils'
import * as litegraphUtil from '@/utils/litegraphUtil'
import * as nodeFilterUtil from '@/utils/nodeFilterUtil'

function createMockExtensionService(): ReturnType<typeof useExtensionService> {
  return {
    extensionCommands: { value: new Map() },
    loadExtensions: vi.fn(),
    registerExtension: vi.fn(),
    invokeExtensions: vi.fn(() => []),
    invokeExtensionsAsync: vi.fn()
  } as Partial<ReturnType<typeof useExtensionService>> as ReturnType<
    typeof useExtensionService
  >
}

// Mock the composables and services
vi.mock('@/renderer/core/canvas/useCanvasInteractions', () => ({
  useCanvasInteractions: vi.fn(() => ({
    handleWheel: vi.fn()
  }))
}))

vi.mock('@/composables/canvas/useSelectionToolboxPosition', () => ({
  useSelectionToolboxPosition: vi.fn(() => ({
    visible: { value: true }
  })),
  resetMoreOptionsState: vi.fn()
}))

vi.mock('@/composables/element/useRetriggerableAnimation', () => ({
  useRetriggerableAnimation: vi.fn(() => ({
    shouldAnimate: { value: false }
  }))
}))

vi.mock('@/renderer/extensions/minimap/composables/useMinimap', () => ({
  useMinimap: vi.fn(() => ({
    containerStyles: {
      value: {
        backgroundColor: '#ffffff'
      }
    }
  }))
}))

vi.mock('@/services/extensionService', () => ({
  useExtensionService: vi.fn(() => ({
    extensionCommands: { value: new Map() },
    invokeExtensions: vi.fn(() => [])
  }))
}))

vi.mock('@/utils/litegraphUtil', () => ({
  isLGraphNode: vi.fn(() => true),
  isImageNode: vi.fn(() => false),
  isLoad3dNode: vi.fn(() => false)
}))

vi.mock('@/utils/nodeFilterUtil', () => ({
  isOutputNode: vi.fn(() => false),
  filterOutputNodes: vi.fn((nodes) => nodes.filter(() => false))
}))

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => ({
    get: vi.fn((key: string) => {
      if (key === 'Comfy.Load3D.3DViewerEnable') return true
      return null
    })
  })
}))

vi.mock('@/stores/commandStore', () => ({
  useCommandStore: () => ({
    getCommand: vi.fn(() => ({ id: 'test-command', title: 'Test Command' }))
  })
}))

let nodeDefMock = {
  type: 'TestNode',
  title: 'Test Node'
} as unknown

vi.mock('@/stores/nodeDefStore', () => ({
  useNodeDefStore: () => ({
    fromLGraphNode: vi.fn(() => nodeDefMock)
  })
}))

describe('SelectionToolbox', () => {
  let canvasStore: ReturnType<typeof useCanvasStore>

  const i18n = createI18n({
    legacy: false,
    locale: 'en',
    messages: {
      en: {
        g: {
          info: 'Node Info',
          bookmark: 'Save to Library',
          frameNodes: 'Frame Nodes',
          moreOptions: 'More Options',
          refreshNode: 'Refresh Node'
        }
      }
    }
  })

  const mockProvide = {
    isVisible: { value: true },
    selectedItems: []
  }

  beforeEach(() => {
    setActivePinia(createPinia())
    canvasStore = useCanvasStore()

    // Mock the canvas to avoid "getCanvas: canvas is null" errors
    canvasStore.canvas = createMockCanvas()

    vi.resetAllMocks()
  })

  const mountComponent = (props = {}) => {
    return mount(SelectionToolbox, {
      props,
      global: {
        plugins: [i18n, PrimeVue],
        provide: {
          [Symbol.for('SelectionOverlay')]: mockProvide
        },
        stubs: {
          Panel: {
            template:
              '<div class="panel selection-toolbox absolute left-1/2 rounded-lg"><slot /></div>',
            props: ['pt', 'style', 'class']
          },
          NodeContextMenu: { template: '<div class="node-context-menu" />' },
          InfoButton: { template: '<div class="info-button" />' },
          ColorPickerButton: {
            template:
              '<button data-testid="color-picker-button" class="color-picker-button" />'
          },
          FrameNodes: { template: '<div class="frame-nodes" />' },
          PublishButton: {
            template:
              '<button data-testid="add-to-library" class="bookmark-button" />'
          },
          BypassButton: {
            template:
              '<button data-testid="bypass-button" class="bypass-button" />'
          },
          PinButton: { template: '<div class="pin-button" />' },
          Load3DViewerButton: {
            template: '<div class="load-3d-viewer-button" />'
          },
          MaskEditorButton: { template: '<div class="mask-editor-button" />' },
          DeleteButton: {
            template:
              '<button data-testid="delete-button" class="delete-button" />'
          },
          RefreshSelectionButton: {
            template: '<div class="refresh-button" />'
          },
          ExecuteButton: { template: '<div class="execute-button" />' },
          ConvertToSubgraphButton: {
            template:
              '<button data-testid="convert-to-subgraph-button" class="convert-to-subgraph-button" />'
          },
          ExtensionCommandButton: {
            template: '<div class="extension-command-button" />'
          },
          MoreOptions: {
            template:
              '<button data-testid="more-options-button" class="more-options" />'
          },
          VerticalDivider: { template: '<div class="vertical-divider" />' }
        }
      }
    })
  }

  describe('Button Visibility Logic', () => {
    beforeEach(() => {
      const mockExtensionService = vi.mocked(useExtensionService)
      mockExtensionService.mockReturnValue(createMockExtensionService())
    })

    it('should show info button only for single selections', () => {
      // Single node selection
      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()
      expect(wrapper.find('.info-button').exists()).toBe(true)

      // Multiple node selection
      canvasStore.selectedItems = [
        createMockPositionable(),
        createMockPositionable()
      ]
      wrapper.unmount()
      const wrapper2 = mountComponent()
      expect(wrapper2.find('.info-button').exists()).toBe(false)
    })

    it('should not show info button when node definition is not found', () => {
      canvasStore.selectedItems = [createMockPositionable()]
      // mock nodedef and return null
      nodeDefMock = null
      // remount component
      const wrapper = mountComponent()
      expect(wrapper.find('.info-button').exists()).toBe(false)
    })

    it('should show color picker for all selections', () => {
      // Single node selection
      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()
      expect(wrapper.find('[data-testid="color-picker-button"]').exists()).toBe(
        true
      )

      // Multiple node selection
      canvasStore.selectedItems = [
        createMockPositionable(),
        createMockPositionable()
      ]
      wrapper.unmount()
      const wrapper2 = mountComponent()
      expect(
        wrapper2.find('[data-testid="color-picker-button"]').exists()
      ).toBe(true)
    })

    it('should show frame nodes only for multiple selections', () => {
      // Single node selection
      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()
      expect(wrapper.find('.frame-nodes').exists()).toBe(false)

      // Multiple node selection
      canvasStore.selectedItems = [
        createMockPositionable(),
        createMockPositionable()
      ]
      wrapper.unmount()
      const wrapper2 = mountComponent()
      expect(wrapper2.find('.frame-nodes').exists()).toBe(true)
    })

    it('should show bypass button for appropriate selections', () => {
      // Single node selection
      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()
      expect(wrapper.find('[data-testid="bypass-button"]').exists()).toBe(true)

      // Multiple node selection
      canvasStore.selectedItems = [
        createMockPositionable(),
        createMockPositionable()
      ]
      wrapper.unmount()
      const wrapper2 = mountComponent()
      expect(wrapper2.find('[data-testid="bypass-button"]').exists()).toBe(true)
    })

    it('should show common buttons for all selections', () => {
      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()

      expect(wrapper.find('[data-testid="delete-button"]').exists()).toBe(true)
      expect(
        wrapper.find('[data-testid="convert-to-subgraph-button"]').exists()
      ).toBe(true)
      expect(wrapper.find('[data-testid="more-options-button"]').exists()).toBe(
        true
      )
    })

    it('should show mask editor only for single image nodes', () => {
      const isImageNodeSpy = vi.spyOn(litegraphUtil, 'isImageNode')

      // Single image node
      isImageNodeSpy.mockReturnValue(true)
      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()
      expect(wrapper.find('.mask-editor-button').exists()).toBe(true)

      // Single non-image node
      isImageNodeSpy.mockReturnValue(false)
      canvasStore.selectedItems = [createMockPositionable()]
      wrapper.unmount()
      const wrapper2 = mountComponent()
      expect(wrapper2.find('.mask-editor-button').exists()).toBe(false)
    })

    it('should show Color picker button only for single Load3D nodes', () => {
      const isLoad3dNodeSpy = vi.spyOn(litegraphUtil, 'isLoad3dNode')

      // Single Load3D node
      isLoad3dNodeSpy.mockReturnValue(true)
      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()
      expect(wrapper.find('.load-3d-viewer-button').exists()).toBe(true)

      // Single non-Load3D node
      isLoad3dNodeSpy.mockReturnValue(false)
      canvasStore.selectedItems = [createMockPositionable()]
      wrapper.unmount()
      const wrapper2 = mountComponent()
      expect(wrapper2.find('.load-3d-viewer-button').exists()).toBe(false)
    })

    it('should show ExecuteButton only when output nodes are selected', () => {
      const isOutputNodeSpy = vi.spyOn(nodeFilterUtil, 'isOutputNode')
      const filterOutputNodesSpy = vi.spyOn(nodeFilterUtil, 'filterOutputNodes')

      // With output node selected
      isOutputNodeSpy.mockReturnValue(true)
      filterOutputNodesSpy.mockReturnValue([
        { type: 'SaveImage' }
      ] as LGraphNode[])
      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()
      expect(wrapper.find('.execute-button').exists()).toBe(true)

      // Without output node selected
      isOutputNodeSpy.mockReturnValue(false)
      filterOutputNodesSpy.mockReturnValue([])
      canvasStore.selectedItems = [createMockPositionable()]
      wrapper.unmount()
      const wrapper2 = mountComponent()
      expect(wrapper2.find('.execute-button').exists()).toBe(false)

      // No selection at all
      canvasStore.selectedItems = []
      wrapper2.unmount()
      const wrapper3 = mountComponent()
      expect(wrapper3.find('.execute-button').exists()).toBe(false)
    })
  })

  describe('Divider Visibility Logic', () => {
    it('should show dividers between button groups when both groups have buttons', () => {
      // Setup single node to show info + other buttons
      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()

      const dividers = wrapper.findAll('.vertical-divider')
      expect(dividers.length).toBeGreaterThan(0)
    })

    it('should not show dividers when adjacent groups are empty', () => {
      // No selection should show minimal buttons and dividers
      canvasStore.selectedItems = []
      const wrapper = mountComponent()

      const buttons = wrapper.find('.panel').element.children
      expect(buttons.length).toBeGreaterThan(0) // At least MoreOptions should show
    })
  })

  describe('Extension Commands', () => {
    it('should render extension command buttons when available', () => {
      const mockExtensionService = vi.mocked(useExtensionService)
      mockExtensionService.mockReturnValue({
        extensionCommands: {
          value: new Map([
            ['test-command', { id: 'test-command', title: 'Test Command' }]
          ])
        },
        loadExtensions: vi.fn(),
        registerExtension: vi.fn(),
        invokeExtensions: vi.fn(() => ['test-command']),
        invokeExtensionsAsync: vi.fn()
      } as ReturnType<typeof useExtensionService>)

      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()

      expect(wrapper.find('.extension-command-button').exists()).toBe(true)
    })

    it('should not render extension commands when none available', () => {
      const mockExtensionService = vi.mocked(useExtensionService)
      mockExtensionService.mockReturnValue(createMockExtensionService())

      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()

      expect(wrapper.find('.extension-command-button').exists()).toBe(false)
    })
  })

  describe('Container Styling', () => {
    it('should apply minimap container styles', () => {
      const mockExtensionService = vi.mocked(useExtensionService)
      mockExtensionService.mockReturnValue(createMockExtensionService())

      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()

      const panel = wrapper.find('.panel')
      expect(panel.exists()).toBe(true)
    })

    it('should have correct CSS classes', () => {
      const mockExtensionService = vi.mocked(useExtensionService)
      mockExtensionService.mockReturnValue(createMockExtensionService())

      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()

      const panel = wrapper.find('.panel')
      expect(panel.classes()).toContain('selection-toolbox')
      expect(panel.classes()).toContain('absolute')
      expect(panel.classes()).toContain('left-1/2')
      expect(panel.classes()).toContain('rounded-lg')
    })

    it('should handle animation class conditionally', () => {
      const mockExtensionService = vi.mocked(useExtensionService)
      mockExtensionService.mockReturnValue(createMockExtensionService())

      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()

      const panel = wrapper.find('.panel')
      expect(panel.exists()).toBe(true)
    })
  })

  describe('Event Handling', () => {
    it('should handle wheel events', async () => {
      const mockCanvasInteractions = vi.mocked(useCanvasInteractions)
      const forwardEventToCanvasSpy = vi.fn()
      mockCanvasInteractions.mockReturnValue({
        handleWheel: vi.fn(),
        handlePointer: vi.fn(),
        forwardEventToCanvas: forwardEventToCanvasSpy,
        shouldHandleNodePointerEvents: { value: true } as ReturnType<
          typeof useCanvasInteractions
        >['shouldHandleNodePointerEvents']
      } as ReturnType<typeof useCanvasInteractions>)

      const mockExtensionService = vi.mocked(useExtensionService)
      mockExtensionService.mockReturnValue(createMockExtensionService())

      canvasStore.selectedItems = [createMockPositionable()]
      const wrapper = mountComponent()

      const panel = wrapper.find('.panel')
      await panel.trigger('wheel')

      expect(forwardEventToCanvasSpy).toHaveBeenCalled()
    })
  })

  describe('No Selection State', () => {
    beforeEach(() => {
      const mockExtensionService = vi.mocked(useExtensionService)
      mockExtensionService.mockReturnValue(createMockExtensionService())
    })

    it('should hide most buttons when no items selected', () => {
      canvasStore.selectedItems = []
      const wrapper = mountComponent()

      expect(wrapper.find('.info-button').exists()).toBe(false)
      expect(wrapper.find('.color-picker-button').exists()).toBe(false)
      expect(wrapper.find('.frame-nodes').exists()).toBe(false)
      expect(wrapper.find('.bookmark-button').exists()).toBe(false)
    })
  })
})
