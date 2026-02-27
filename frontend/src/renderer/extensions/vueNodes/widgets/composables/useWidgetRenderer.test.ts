import { setActivePinia } from 'pinia'
import { createTestingPinia } from '@pinia/testing'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getComponent,
  isEssential,
  shouldRenderAsVue,
  FOR_TESTING
} from '@/renderer/extensions/vueNodes/widgets/registry/widgetRegistry'
import type { SafeWidgetData } from '@/composables/graph/useGraphNodeManager'

const {
  WidgetButton,
  WidgetColorPicker,
  WidgetInputNumber,
  WidgetInputText,
  WidgetMarkdown,
  WidgetSelect,
  WidgetTextarea,
  WidgetToggleSwitch
} = FOR_TESTING

vi.mock('@/stores/queueStore', () => ({
  useQueueStore: vi.fn(() => ({
    historyTasks: []
  }))
}))

// Mock the settings store for components that might use it
vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => ({
    get: vi.fn(() => 'before')
  })
}))

describe('widgetRegistry', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia())
    vi.clearAllMocks()
  })
  describe('getComponent', () => {
    // Test number type mappings
    describe('number types', () => {
      it('should map int types to slider widget', () => {
        expect(getComponent('int')).toBe(WidgetInputNumber)
        expect(getComponent('INT')).toBe(WidgetInputNumber)
      })

      it('should map float types to slider widget', () => {
        expect(getComponent('float')).toBe(WidgetInputNumber)
        expect(getComponent('FLOAT')).toBe(WidgetInputNumber)
        expect(getComponent('number')).toBe(WidgetInputNumber)
        expect(getComponent('slider')).toBe(WidgetInputNumber)
      })
    })

    // Test text type mappings
    describe('text types', () => {
      it('should map text variations to input text widget', () => {
        expect(getComponent('text')).toBe(WidgetInputText)
        expect(getComponent('string')).toBe(WidgetInputText)
        expect(getComponent('STRING')).toBe(WidgetInputText)
      })

      it('should map multiline text types to textarea widget', () => {
        expect(getComponent('multiline')).toBe(WidgetTextarea)
        expect(getComponent('textarea')).toBe(WidgetTextarea)
        expect(getComponent('TEXTAREA')).toBe(WidgetTextarea)
        expect(getComponent('customtext')).toBe(WidgetTextarea)
      })

      it('should map markdown to markdown widget', () => {
        expect(getComponent('MARKDOWN')).toBe(WidgetMarkdown)
        expect(getComponent('markdown')).toBe(WidgetMarkdown)
      })
    })

    // Test selection type mappings
    describe('selection types', () => {
      it('should map combo types to select widget', () => {
        expect(getComponent('combo')).toBe(WidgetSelect)
        expect(getComponent('COMBO')).toBe(WidgetSelect)
      })
    })

    // Test boolean type mappings
    describe('boolean types', () => {
      it('should map boolean types to toggle switch widget', () => {
        expect(getComponent('toggle')).toBe(WidgetToggleSwitch)
        expect(getComponent('boolean')).toBe(WidgetToggleSwitch)
        expect(getComponent('BOOLEAN')).toBe(WidgetToggleSwitch)
      })
    })

    // Test advanced widget mappings
    describe('advanced widgets', () => {
      it('should map color types to color picker widget', () => {
        expect(getComponent('color')).toBe(WidgetColorPicker)
        expect(getComponent('COLOR')).toBe(WidgetColorPicker)
      })

      it('should map button types to button widget', () => {
        expect(getComponent('button')).toBe(WidgetButton)
        expect(getComponent('BUTTON')).toBe(WidgetButton)
      })
    })

    // Test fallback behavior
    describe('fallback behavior', () => {
      it('should return null for unknown types', () => {
        expect(getComponent('unknown')).toBe(null)
        expect(getComponent('custom_widget')).toBe(null)
        expect(getComponent('')).toBe(null)
      })
    })
  })

  describe('shouldRenderAsVue', () => {
    it('should return false for widgets marked as canvas-only', () => {
      const widget = { type: 'text', options: { canvasOnly: true } }
      expect(shouldRenderAsVue(widget)).toBe(false)
    })

    it('should return false for widgets without a type', () => {
      const widget = {}
      expect(shouldRenderAsVue(widget)).toBe(false)
    })

    it('should return true for widgets with mapped types', () => {
      expect(shouldRenderAsVue({ type: 'text' })).toBe(true)
      expect(shouldRenderAsVue({ type: 'int' })).toBe(true)
      expect(shouldRenderAsVue({ type: 'combo' })).toBe(true)
    })

    it('should respect options while checking type', () => {
      const widget: Partial<SafeWidgetData> = {
        type: 'text',
        options: { canvasOnly: false }
      }
      expect(shouldRenderAsVue(widget)).toBe(true)
    })
  })

  describe('isEssential', () => {
    it('should identify essential widget types', () => {
      expect(isEssential('int')).toBe(true)
      expect(isEssential('INT')).toBe(true)
      expect(isEssential('float')).toBe(true)
      expect(isEssential('FLOAT')).toBe(true)
      expect(isEssential('boolean')).toBe(true)
      expect(isEssential('BOOLEAN')).toBe(true)
      expect(isEssential('combo')).toBe(true)
      expect(isEssential('COMBO')).toBe(true)
    })

    it('should identify non-essential widget types', () => {
      expect(isEssential('button')).toBe(false)
      expect(isEssential('color')).toBe(false)
      expect(isEssential('chart')).toBe(false)
      expect(isEssential('fileupload')).toBe(false)
    })

    it('should return false for unknown types', () => {
      expect(isEssential('unknown')).toBe(false)
      expect(isEssential('')).toBe(false)
    })
  })

  describe('edge cases', () => {
    it('should handle widgets with empty options', () => {
      const widget = { type: 'text', options: {} }
      expect(shouldRenderAsVue(widget)).toBe(true)
    })

    it('should handle case sensitivity correctly through aliases', () => {
      // Test that both lowercase and uppercase work
      expect(getComponent('string')).toBe(WidgetInputText)
      expect(getComponent('STRING')).toBe(WidgetInputText)
      expect(getComponent('combo')).toBe(WidgetSelect)
      expect(getComponent('COMBO')).toBe(WidgetSelect)
    })
  })
})
