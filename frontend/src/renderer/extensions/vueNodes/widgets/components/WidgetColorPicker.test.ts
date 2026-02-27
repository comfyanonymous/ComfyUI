import { mount } from '@vue/test-utils'
import ColorPicker from 'primevue/colorpicker'
import type { ColorPickerProps } from 'primevue/colorpicker'
import PrimeVue from 'primevue/config'
import { describe, expect, it } from 'vitest'

import type { SimplifiedWidget } from '@/types/simplifiedWidget'

import WidgetColorPicker from './WidgetColorPicker.vue'
import WidgetLayoutField from './layout/WidgetLayoutField.vue'

describe('WidgetColorPicker Value Binding', () => {
  const createMockWidget = (
    value: string = '#000000',
    options: Partial<ColorPickerProps> = {},
    callback?: (value: string) => void
  ): SimplifiedWidget<string> => ({
    name: 'test_color_picker',
    type: 'color',
    value,
    options,
    callback
  })

  const mountComponent = (
    widget: SimplifiedWidget<string>,
    modelValue: string,
    readonly = false
  ) => {
    return mount(WidgetColorPicker, {
      global: {
        plugins: [PrimeVue],
        components: {
          ColorPicker,
          WidgetLayoutField
        }
      },
      props: {
        widget,
        modelValue,
        readonly
      }
    })
  }

  const setColorPickerValue = async (
    wrapper: ReturnType<typeof mount>,
    value: unknown
  ) => {
    const colorPicker = wrapper.findComponent({ name: 'ColorPicker' })
    await colorPicker.setValue(value)
    return wrapper.emitted('update:modelValue')
  }

  describe('Vue Event Emission', () => {
    it('emits Vue event when color changes', async () => {
      const widget = createMockWidget('#ff0000')
      const wrapper = mountComponent(widget, '#ff0000')

      const emitted = await setColorPickerValue(wrapper, '#00ff00')

      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('#00ff00')
    })

    it('handles different color formats', async () => {
      const widget = createMockWidget('#ffffff')
      const wrapper = mountComponent(widget, '#ffffff')

      const emitted = await setColorPickerValue(wrapper, '#123abc')

      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('#123abc')
    })

    it('handles missing callback gracefully', async () => {
      const widget = createMockWidget('#000000', {}, undefined)
      const wrapper = mountComponent(widget, '#000000')

      const emitted = await setColorPickerValue(wrapper, '#ff00ff')

      // Should still emit Vue event
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('#ff00ff')
    })

    it('normalizes bare hex without # to #hex on emit', async () => {
      const widget = createMockWidget('ff0000')
      const wrapper = mountComponent(widget, 'ff0000')

      const emitted = await setColorPickerValue(wrapper, '00ff00')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('#00ff00')
    })

    it('normalizes rgb() strings to #hex on emit', async (context) => {
      context.skip('needs diagnosis')
      const widget = createMockWidget('#000000')
      const wrapper = mountComponent(widget, '#000000')

      const emitted = await setColorPickerValue(wrapper, 'rgb(255, 0, 0)')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('#ff0000')
    })

    it('normalizes hsb() strings to #hex on emit', async () => {
      const widget = createMockWidget('#000000', { format: 'hsb' })
      const wrapper = mountComponent(widget, '#000000')

      const emitted = await setColorPickerValue(wrapper, 'hsb(120, 100, 100)')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('#00ff00')
    })

    it('normalizes HSB object values to #hex on emit', async () => {
      const widget = createMockWidget('#000000', { format: 'hsb' })
      const wrapper = mountComponent(widget, '#000000')

      const emitted = await setColorPickerValue(wrapper, {
        h: 240,
        s: 100,
        b: 100
      })
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('#0000ff')
    })
  })

  describe('Component Rendering', () => {
    it('renders color picker component', () => {
      const widget = createMockWidget('#ff0000')
      const wrapper = mountComponent(widget, '#ff0000')

      const colorPicker = wrapper.findComponent({ name: 'ColorPicker' })
      expect(colorPicker.exists()).toBe(true)
    })

    it('normalizes display to a single leading #', () => {
      // Case 1: model value already includes '#'
      let widget = createMockWidget('#ff0000')
      let wrapper = mountComponent(widget, '#ff0000')
      let colorText = wrapper.find('[data-testid="widget-color-text"]')
      expect.soft(colorText.text()).toBe('#ff0000')

      // Case 2: model value missing '#'
      widget = createMockWidget('ff0000')
      wrapper = mountComponent(widget, 'ff0000')
      colorText = wrapper.find('[data-testid="widget-color-text"]')
      expect.soft(colorText.text()).toBe('#ff0000')
    })

    it('renders layout field wrapper', () => {
      const widget = createMockWidget('#ff0000')
      const wrapper = mountComponent(widget, '#ff0000')

      const layoutField = wrapper.findComponent({ name: 'WidgetLayoutField' })
      expect(layoutField.exists()).toBe(true)
    })

    it('displays current color value as text', () => {
      const widget = createMockWidget('#ff0000')
      const wrapper = mountComponent(widget, '#ff0000')

      const colorText = wrapper.find('[data-testid="widget-color-text"]')
      expect(colorText.text()).toBe('#ff0000')
    })

    it('updates color text when value changes', async () => {
      const widget = createMockWidget('#ff0000')
      const wrapper = mountComponent(widget, '#ff0000')

      await setColorPickerValue(wrapper, '#00ff00')

      // Need to check the local state update
      const colorText = wrapper.find('[data-testid="widget-color-text"]')
      // Be specific about the displayed value including the leading '#'
      expect.soft(colorText.text()).toBe('#00ff00')
    })

    it('uses default color when no value provided', () => {
      const widget = createMockWidget('')
      const wrapper = mountComponent(widget, '')

      const colorPicker = wrapper.findComponent({ name: 'ColorPicker' })
      // Should use the default value from the composable
      expect(colorPicker.exists()).toBe(true)
    })
  })

  describe('Color Formats', () => {
    it('handles valid hex colors', async () => {
      const validHexColors = [
        '#000000',
        '#ffffff',
        '#ff0000',
        '#00ff00',
        '#0000ff',
        '#123abc'
      ]

      for (const color of validHexColors) {
        const widget = createMockWidget(color)
        const wrapper = mountComponent(widget, color)

        const colorText = wrapper.find('[data-testid="widget-color-text"]')
        expect.soft(colorText.text()).toBe(color)
      }
    })

    it('handles short hex colors', () => {
      const widget = createMockWidget('#fff')
      const wrapper = mountComponent(widget, '#fff')

      const colorText = wrapper.find('[data-testid="widget-color-text"]')
      expect(colorText.text()).toBe('#fff')
    })

    it('passes widget options to color picker', () => {
      const colorOptions = {
        format: 'hex' as const,
        inline: true
      }
      const widget = createMockWidget('#ff0000', colorOptions)
      const wrapper = mountComponent(widget, '#ff0000')

      const colorPicker = wrapper.findComponent({ name: 'ColorPicker' })
      expect(colorPicker.props('format')).toBe('hex')
      expect(colorPicker.props('inline')).toBe(true)
    })
  })

  describe('Widget Layout Integration', () => {
    it('passes widget to layout field', () => {
      const widget = createMockWidget('#ff0000')
      const wrapper = mountComponent(widget, '#ff0000')

      const layoutField = wrapper.findComponent({ name: 'WidgetLayoutField' })
      expect(layoutField.props('widget')).toEqual(widget)
    })

    it('maintains proper component structure', () => {
      const widget = createMockWidget('#ff0000')
      const wrapper = mountComponent(widget, '#ff0000')

      // Should have layout field containing label with color picker and text
      const layoutField = wrapper.findComponent({ name: 'WidgetLayoutField' })
      const label = wrapper.find('label')
      const colorPicker = wrapper.findComponent({ name: 'ColorPicker' })
      const colorText = wrapper.find('span')

      expect(layoutField.exists()).toBe(true)
      expect(label.exists()).toBe(true)
      expect(colorPicker.exists()).toBe(true)
      expect(colorText.exists()).toBe(true)
    })
  })

  describe('Edge Cases', () => {
    it('handles empty color value', () => {
      const widget = createMockWidget('')
      const wrapper = mountComponent(widget, '')

      const colorPicker = wrapper.findComponent({ name: 'ColorPicker' })
      expect(colorPicker.exists()).toBe(true)
    })

    it('handles invalid color formats gracefully', async () => {
      const widget = createMockWidget('invalid-color')
      const wrapper = mountComponent(widget, 'invalid-color')

      const colorText = wrapper.find('[data-testid="widget-color-text"]')
      expect(colorText.text()).toBe('#000000')

      const emitted = await setColorPickerValue(wrapper, 'invalid-color')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('#000000')
    })

    it('handles widget with no options', () => {
      const widget = createMockWidget('#ff0000')
      const wrapper = mountComponent(widget, '#ff0000')

      const colorPicker = wrapper.findComponent({ name: 'ColorPicker' })
      expect(colorPicker.exists()).toBe(true)
    })
  })
})
