import { mount } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import ToggleSwitch from 'primevue/toggleswitch'
import { createI18n } from 'vue-i18n'
import { describe, expect, it } from 'vitest'

import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { IWidgetOptions } from '@/lib/litegraph/src/types/widgets'
import type { SimplifiedWidget } from '@/types/simplifiedWidget'

import WidgetToggleSwitch from './WidgetToggleSwitch.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      widgets: {
        boolean: {
          true: 'true',
          false: 'false'
        }
      }
    }
  }
})

describe('WidgetToggleSwitch Value Binding', () => {
  const createMockWidget = (
    value: boolean = false,
    options: IWidgetOptions = {},
    callback?: (value: boolean) => void
  ): SimplifiedWidget<boolean, IWidgetOptions> => ({
    name: 'test_toggle',
    type: 'boolean',
    value,
    options,
    callback
  })

  const mountComponent = (
    widget: SimplifiedWidget<boolean>,
    modelValue: boolean,
    readonly = false
  ) => {
    return mount(WidgetToggleSwitch, {
      props: {
        widget,
        modelValue,
        readonly
      },
      global: {
        plugins: [PrimeVue, i18n],
        components: { ToggleSwitch, ToggleGroup, ToggleGroupItem }
      }
    })
  }

  describe('Vue Event Emission', () => {
    it('emits Vue event when toggled from false to true', async () => {
      const widget = createMockWidget(false)
      const wrapper = mountComponent(widget, false)

      const toggle = wrapper.findComponent({ name: 'ToggleSwitch' })
      await toggle.setValue(true)

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain(true)
    })

    it('emits Vue event when toggled from true to false', async () => {
      const widget = createMockWidget(true)
      const wrapper = mountComponent(widget, true)

      const toggle = wrapper.findComponent({ name: 'ToggleSwitch' })
      await toggle.setValue(false)

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain(false)
    })

    it('handles value changes gracefully', async () => {
      const widget = createMockWidget(false)
      const wrapper = mountComponent(widget, false)

      const toggle = wrapper.findComponent({ name: 'ToggleSwitch' })
      await toggle.setValue(true)
      await toggle.setValue(false)

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toHaveLength(2)
      expect(emitted![0]).toContain(true)
      expect(emitted![1]).toContain(false)
    })
  })

  describe('Component Rendering', () => {
    it('renders toggle switch component', () => {
      const widget = createMockWidget(false)
      const wrapper = mountComponent(widget, false)

      expect(wrapper.findComponent({ name: 'ToggleSwitch' }).exists()).toBe(
        true
      )
    })

    it('displays correct initial state for false', () => {
      const widget = createMockWidget(false)
      const wrapper = mountComponent(widget, false)

      const toggle = wrapper.findComponent({ name: 'ToggleSwitch' })
      expect(toggle.props('modelValue')).toBe(false)
    })

    it('displays correct initial state for true', () => {
      const widget = createMockWidget(true)
      const wrapper = mountComponent(widget, true)

      const toggle = wrapper.findComponent({ name: 'ToggleSwitch' })
      expect(toggle.props('modelValue')).toBe(true)
    })
  })

  describe('Multiple Value Changes', () => {
    it('handles rapid toggling correctly', async () => {
      const widget = createMockWidget(false)
      const wrapper = mountComponent(widget, false)

      const toggle = wrapper.findComponent({ name: 'ToggleSwitch' })

      await toggle.setValue(true)
      await toggle.setValue(false)
      await toggle.setValue(true)

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toHaveLength(3)
      expect(emitted![0]).toContain(true)
      expect(emitted![1]).toContain(false)
      expect(emitted![2]).toContain(true)
    })

    it('maintains state consistency during multiple changes', async () => {
      const widget = createMockWidget(false)
      const wrapper = mountComponent(widget, false)

      const toggle = wrapper.findComponent({ name: 'ToggleSwitch' })

      await toggle.setValue(true)
      await toggle.setValue(false)
      await toggle.setValue(true)
      await toggle.setValue(false)

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toHaveLength(4)
      expect(emitted![0]).toContain(true)
      expect(emitted![1]).toContain(false)
      expect(emitted![2]).toContain(true)
      expect(emitted![3]).toContain(false)
    })
  })

  describe('Label Display (label_on/label_off)', () => {
    it('renders ToggleGroup when labels are provided', () => {
      const widget = createMockWidget(false, { on: 'inside', off: 'outside' })
      const wrapper = mountComponent(widget, false)

      expect(wrapper.findComponent({ name: 'ToggleGroupRoot' }).exists()).toBe(
        true
      )
      expect(wrapper.findComponent({ name: 'ToggleSwitch' }).exists()).toBe(
        false
      )
    })

    it('renders ToggleSwitch when no labels are provided', () => {
      const widget = createMockWidget(false, {})
      const wrapper = mountComponent(widget, false)

      expect(wrapper.findComponent({ name: 'ToggleSwitch' }).exists()).toBe(
        true
      )
      expect(wrapper.findComponent({ name: 'ToggleGroupRoot' }).exists()).toBe(
        false
      )
    })

    it('displays both on and off labels in ToggleGroup', () => {
      const widget = createMockWidget(false, { on: 'inside', off: 'outside' })
      const wrapper = mountComponent(widget, false)

      expect(wrapper.text()).toContain('inside')
      expect(wrapper.text()).toContain('outside')
    })

    it('selects correct option based on boolean value (false)', () => {
      const widget = createMockWidget(false, { on: 'enabled', off: 'disabled' })
      const wrapper = mountComponent(widget, false)

      const toggleGroup = wrapper.findComponent({ name: 'ToggleGroupRoot' })
      expect(toggleGroup.props('modelValue')).toBe('off')
    })

    it('selects correct option based on boolean value (true)', () => {
      const widget = createMockWidget(true, { on: 'enabled', off: 'disabled' })
      const wrapper = mountComponent(widget, true)

      const toggleGroup = wrapper.findComponent({ name: 'ToggleGroupRoot' })
      expect(toggleGroup.props('modelValue')).toBe('on')
    })

    it('emits true when "on" option is clicked', async () => {
      const widget = createMockWidget(false, { on: 'enabled', off: 'disabled' })
      const wrapper = mountComponent(widget, false)

      const buttons = wrapper.findAll('button')
      const onButton = buttons.find((b) => b.text() === 'enabled')
      await onButton!.trigger('click')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain(true)
    })

    it('emits false when "off" option is clicked', async () => {
      const widget = createMockWidget(true, { on: 'enabled', off: 'disabled' })
      const wrapper = mountComponent(widget, true)

      const buttons = wrapper.findAll('button')
      const offButton = buttons.find((b) => b.text() === 'disabled')
      await offButton!.trigger('click')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain(false)
    })

    it('falls back to i18n defaults when only partial options provided', () => {
      const widgetOnOnly = createMockWidget(true, { on: 'active' })
      const wrapperOn = mountComponent(widgetOnOnly, true)
      expect(wrapperOn.text()).toContain('active')
      expect(wrapperOn.text()).toContain('false')

      const widgetOffOnly = createMockWidget(false, { off: 'inactive' })
      const wrapperOff = mountComponent(widgetOffOnly, false)
      expect(wrapperOff.text()).toContain('inactive')
      expect(wrapperOff.text()).toContain('true')
    })

    it('treats empty string labels as explicit values', () => {
      const widget = createMockWidget(false, { on: '', off: 'disabled' })
      const wrapper = mountComponent(widget, false)

      expect(wrapper.findComponent({ name: 'ToggleGroupRoot' }).exists()).toBe(
        true
      )
      expect(wrapper.findComponent({ name: 'ToggleSwitch' }).exists()).toBe(
        false
      )
      expect(wrapper.text()).not.toContain('true')
    })

    it('disables ToggleGroup when read_only option is set', () => {
      const widget = createMockWidget(false, {
        on: 'yes',
        off: 'no',
        read_only: true
      })
      const wrapper = mountComponent(widget, false)

      const toggleGroup = wrapper.findComponent({ name: 'ToggleGroupRoot' })
      expect(toggleGroup.props('disabled')).toBe(true)
    })

    it('does not emit when clicking already-selected option', async () => {
      const widget = createMockWidget(false, { on: 'yes', off: 'no' })
      const wrapper = mountComponent(widget, false)

      const buttons = wrapper.findAll('button')
      const offButton = buttons.find((b) => b.text() === 'no')
      await offButton!.trigger('click')

      expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    })
  })
})
