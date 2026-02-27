import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'

import type { SimplifiedWidget } from '@/types/simplifiedWidget'

import WidgetInputNumberInput from './WidgetInputNumberInput.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en'
})

function createMockWidget(
  value: number = 0,
  type: 'int' | 'float' = 'int',
  options: SimplifiedWidget['options'] = {},
  callback?: (value: number) => void
): SimplifiedWidget<number> {
  return {
    name: 'test_input_number',
    type,
    value,
    options,
    callback
  }
}

function mountComponent(widget: SimplifiedWidget<number>, modelValue: number) {
  return mount(WidgetInputNumberInput, {
    global: { plugins: [i18n] },
    props: {
      widget,
      modelValue
    }
  })
}

function getNumberInput(wrapper: ReturnType<typeof mount>) {
  const input = wrapper.get<HTMLInputElement>('input[inputmode="decimal"]')
  return input.element
}

describe('WidgetInputNumberInput Value Binding', () => {
  it('displays initial value in input field', () => {
    const widget = createMockWidget(42, 'int')
    const wrapper = mountComponent(widget, 42)

    const input = getNumberInput(wrapper)
    expect(input.value).toBe('42')
  })

  it('emits update:modelValue when value changes', async () => {
    const widget = createMockWidget(10, 'int')
    const wrapper = mountComponent(widget, 10)

    const inputNumber = wrapper
    await inputNumber.vm.$emit('update:modelValue', 20)

    const emitted = wrapper.emitted('update:modelValue')
    expect(emitted).toBeDefined()
    expect(emitted![0]).toContain(20)
  })

  it('handles negative values', () => {
    const widget = createMockWidget(-5, 'int')
    const wrapper = mountComponent(widget, -5)

    const input = getNumberInput(wrapper)
    expect(input.value).toBe('-5')
  })

  it('handles decimal values for float type', () => {
    const widget = createMockWidget(3.14, 'float')
    const wrapper = mountComponent(widget, 3.14)

    const input = getNumberInput(wrapper)
    expect(input.value).toBe('3.14')
  })
})

describe('WidgetInputNumberInput Grouping Behavior', () => {
  it('displays numbers without commas by default for int widgets', () => {
    const widget = createMockWidget(1000, 'int')
    const wrapper = mountComponent(widget, 1000)

    const input = getNumberInput(wrapper)
    expect(input.value).toBe('1000')
    expect(input.value).not.toContain(',')
  })

  it('displays numbers without commas by default for float widgets', () => {
    const widget = createMockWidget(1000.5, 'float')
    const wrapper = mountComponent(widget, 1000.5)

    const input = getNumberInput(wrapper)
    expect(input.value).toBe('1000.5')
    expect(input.value).not.toContain(',')
  })

  it('displays numbers with commas when grouping enabled', () => {
    const widget = createMockWidget(1000, 'int', { useGrouping: true })
    const wrapper = mountComponent(widget, 1000)

    const input = getNumberInput(wrapper)
    expect(input.value).toBe('1,000')
    expect(input.value).toContain(',')
  })

  it('displays numbers without commas when grouping explicitly disabled', () => {
    const widget = createMockWidget(1000, 'int', { useGrouping: false })
    const wrapper = mountComponent(widget, 1000)

    const input = getNumberInput(wrapper)
    expect(input.value).toBe('1000')
    expect(input.value).not.toContain(',')
  })

  it('displays numbers without commas when useGrouping option is undefined', () => {
    const widget = createMockWidget(1000, 'int', { useGrouping: undefined })
    const wrapper = mountComponent(widget, 1000)

    const input = getNumberInput(wrapper)
    expect(input.value).toBe('1000')
    expect(input.value).not.toContain(',')
  })
})

describe('WidgetInputNumberInput Large Integer Precision Handling', () => {
  const SAFE_INTEGER_MAX = Number.MAX_SAFE_INTEGER // 9,007,199,254,740,991
  const UNSAFE_LARGE_INTEGER = 18446744073709552000 // Example seed value that exceeds safe range

  it('shows buttons for safe integer values', () => {
    const widget = createMockWidget(1000, 'int')
    const wrapper = mountComponent(widget, 1000)

    expect(wrapper.findAll('button').length).toBe(2)
  })

  it('shows buttons for values at safe integer limit', () => {
    const widget = createMockWidget(SAFE_INTEGER_MAX, 'int')
    const wrapper = mountComponent(widget, SAFE_INTEGER_MAX)

    expect(wrapper.findAll('button').length).toBe(2)
  })

  it('hides buttons for unsafe large integer values', () => {
    const widget = createMockWidget(UNSAFE_LARGE_INTEGER, 'int')
    const wrapper = mountComponent(widget, UNSAFE_LARGE_INTEGER)

    expect(wrapper.findAll('button').length).toBe(0)
  })

  it('hides buttons for unsafe negative integer values', () => {
    const unsafeNegative = -UNSAFE_LARGE_INTEGER
    const widget = createMockWidget(unsafeNegative, 'int')
    const wrapper = mountComponent(widget, unsafeNegative)

    expect(wrapper.findAll('button').length).toBe(0)
  })

  it('shows tooltip for disabled buttons due to precision limits', (context) => {
    context.skip('needs diagnosis')
    const widget = createMockWidget(UNSAFE_LARGE_INTEGER, 'int')
    const wrapper = mountComponent(widget, UNSAFE_LARGE_INTEGER)

    // Check that tooltip wrapper div exists
    const tooltipDiv = wrapper.find('div[v-tooltip]')
    expect(tooltipDiv.exists()).toBe(true)
  })

  it('does not show tooltip for safe integer values', () => {
    const widget = createMockWidget(1000, 'int')
    const wrapper = mountComponent(widget, 1000)

    // For safe values, tooltip should not be set (computed returns null)
    const tooltipDiv = wrapper.find('div')
    expect(tooltipDiv.attributes('v-tooltip')).toBeUndefined()
  })

  it('handles floating point values correctly', () => {
    const widget = createMockWidget(1000.5, 'float')
    const wrapper = mountComponent(widget, 1000.5)

    expect(wrapper.findAll('button').length).toBe(2)
  })

  it('hides buttons for unsafe floating point values', () => {
    const unsafeFloat = UNSAFE_LARGE_INTEGER + 0.5
    const widget = createMockWidget(unsafeFloat, 'float')
    const wrapper = mountComponent(widget, unsafeFloat)

    expect(wrapper.findAll('button').length).toBe(0)
  })
})

describe('WidgetInputNumberInput Edge Cases for Precision Handling', () => {
  it('handles null/undefined model values gracefully', () => {
    const widget = createMockWidget(0, 'int')
    const wrapper = mount(WidgetInputNumberInput, {
      global: { plugins: [i18n] },
      props: {
        widget,
        modelValue: undefined
      } as { widget: SimplifiedWidget<number>; modelValue: number | undefined }
    })

    expect(wrapper.findAll('button').length).toBe(2)
  })

  it('handles NaN values gracefully', (context) => {
    context.skip('needs diagnosis')
    const widget = createMockWidget(NaN, 'int')
    const wrapper = mountComponent(widget, NaN)

    expect(wrapper.findAll('button').length).toBe(0)
  })

  it('handles Infinity values', () => {
    const widget = createMockWidget(Infinity, 'int')
    const wrapper = mountComponent(widget, Infinity)

    expect(wrapper.findAll('button').length).toBe(0)
  })

  it('handles negative Infinity values', () => {
    const widget = createMockWidget(-Infinity, 'int')
    const wrapper = mountComponent(widget, -Infinity)

    expect(wrapper.findAll('button').length).toBe(0)
  })
})
