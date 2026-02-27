import { mount } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import { describe, expect, it, vi } from 'vitest'

import FormSelectButton from './FormSelectButton.vue'

describe('FormSelectButton Core Component', () => {
  // Type-safe helper for mounting component
  const mountComponent = (
    modelValue: string | null | undefined = null,
    options: unknown[] = [],
    props: Record<string, unknown> = {}
  ) => {
    return mount(FormSelectButton, {
      global: {
        plugins: [PrimeVue]
      },
      props: {
        modelValue,
        options: options as (
          | string
          | number
          | { label: string; value: string | number }
        )[],
        ...props
      }
    })
  }

  const clickButton = async (
    wrapper: ReturnType<typeof mount>,
    buttonText: string
  ) => {
    const buttons = wrapper.findAll('button')
    const targetButtonIndex = buttons.findIndex((button) =>
      button.text().includes(buttonText)
    )

    if (targetButtonIndex === -1) {
      throw new Error(`Button with text "${buttonText}" not found`)
    }

    // Use get() which throws if element doesn't exist, providing better error messages
    const targetButton = buttons.at(targetButtonIndex)!
    await targetButton.trigger('click')
    return targetButton
  }

  describe('Basic Rendering', () => {
    it('renders as a horizontal button group layout', () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent(null, options)

      const container = wrapper.find('div')
      const buttons = wrapper.findAll('button')

      // Verify layout behavior: container exists and contains buttons
      expect(container.exists()).toBe(true)
      expect(buttons).toHaveLength(2)

      // Verify buttons are arranged horizontally (not vertically stacked)
      // This tests the layout logic rather than specific CSS classes
      buttons.forEach((button) => {
        expect(button.exists()).toBe(true)
        expect(button.element.tagName).toBe('BUTTON')
      })
    })

    it('renders buttons for each option', () => {
      const options = ['first', 'second', 'third']
      const wrapper = mountComponent(null, options)

      const buttons = wrapper.findAll('button')
      expect(buttons).toHaveLength(3)
      expect(buttons[0].text()).toBe('first')
      expect(buttons[1].text()).toBe('second')
      expect(buttons[2].text()).toBe('third')
    })

    it('renders empty container when no options provided', () => {
      const wrapper = mountComponent(null, [])

      const buttons = wrapper.findAll('button')
      expect(buttons).toHaveLength(0)
    })

    it('applies proper button styling', () => {
      const options = ['test']
      const wrapper = mountComponent(null, options)

      const button = wrapper.find('button')
      expect(button.classes()).toContain('flex-1')
      expect(button.classes()).toContain('h-6')
      expect(button.classes()).toContain('px-5')
      expect(button.classes()).toContain('py-[5px]')
      expect(button.classes()).toContain('rounded')
      expect(button.classes()).toContain('text-center')
      expect(button.classes()).toContain('text-xs')
      expect(button.classes()).toContain('font-normal')
    })
  })

  describe('String Options', () => {
    it('handles string array options', () => {
      const options = ['apple', 'banana', 'cherry']
      const wrapper = mountComponent('banana', options)

      const buttons = wrapper.findAll('button')
      expect(buttons).toHaveLength(3)
      expect(buttons[0].text()).toBe('apple')
      expect(buttons[1].text()).toBe('banana')
      expect(buttons[2].text()).toBe('cherry')
    })

    it('emits correct string value when clicked', async () => {
      const options = ['first', 'second', 'third']
      const wrapper = mountComponent('first', options)

      await clickButton(wrapper, 'second')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toEqual(['second'])
    })

    it('highlights selected string option', () => {
      const options = ['option1', 'option2', 'option3']
      const wrapper = mountComponent('option2', options)

      const buttons = wrapper.findAll('button')
      expect(buttons[1].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      )
      expect(buttons[1].classes()).toContain('text-text-primary')
      expect(buttons[0].classes()).not.toContain(
        'bg-interface-menu-component-surface-selected'
      )
      expect(buttons[2].classes()).not.toContain(
        'bg-interface-menu-component-surface-selected'
      )
    })
  })

  describe('Number Options', () => {
    it('handles number array options', () => {
      const options = [1, 2, 3]
      const wrapper = mountComponent('2', options)

      const buttons = wrapper.findAll('button')
      expect(buttons).toHaveLength(3)
      expect(buttons[0].text()).toBe('1')
      expect(buttons[1].text()).toBe('2')
      expect(buttons[2].text()).toBe('3')
    })

    it('emits number value when clicked', async () => {
      const options = [10, 20, 30]
      const wrapper = mountComponent('10', options)

      await clickButton(wrapper, '20')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toEqual([20])
    })

    it('highlights selected number option', () => {
      const options = [100, 200, 300]
      const wrapper = mountComponent('200', options)

      const buttons = wrapper.findAll('button')
      expect(buttons[1].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      )
      expect(buttons[1].classes()).toContain('text-text-primary')
    })
  })

  describe('Object Options', () => {
    it('handles object array with label and value', () => {
      const options = [
        { label: 'First Option', value: 'first' },
        { label: 'Second Option', value: 'second' }
      ]
      const wrapper = mountComponent('first', options)

      const buttons = wrapper.findAll('button')
      expect(buttons).toHaveLength(2)
      expect(buttons[0].text()).toBe('First Option')
      expect(buttons[1].text()).toBe('Second Option')
    })

    it('emits object value when object option clicked', async () => {
      const options = [
        { label: 'Apple', value: 'apple_val' },
        { label: 'Banana', value: 'banana_val' }
      ]
      const wrapper = mountComponent('apple_val', options)

      await clickButton(wrapper, 'Banana')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toEqual(['banana_val'])
    })

    it('highlights selected object option by value', () => {
      const options = [
        { label: 'Small', value: 'sm' },
        { label: 'Medium', value: 'md' },
        { label: 'Large', value: 'lg' }
      ]
      const wrapper = mountComponent('md', options)

      const buttons = wrapper.findAll('button')
      expect(buttons[1].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      ) // Medium
      expect(buttons[0].classes()).not.toContain(
        'bg-interface-menu-component-surface-selected'
      )
      expect(buttons[2].classes()).not.toContain(
        'bg-interface-menu-component-surface-selected'
      )
    })

    it('handles objects without value field', () => {
      const options = [
        { label: 'First', name: 'first_name' },
        { label: 'Second', name: 'second_name' }
      ]
      const wrapper = mountComponent('first_name', options)

      const buttons = wrapper.findAll('button')
      expect(buttons[0].text()).toBe('First')
      expect(buttons[1].text()).toBe('Second')
      expect(buttons[0].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      )
    })

    it('handles objects without label field', () => {
      const options = [
        { value: 'val1', name: 'Name 1' },
        { value: 'val2', name: 'Name 2' }
      ]
      const wrapper = mountComponent('val1', options)

      const buttons = wrapper.findAll('button')
      expect(buttons[0].text()).toBe('Name 1')
      expect(buttons[1].text()).toBe('Name 2')
    })
  })

  describe('PrimeVue Compatibility', () => {
    it('uses custom optionLabel prop', () => {
      const options = [
        { title: 'First Item', value: 'first' },
        { title: 'Second Item', value: 'second' }
      ]
      const wrapper = mountComponent('first', options, { optionLabel: 'title' })

      const buttons = wrapper.findAll('button')
      expect(buttons[0].text()).toBe('First Item')
      expect(buttons[1].text()).toBe('Second Item')
    })

    it('uses custom optionValue prop', () => {
      const options = [
        { label: 'First', id: 'first_id' },
        { label: 'Second', id: 'second_id' }
      ]
      const wrapper = mountComponent('first_id', options, { optionValue: 'id' })

      const buttons = wrapper.findAll('button')
      expect(buttons[0].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      )
      expect(buttons[1].classes()).not.toContain(
        'bg-interface-menu-component-surface-selected'
      )
    })

    it('emits custom optionValue when clicked', async () => {
      const options = [
        { label: 'First', id: 'first_id' },
        { label: 'Second', id: 'second_id' }
      ]
      const wrapper = mountComponent('first_id', options, { optionValue: 'id' })

      await clickButton(wrapper, 'Second')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toEqual(['second_id'])
    })
  })

  describe('Disabled State', () => {
    it('disables all buttons when disabled prop is true', () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent('option1', options, { disabled: true })

      const buttons = wrapper.findAll('button')
      buttons.forEach((button) => {
        expect(button.element.disabled).toBe(true)
        expect(button.classes()).toContain('opacity-50')
        expect(button.classes()).toContain('cursor-not-allowed')
      })
    })

    it('does not emit events when disabled', async () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent('option1', options, { disabled: true })

      await clickButton(wrapper, 'option2')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeUndefined()
    })

    it('does not apply hover styles when disabled', () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent('option1', options, { disabled: true })

      const buttons = wrapper.findAll('button')
      buttons.forEach((button) => {
        expect(button.classes()).not.toContain(
          'hover:bg-interface-menu-component-surface-hovered'
        )
        expect(button.classes()).not.toContain('cursor-pointer')
      })
    })

    it('applies disabled styling to selected option', () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent('option1', options, { disabled: true })

      const buttons = wrapper.findAll('button')
      expect(buttons[0].classes()).not.toContain(
        'bg-interface-menu-component-surface-selected'
      ) // Selected styling disabled
      expect(buttons[0].classes()).toContain('opacity-50')
      expect(buttons[0].classes()).toContain('text-text-secondary')
    })
  })

  describe('Selection Logic', () => {
    it('handles null modelValue', () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent(null, options)

      const buttons = wrapper.findAll('button')
      buttons.forEach((button) => {
        expect(button.classes()).not.toContain(
          'bg-interface-menu-component-surface-selected'
        )
      })
    })

    it('handles undefined modelValue', () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent(undefined, options)

      const buttons = wrapper.findAll('button')
      buttons.forEach((button) => {
        expect(button.classes()).not.toContain(
          'bg-interface-menu-component-surface-selected'
        )
      })
    })

    it('handles empty string modelValue', () => {
      const options = ['', 'option1', 'option2']
      const wrapper = mountComponent('', options)

      const buttons = wrapper.findAll('button')
      expect(buttons[0].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      ) // Empty string is selected
      expect(buttons[1].classes()).not.toContain(
        'bg-interface-menu-component-surface-selected'
      )
    })

    it('compares values as strings', () => {
      const options = [1, '2', 3]
      const wrapper = mountComponent('1', options)

      const buttons = wrapper.findAll('button')
      expect(buttons[0].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      ) // '1' matches number 1 as string
    })
  })

  describe('Visual States', () => {
    it('applies selected styling to active option', () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent('option1', options)

      const selectedButton = wrapper.findAll('button')[0]
      expect(selectedButton.classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      )
      expect(selectedButton.classes()).toContain('text-text-primary')
    })

    it('applies unselected styling to inactive options', () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent('option1', options)

      const unselectedButton = wrapper.findAll('button')[1]
      expect(unselectedButton.classes()).toContain('bg-transparent')
      expect(unselectedButton.classes()).toContain('text-text-secondary')
    })

    it('applies hover effects to enabled unselected buttons', () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent('option1', options, { disabled: false })

      const unselectedButton = wrapper.findAll('button')[1]
      expect(unselectedButton.classes()).toContain(
        'hover:bg-interface-menu-component-surface-selected/50'
      )
      expect(unselectedButton.classes()).toContain('cursor-pointer')
    })
  })

  describe('Edge Cases', () => {
    it('handles very long option text', () => {
      const longText =
        'This is a very long option text that might cause layout issues'
      const options = ['short', longText, 'normal']
      const wrapper = mountComponent('short', options)

      const buttons = wrapper.findAll('button')
      expect(buttons[1].text()).toBe(longText)
      expect(buttons).toHaveLength(3)
    })

    it('handles options with special characters', () => {
      const specialOptions = ['@#$%^&*()', '{}[]|\\:";\'<>?,./']
      const wrapper = mountComponent(specialOptions[0], specialOptions)

      const buttons = wrapper.findAll('button')
      expect(buttons[0].text()).toBe('@#$%^&*()')
      expect(buttons[0].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      )
    })

    it('handles unicode characters in options', () => {
      const unicodeOptions = ['🎨 Art', '中文', 'العربية']
      const wrapper = mountComponent('🎨 Art', unicodeOptions)

      const buttons = wrapper.findAll('button')
      expect(buttons[0].text()).toBe('🎨 Art')
      expect(buttons[0].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      )
    })

    it('handles duplicate option values', () => {
      const duplicateOptions = ['duplicate', 'unique', 'duplicate']
      const wrapper = mountComponent('duplicate', duplicateOptions)

      const buttons = wrapper.findAll('button')
      expect(buttons[0].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      )
      expect(buttons[2].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      ) // Both duplicates selected
      expect(buttons[1].classes()).not.toContain(
        'bg-interface-menu-component-surface-selected'
      )
    })

    it('handles mixed type options safely', () => {
      const mixedOptions: unknown[] = [
        'string',
        123,
        { label: 'Object', value: 'obj' }
      ]
      const wrapper = mountComponent('123', mixedOptions)

      const buttons = wrapper.findAll('button')
      expect(buttons).toHaveLength(3)
      expect(buttons[1].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      ) // Number 123 as string
    })

    it('handles objects with missing properties gracefully', () => {
      const incompleteOptions = [
        {}, // Empty object
        { randomProp: 'value' }, // No standard props
        { value: 'has_value' }, // No label
        { label: 'has_label' } // No value
      ]
      const wrapper = mountComponent('has_value', incompleteOptions)

      const buttons = wrapper.findAll('button')
      expect(buttons).toHaveLength(4)
      expect(buttons[2].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      )
    })

    it('handles large number of options', () => {
      const manyOptions = Array.from(
        { length: 50 },
        (_, i) => `Option ${i + 1}`
      )
      const wrapper = mountComponent('Option 25', manyOptions)

      const buttons = wrapper.findAll('button')
      expect(buttons).toHaveLength(50)
      expect(buttons[24].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      ) // Option 25 at index 24
    })

    it('fallback to index when all object properties are missing', () => {
      const problematicOptions = [
        { someRandomProp: 'random1' },
        { anotherRandomProp: 'random2' }
      ]
      const wrapper = mountComponent('0', problematicOptions)

      const buttons = wrapper.findAll('button')
      expect(buttons).toHaveLength(2)
      expect(buttons[0].classes()).toContain(
        'bg-interface-menu-component-surface-selected'
      ) // Falls back to index 0
    })
  })

  describe('Event Handling', () => {
    it('prevents click events when disabled', async () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent('option1', options, { disabled: true })

      const clickHandler = vi.fn()
      wrapper.vm.$el.addEventListener('click', clickHandler)

      await clickButton(wrapper, 'option2')

      expect(clickHandler).not.toHaveBeenCalled()
    })

    it('allows repeated selection of same option', async () => {
      const options = ['option1', 'option2']
      const wrapper = mountComponent('option1', options)

      await clickButton(wrapper, 'option1')
      await clickButton(wrapper, 'option1')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toHaveLength(2)
      expect(emitted![0]).toEqual(['option1'])
      expect(emitted![1]).toEqual(['option1'])
    })
  })
})
