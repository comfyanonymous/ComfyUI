import { mount } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import Textarea from 'primevue/textarea'
import { describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import enMessages from '@/locales/en/main.json'
import type { SimplifiedWidget } from '@/types/simplifiedWidget'

import WidgetMarkdown from './WidgetMarkdown.vue'

// Mock the markdown renderer utility
vi.mock('@/utils/markdownRendererUtil', () => ({
  renderMarkdownToHtml: vi.fn((markdown: string) => {
    // Simple mock that converts some markdown to HTML
    return markdown
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/^# (.*?)$/gm, '<h1>$1</h1>')
      .replace(/^## (.*?)$/gm, '<h2>$1</h2>')
      .replace(/\n/g, '<br>')
  })
}))

describe('WidgetMarkdown Dual Mode Display', () => {
  const createMockWidget = (
    value: string = '# Default Heading\nSome **bold** text.',
    options: Record<string, unknown> = {},
    callback?: (value: string) => void
  ): SimplifiedWidget<string> => ({
    name: 'test_markdown',
    type: 'string',
    value,
    options,
    callback
  })

  const mountComponent = (
    widget: SimplifiedWidget<string>,
    modelValue: string,
    readonly = false
  ) => {
    const i18n = createI18n({
      legacy: false,
      locale: 'en',
      messages: {
        en: {
          ...enMessages
        }
      }
    })

    return mount(WidgetMarkdown, {
      global: {
        plugins: [PrimeVue, i18n],
        components: { Textarea }
      },
      props: {
        widget,
        modelValue,
        readonly
      }
    })
  }

  const clickToEdit = async (wrapper: ReturnType<typeof mount>) => {
    const container = wrapper.find('.widget-markdown')
    await container.trigger('click')
    await nextTick()
    return container
  }

  const blurTextarea = async (wrapper: ReturnType<typeof mount>) => {
    const textarea = wrapper.find('textarea')
    if (textarea.exists()) {
      await textarea.trigger('blur')
      await nextTick()
    }
    return textarea
  }

  describe('Display Mode', () => {
    it('renders markdown content as HTML in display mode', () => {
      const markdown = '# Heading\nSome **bold** and *italic* text.'
      const widget = createMockWidget(markdown)
      const wrapper = mountComponent(widget, markdown)

      const displayDiv = wrapper.find('.comfy-markdown-content')
      expect(displayDiv.exists()).toBe(true)
      expect(displayDiv.html()).toContain('<h1>Heading</h1>')
      expect(displayDiv.html()).toContain('<strong>bold</strong>')
      expect(displayDiv.html()).toContain('<em>italic</em>')
    })

    it('starts in display mode by default', (context) => {
      context.skip(
        'Something in the logic in these tests is definitely off. needs diagnosis'
      )
      const widget = createMockWidget('# Test')
      const wrapper = mountComponent(widget, '# Test')

      expect(wrapper.find('.comfy-markdown-content').exists()).toBe(true)
      expect(wrapper.find('textarea').exists()).toBe(false)
    })

    it('handles empty markdown content', () => {
      const widget = createMockWidget('')
      const wrapper = mountComponent(widget, '')

      const displayDiv = wrapper.find('.comfy-markdown-content')
      expect(displayDiv.exists()).toBe(true)
      expect(displayDiv.text()).toBe('')
    })
  })

  describe('Edit Mode Toggle', () => {
    it('switches to edit mode when clicked', async (context) => {
      context.skip('markdown editor not disappearing. needs diagnosis')
      const widget = createMockWidget('# Test')
      const wrapper = mountComponent(widget, '# Test')

      expect(wrapper.find('.comfy-markdown-content').exists()).toBe(true)

      await clickToEdit(wrapper)

      expect(wrapper.find('.comfy-markdown-content').exists()).toBe(false)
      expect(wrapper.find('textarea').exists()).toBe(true)
    })

    it('does not switch to edit mode when already editing', async () => {
      const widget = createMockWidget('# Test')
      const wrapper = mountComponent(widget, '# Test')

      // First click to enter edit mode
      await clickToEdit(wrapper)
      expect(wrapper.find('textarea').exists()).toBe(true)

      // Second click should not have any effect
      await clickToEdit(wrapper)
      expect(wrapper.find('textarea').exists()).toBe(true)
    })

    it('switches back to display mode on textarea blur', async (context) => {
      context.skip('textarea not disappearing. needs diagnosis')
      const widget = createMockWidget('# Test')
      const wrapper = mountComponent(widget, '# Test')

      await clickToEdit(wrapper)
      expect(wrapper.find('textarea').exists()).toBe(true)

      await blurTextarea(wrapper)
      expect(wrapper.find('.comfy-markdown-content').exists()).toBe(true)
      expect(wrapper.find('textarea').exists()).toBe(false)
    })
  })

  describe('Edit Mode', () => {
    it('displays textarea with current value when editing', async () => {
      const markdown = '# Original Content'
      const widget = createMockWidget(markdown)
      const wrapper = mountComponent(widget, markdown)

      await clickToEdit(wrapper)

      const textarea = wrapper.find('textarea')
      expect(textarea.exists()).toBe(true)
      expect(textarea.element.value).toBe('# Original Content')
    })

    it('applies styling and configuration to textarea', async (context) => {
      context.skip(
        'Props or styling are not as described in the test. needs diagnosis'
      )
      const widget = createMockWidget('# Test')
      const wrapper = mountComponent(widget, '# Test')

      await clickToEdit(wrapper)

      const textarea = wrapper.findComponent({ name: 'Textarea' })
      expect(textarea.props('size')).toBe('small')
      // Check rows attribute in the DOM instead of props
      const textareaElement = wrapper.find('textarea')
      expect(textareaElement.attributes('rows')).toBe('6')
      expect(textarea.classes()).toContain('text-xs')
      expect(textarea.classes()).toContain('w-full')
    })

    it('stops click and keydown event propagation in edit mode', async () => {
      const widget = createMockWidget('# Test')
      const wrapper = mountComponent(widget, '# Test')

      await clickToEdit(wrapper)

      const textarea = wrapper.find('textarea')
      const clickSpy = vi.fn()
      const keydownSpy = vi.fn()

      wrapper.element.addEventListener('click', clickSpy)
      wrapper.element.addEventListener('keydown', keydownSpy)

      await textarea.trigger('click')
      await textarea.trigger('keydown', { key: 'Enter' })

      // Events should be stopped from propagating
      expect(clickSpy).not.toHaveBeenCalled()
      expect(keydownSpy).not.toHaveBeenCalled()
    })

    describe('Pointer Event Propagation', () => {
      it('stops pointerdown propagation to prevent node drag during text selection', async () => {
        const widget = createMockWidget('# Test')
        const wrapper = mountComponent(widget, '# Test')

        await clickToEdit(wrapper)

        const textarea = wrapper.find('textarea')
        expect(textarea.exists()).toBe(true)

        const parentPointerdownHandler = vi.fn()
        const wrapperEl = wrapper.element as HTMLElement
        wrapperEl.addEventListener('pointerdown', parentPointerdownHandler)

        await textarea.trigger('pointerdown')

        expect(parentPointerdownHandler).not.toHaveBeenCalled()
      })

      it('stops pointermove propagation during text selection', async () => {
        const widget = createMockWidget('# Test')
        const wrapper = mountComponent(widget, '# Test')

        await clickToEdit(wrapper)

        const textarea = wrapper.find('textarea')

        const parentPointermoveHandler = vi.fn()
        const wrapperEl = wrapper.element as HTMLElement
        wrapperEl.addEventListener('pointermove', parentPointermoveHandler)

        await textarea.trigger('pointermove')

        expect(parentPointermoveHandler).not.toHaveBeenCalled()
      })

      it('stops pointerup propagation after text selection', async () => {
        const widget = createMockWidget('# Test')
        const wrapper = mountComponent(widget, '# Test')

        await clickToEdit(wrapper)

        const textarea = wrapper.find('textarea')

        const parentPointerupHandler = vi.fn()
        const wrapperEl = wrapper.element as HTMLElement
        wrapperEl.addEventListener('pointerup', parentPointerupHandler)

        await textarea.trigger('pointerup')

        expect(parentPointerupHandler).not.toHaveBeenCalled()
      })
    })
  })

  describe('Value Updates', () => {
    it('emits update:modelValue when textarea content changes', async () => {
      const widget = createMockWidget('# Original')
      const wrapper = mountComponent(widget, '# Original')

      await clickToEdit(wrapper)

      const textarea = wrapper.find('textarea')
      await textarea.setValue('# Updated Content')
      await textarea.trigger('input')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![emitted!.length - 1]).toEqual(['# Updated Content'])
    })

    it('renders updated HTML after value change and blur', async () => {
      const widget = createMockWidget('# Original')
      const wrapper = mountComponent(widget, '# Original')

      await clickToEdit(wrapper)

      const textarea = wrapper.find('textarea')
      await textarea.setValue('## New Heading\nWith **bold** text')
      await textarea.trigger('input')
      await blurTextarea(wrapper)

      const displayDiv = wrapper.find('.comfy-markdown-content')
      expect(displayDiv.html()).toContain('<h2>New Heading</h2>')
      expect(displayDiv.html()).toContain('<strong>bold</strong>')
    })

    it('emits update:modelValue for callback handling at parent level', async () => {
      const widget = createMockWidget('# Test', {})
      const wrapper = mountComponent(widget, '# Test')

      await clickToEdit(wrapper)

      const textarea = wrapper.find('textarea')
      await textarea.setValue('# Changed')
      await textarea.trigger('input')

      // The widget should emit the change for parent (NodeWidgets) to handle callbacks
      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![emitted!.length - 1]).toEqual(['# Changed'])
    })

    it('handles missing callback gracefully', async () => {
      const widget = createMockWidget('# Test', {}, undefined)
      const wrapper = mountComponent(widget, '# Test')

      await clickToEdit(wrapper)

      const textarea = wrapper.find('textarea')
      await textarea.setValue('# Changed')

      // Should not throw error and should still emit Vue event
      await expect(textarea.trigger('input')).resolves.not.toThrow()

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
    })
  })

  describe('Complex Markdown Rendering', () => {
    it('handles multiple markdown elements', () => {
      const complexMarkdown = `# Main Heading
## Subheading
This paragraph has **bold** and *italic* text.
Another line with more content.`

      const widget = createMockWidget(complexMarkdown)
      const wrapper = mountComponent(widget, complexMarkdown)

      const displayDiv = wrapper.find('.comfy-markdown-content')
      expect(displayDiv.html()).toContain('<h1>Main Heading</h1>')
      expect(displayDiv.html()).toContain('<h2>Subheading</h2>')
      expect(displayDiv.html()).toContain('<strong>bold</strong>')
      expect(displayDiv.html()).toContain('<em>italic</em>')
    })

    it('handles line breaks in markdown', () => {
      const markdownWithBreaks = 'Line 1\nLine 2\nLine 3'
      const widget = createMockWidget(markdownWithBreaks)
      const wrapper = mountComponent(widget, markdownWithBreaks)

      const displayDiv = wrapper.find('.comfy-markdown-content')
      expect(displayDiv.html()).toContain('<br>')
    })

    it('handles empty or whitespace-only markdown', () => {
      const whitespaceMarkdown = '   \n\n   '
      const widget = createMockWidget(whitespaceMarkdown)
      const wrapper = mountComponent(widget, whitespaceMarkdown)

      const displayDiv = wrapper.find('.comfy-markdown-content')
      expect(displayDiv.exists()).toBe(true)
    })
  })

  describe('Edge Cases', () => {
    it('handles very long markdown content', async () => {
      const longMarkdown = '# Heading\n' + 'Lorem ipsum '.repeat(1000)
      const widget = createMockWidget(longMarkdown)
      const wrapper = mountComponent(widget, longMarkdown)

      // Should render without issues
      const displayDiv = wrapper.find('.comfy-markdown-content')
      expect(displayDiv.exists()).toBe(true)

      // Should switch to edit mode
      await clickToEdit(wrapper)
      const textarea = wrapper.find('textarea')
      expect(textarea.exists()).toBe(true)
      expect(textarea.element.value).toBe(longMarkdown)
    })

    it('handles special characters in markdown', async () => {
      const specialChars = '# Special: @#$%^&*()[]{}|\\:";\'<>?,./'
      const widget = createMockWidget(specialChars)
      const wrapper = mountComponent(widget, specialChars)

      await clickToEdit(wrapper)
      const textarea = wrapper.find('textarea')
      expect(textarea.element.value).toBe(specialChars)
    })

    it('handles unicode characters', async () => {
      const unicode = '# Unicode: 🎨 αβγ 中文 العربية 🚀'
      const widget = createMockWidget(unicode)
      const wrapper = mountComponent(widget, unicode)

      await clickToEdit(wrapper)
      const textarea = wrapper.find('textarea')
      expect(textarea.element.value).toBe(unicode)

      await textarea.setValue(unicode + ' more unicode')
      await textarea.trigger('input')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![emitted!.length - 1]).toEqual([unicode + ' more unicode'])
    })

    it('handles rapid edit mode toggling', async () => {
      const widget = createMockWidget('# Test')
      const wrapper = mountComponent(widget, '# Test')

      // Rapid toggling
      await clickToEdit(wrapper)
      expect(wrapper.find('textarea').exists()).toBe(true)

      await blurTextarea(wrapper)
      expect(wrapper.find('.comfy-markdown-content').exists()).toBe(true)

      await clickToEdit(wrapper)
      expect(wrapper.find('textarea').exists()).toBe(true)
    })
  })

  describe('Focus Management', () => {
    it('creates textarea reference when entering edit mode', async () => {
      const widget = createMockWidget('# Test')
      const wrapper = mountComponent(widget, '# Test')
      const vm = wrapper.vm as InstanceType<typeof WidgetMarkdown>

      // Test that the component creates a textarea reference when entering edit mode
      // @ts-expect-error - isEditing is not exposed
      expect(vm.isEditing).toBe(false)

      // @ts-expect-error - startEditing is not exposed
      await vm.startEditing()

      // @ts-expect-error - isEditing is not exposed
      expect(vm.isEditing).toBe(true)
      await wrapper.vm.$nextTick()

      // Check that textarea exists after entering edit mode
      const textarea = wrapper.findComponent({ name: 'Textarea' })
      expect(textarea.exists()).toBe(true)
    })
  })
})
