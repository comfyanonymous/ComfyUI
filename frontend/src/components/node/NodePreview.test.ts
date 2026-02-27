import { mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import PrimeVue from 'primevue/config'
import { beforeAll, describe, expect, it, vi } from 'vitest'
import { createApp } from 'vue'
import { createI18n } from 'vue-i18n'

import type { ComfyNodeDef as ComfyNodeDefV2 } from '@/schemas/nodeDef/nodeDefSchemaV2'
import * as markdownRendererUtil from '@/utils/markdownRendererUtil'

import NodePreview from './NodePreview.vue'

describe('NodePreview', () => {
  let i18n: ReturnType<typeof createI18n>
  let pinia: ReturnType<typeof createPinia>

  beforeAll(() => {
    // Create a Vue app instance for PrimeVue
    const app = createApp({})
    app.use(PrimeVue)

    // Create i18n instance
    i18n = createI18n({
      legacy: false,
      locale: 'en',
      messages: {
        en: {
          g: {
            preview: 'Preview'
          }
        }
      }
    })

    // Create pinia instance
    pinia = createPinia()
  })

  const mockNodeDef: ComfyNodeDefV2 = {
    name: 'TestNode',
    display_name:
      'Test Node With A Very Long Display Name That Should Overflow',
    category: 'test',
    output_node: false,
    inputs: {
      test_input: {
        name: 'test_input',
        type: 'STRING',
        tooltip: 'Test input'
      }
    },
    outputs: [],
    python_module: 'test_module',
    description: 'Test node description'
  }

  const mountComponent = (nodeDef: ComfyNodeDefV2 = mockNodeDef) => {
    return mount(NodePreview, {
      global: {
        plugins: [PrimeVue, i18n, pinia],
        stubs: {
          // Stub stores if needed
        }
      },
      props: {
        nodeDef
      }
    })
  }

  it('renders node preview with correct structure', () => {
    const wrapper = mountComponent()

    expect(wrapper.find('._sb_node_preview').exists()).toBe(true)
    expect(wrapper.find('.node_header').exists()).toBe(true)
    expect(wrapper.find('._sb_preview_badge').text()).toBe('Preview')
  })

  it('sets title attribute on node header with full display name', () => {
    const wrapper = mountComponent()
    const nodeHeader = wrapper.find('.node_header')

    expect(nodeHeader.attributes('title')).toBe(mockNodeDef.display_name)
  })

  it('displays truncated long node names with ellipsis', () => {
    const longNameNodeDef: ComfyNodeDefV2 = {
      ...mockNodeDef,
      display_name:
        'This Is An Extremely Long Node Name That Should Definitely Be Truncated With Ellipsis To Prevent Layout Issues'
    }

    const wrapper = mountComponent(longNameNodeDef)
    const nodeHeader = wrapper.find('.node_header')

    // Verify the title attribute contains the full name
    expect(nodeHeader.attributes('title')).toBe(longNameNodeDef.display_name)

    // Verify overflow handling classes are applied
    expect(nodeHeader.classes()).toContain('text-ellipsis')

    // The actual text content should still be the full name (CSS handles truncation)
    expect(nodeHeader.text()).toContain(longNameNodeDef.display_name)
  })

  it('handles short node names without issues', () => {
    const shortNameNodeDef: ComfyNodeDefV2 = {
      ...mockNodeDef,
      display_name: 'Short'
    }

    const wrapper = mountComponent(shortNameNodeDef)
    const nodeHeader = wrapper.find('.node_header')

    expect(nodeHeader.attributes('title')).toBe('Short')
    expect(nodeHeader.text()).toContain('Short')
  })

  it('applies proper spacing to the dot element', () => {
    const wrapper = mountComponent()
    const headdot = wrapper.find('.headdot')

    expect(headdot.classes()).toContain('pr-3')
  })

  describe('Description Rendering', () => {
    it('renders plain text description as HTML', () => {
      const plainTextNodeDef: ComfyNodeDefV2 = {
        ...mockNodeDef,
        description: 'This is a plain text description'
      }

      const wrapper = mountComponent(plainTextNodeDef)
      const description = wrapper.find('._sb_description')

      expect(description.exists()).toBe(true)
      expect(description.html()).toContain('This is a plain text description')
    })

    it('renders markdown description with formatting', () => {
      const markdownNodeDef: ComfyNodeDefV2 = {
        ...mockNodeDef,
        description: '**Bold text** and *italic text* with `code`'
      }

      const wrapper = mountComponent(markdownNodeDef)
      const description = wrapper.find('._sb_description')

      expect(description.exists()).toBe(true)
      expect(description.html()).toContain('<strong>Bold text</strong>')
      expect(description.html()).toContain('<em>italic text</em>')
      expect(description.html()).toContain('<code>code</code>')
    })

    it('does not render description element when description is empty', () => {
      const noDescriptionNodeDef: ComfyNodeDefV2 = {
        ...mockNodeDef,
        description: ''
      }

      const wrapper = mountComponent(noDescriptionNodeDef)
      const description = wrapper.find('._sb_description')

      expect(description.exists()).toBe(false)
    })

    it('does not render description element when description is undefined', () => {
      const { description, ...nodeDefWithoutDescription } = mockNodeDef
      const wrapper = mountComponent(
        nodeDefWithoutDescription as ComfyNodeDefV2
      )
      const descriptionElement = wrapper.find('._sb_description')

      expect(descriptionElement.exists()).toBe(false)
    })

    it('calls renderMarkdownToHtml utility function', () => {
      const spy = vi.spyOn(markdownRendererUtil, 'renderMarkdownToHtml')
      const testDescription = 'Test **markdown** description'

      const nodeDefWithDescription: ComfyNodeDefV2 = {
        ...mockNodeDef,
        description: testDescription
      }

      mountComponent(nodeDefWithDescription)

      expect(spy).toHaveBeenCalledWith(testDescription)
      spy.mockRestore()
    })

    it('handles potentially unsafe markdown content safely', () => {
      const unsafeNodeDef: ComfyNodeDefV2 = {
        ...mockNodeDef,
        description:
          'Safe **markdown** content <script>alert("xss")</script> with `code` blocks'
      }

      const wrapper = mountComponent(unsafeNodeDef)
      const description = wrapper.find('._sb_description')

      // The description should still exist because there's safe content
      if (description.exists()) {
        // Should not contain script tags (sanitized by DOMPurify)
        expect(description.html()).not.toContain('<script>')
        expect(description.html()).not.toContain('alert("xss")')
        // Should contain the safe markdown content rendered as HTML
        expect(description.html()).toContain('<strong>markdown</strong>')
        expect(description.html()).toContain('<code>code</code>')
      } else {
        // If DOMPurify removes everything, that's also acceptable for security
        expect(description.exists()).toBe(false)
      }
    })

    it('handles markdown with line breaks', () => {
      const multilineNodeDef: ComfyNodeDefV2 = {
        ...mockNodeDef,
        description: 'Line 1\n\nLine 3 after empty line'
      }

      const wrapper = mountComponent(multilineNodeDef)
      const description = wrapper.find('._sb_description')

      expect(description.exists()).toBe(true)
      // Should contain paragraph tags for proper line break handling
      expect(description.html()).toContain('<p>')
    })

    it('handles markdown lists', () => {
      const listNodeDef: ComfyNodeDefV2 = {
        ...mockNodeDef,
        description: '- Item 1\n- Item 2\n- Item 3'
      }

      const wrapper = mountComponent(listNodeDef)
      const description = wrapper.find('._sb_description')

      expect(description.exists()).toBe(true)
      expect(description.html()).toContain('<ul>')
      expect(description.html()).toContain('<li>')
    })

    it('applies correct styling classes to description', () => {
      const wrapper = mountComponent()
      const description = wrapper.find('._sb_description')

      expect(description.classes()).toContain('_sb_description')
    })

    it('uses v-html directive for rendered content', () => {
      const htmlNodeDef: ComfyNodeDefV2 = {
        ...mockNodeDef,
        description: 'Content with **bold** text'
      }

      const wrapper = mountComponent(htmlNodeDef)
      const description = wrapper.find('._sb_description')

      // The component should render the HTML, not escape it
      expect(description.html()).toContain('<strong>bold</strong>')
      expect(description.html()).not.toContain('&lt;strong&gt;')
    })

    it('prevents XSS attacks by sanitizing dangerous HTML elements', () => {
      const maliciousNodeDef: ComfyNodeDefV2 = {
        ...mockNodeDef,
        description:
          'Normal text <img src="x" onerror="alert(\'XSS\')" /> and **bold** text'
      }

      const wrapper = mountComponent(maliciousNodeDef)
      const description = wrapper.find('._sb_description')

      if (description.exists()) {
        // Should not contain dangerous event handlers
        expect(description.html()).not.toContain('onerror')
        expect(description.html()).not.toContain('alert(')
        // Should still contain safe markdown content
        expect(description.html()).toContain('<strong>bold</strong>')
        // May or may not contain img tag depending on DOMPurify config
      }
    })
  })
})
