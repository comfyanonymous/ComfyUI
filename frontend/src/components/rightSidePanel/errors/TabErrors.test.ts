import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import PrimeVue from 'primevue/config'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import TabErrors from './TabErrors.vue'

// Mock dependencies
vi.mock('@/scripts/app', () => ({
  app: {
    rootGraph: {
      serialize: vi.fn(() => ({})),
      getNodeById: vi.fn()
    }
  }
}))

vi.mock('@/utils/graphTraversalUtil', () => ({
  getNodeByExecutionId: vi.fn(),
  getRootParentNode: vi.fn(() => null),
  forEachNode: vi.fn(),
  mapAllNodes: vi.fn(() => [])
}))

vi.mock('@/composables/useCopyToClipboard', () => ({
  useCopyToClipboard: vi.fn(() => ({
    copyToClipboard: vi.fn()
  }))
}))

vi.mock('@/services/litegraphService', () => ({
  useLitegraphService: vi.fn(() => ({
    fitView: vi.fn()
  }))
}))

describe('TabErrors.vue', () => {
  let i18n: ReturnType<typeof createI18n>

  beforeEach(() => {
    i18n = createI18n({
      legacy: false,
      locale: 'en',
      messages: {
        en: {
          g: {
            workflow: 'Workflow',
            copy: 'Copy'
          },
          rightSidePanel: {
            noErrors: 'No errors',
            noneSearchDesc: 'No results found',
            promptErrors: {
              prompt_no_outputs: {
                desc: 'Prompt has no outputs'
              }
            }
          }
        }
      }
    })
  })

  function mountComponent(initialState = {}) {
    return mount(TabErrors, {
      global: {
        plugins: [
          PrimeVue,
          i18n,
          createTestingPinia({
            createSpy: vi.fn,
            initialState
          })
        ],
        stubs: {
          FormSearchInput: {
            template:
              '<input @input="$emit(\'update:modelValue\', $event.target.value)" />'
          },
          PropertiesAccordionItem: {
            template: '<div><slot name="label" /><slot /></div>'
          },
          Button: {
            template: '<button><slot /></button>'
          }
        }
      }
    })
  }

  it('renders "no errors" state when store is empty', () => {
    const wrapper = mountComponent()
    expect(wrapper.text()).toContain('No errors')
  })

  it('renders prompt-level errors (Group title = error message)', async () => {
    const wrapper = mountComponent({
      executionError: {
        lastPromptError: {
          type: 'prompt_no_outputs',
          message: 'Server Error: No outputs',
          details: 'Error details'
        }
      }
    })

    // Group title should be the raw message from store
    expect(wrapper.text()).toContain('Server Error: No outputs')
    // Item message should be localized desc
    expect(wrapper.text()).toContain('Prompt has no outputs')
    // Details should not be rendered for prompt errors
    expect(wrapper.text()).not.toContain('Error details')
  })

  it('renders node validation errors grouped by class_type', async () => {
    const { getNodeByExecutionId } = await import('@/utils/graphTraversalUtil')
    vi.mocked(getNodeByExecutionId).mockReturnValue({
      title: 'CLIP Text Encode'
    } as ReturnType<typeof getNodeByExecutionId>)

    const wrapper = mountComponent({
      executionError: {
        lastNodeErrors: {
          '6': {
            class_type: 'CLIPTextEncode',
            errors: [
              { message: 'Required input is missing', details: 'Input: text' }
            ]
          }
        }
      }
    })

    expect(wrapper.text()).toContain('CLIPTextEncode')
    expect(wrapper.text()).toContain('#6')
    expect(wrapper.text()).toContain('CLIP Text Encode')
    expect(wrapper.text()).toContain('Required input is missing')
  })

  it('renders runtime execution errors from WebSocket', async () => {
    const { getNodeByExecutionId } = await import('@/utils/graphTraversalUtil')
    vi.mocked(getNodeByExecutionId).mockReturnValue({
      title: 'KSampler'
    } as ReturnType<typeof getNodeByExecutionId>)

    const wrapper = mountComponent({
      executionError: {
        lastExecutionError: {
          prompt_id: 'abc',
          node_id: '10',
          node_type: 'KSampler',
          exception_message: 'Out of memory',
          exception_type: 'RuntimeError',
          traceback: ['Line 1', 'Line 2'],
          timestamp: Date.now()
        }
      }
    })

    expect(wrapper.text()).toContain('KSampler')
    expect(wrapper.text()).toContain('#10')
    expect(wrapper.text()).toContain('RuntimeError: Out of memory')
    expect(wrapper.text()).toContain('Line 1')
  })

  it('filters errors based on search query', async () => {
    const { getNodeByExecutionId } = await import('@/utils/graphTraversalUtil')
    vi.mocked(getNodeByExecutionId).mockReturnValue(null)

    const wrapper = mountComponent({
      executionError: {
        lastNodeErrors: {
          '1': {
            class_type: 'CLIPTextEncode',
            errors: [{ message: 'Missing text input' }]
          },
          '2': {
            class_type: 'KSampler',
            errors: [{ message: 'Out of memory' }]
          }
        }
      }
    })

    expect(wrapper.text()).toContain('CLIPTextEncode')
    expect(wrapper.text()).toContain('KSampler')

    const searchInput = wrapper.find('input')
    await searchInput.setValue('Missing text input')

    expect(wrapper.text()).toContain('CLIPTextEncode')
    expect(wrapper.text()).not.toContain('KSampler')
  })

  it('calls copyToClipboard when copy button is clicked', async () => {
    const { useCopyToClipboard } =
      await import('@/composables/useCopyToClipboard')
    const mockCopy = vi.fn()
    vi.mocked(useCopyToClipboard).mockReturnValue({ copyToClipboard: mockCopy })

    const wrapper = mountComponent({
      executionError: {
        lastNodeErrors: {
          '1': {
            class_type: 'TestNode',
            errors: [{ message: 'Test message', details: 'Test details' }]
          }
        }
      }
    })

    // Find the copy button (rendered inside ErrorNodeCard)
    const copyButtons = wrapper.findAll('button')
    const copyButton = copyButtons.find((btn) => btn.text().includes('Copy'))
    expect(copyButton).toBeTruthy()
    await copyButton!.trigger('click')

    expect(mockCopy).toHaveBeenCalledWith('Test message\n\nTest details')
  })
})
