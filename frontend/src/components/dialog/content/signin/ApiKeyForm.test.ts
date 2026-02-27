import type { ComponentProps } from 'vue-component-type-helpers'

import { Form } from '@primevue/forms'
import { mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import Button from '@/components/ui/button/Button.vue'
import PrimeVue from 'primevue/config'
import InputText from 'primevue/inputtext'
import Message from 'primevue/message'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createApp } from 'vue'
import { createI18n } from 'vue-i18n'

import { getComfyPlatformBaseUrl } from '@/config/comfyApi'

import ApiKeyForm from './ApiKeyForm.vue'

const mockStoreApiKey = vi.fn()
const mockLoading = vi.fn(() => false)

vi.mock('@/stores/firebaseAuthStore', () => ({
  useFirebaseAuthStore: vi.fn(() => ({
    loading: mockLoading()
  }))
}))

vi.mock('@/stores/apiKeyAuthStore', () => ({
  useApiKeyAuthStore: vi.fn(() => ({
    storeApiKey: mockStoreApiKey
  }))
}))

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      auth: {
        apiKey: {
          title: 'API Key',
          label: 'API Key',
          placeholder: 'Enter your API Key',
          error: 'Invalid API Key',
          helpText: 'Need an API key?',
          generateKey: 'Get one here',
          whitelistInfo: 'About non-whitelisted sites',
          description: 'Use your Comfy API key to enable API Nodes'
        }
      },
      g: {
        back: 'Back',
        save: 'Save',
        learnMore: 'Learn more'
      }
    }
  }
})

describe('ApiKeyForm', () => {
  beforeEach(() => {
    const app = createApp({})
    app.use(PrimeVue)
    vi.clearAllMocks()
    mockStoreApiKey.mockReset()
    mockLoading.mockReset()
  })

  const mountComponent = (props: ComponentProps<typeof ApiKeyForm> = {}) => {
    return mount(ApiKeyForm, {
      global: {
        plugins: [PrimeVue, createPinia(), i18n],
        components: { Button, Form, InputText, Message }
      },
      props
    })
  }

  it('renders correctly with all required elements', () => {
    const wrapper = mountComponent()

    expect(wrapper.find('h1').text()).toBe('API Key')
    expect(wrapper.find('label').text()).toBe('API Key')
    expect(wrapper.findComponent(InputText).exists()).toBe(true)
    expect(wrapper.findComponent(Button).exists()).toBe(true)
  })

  it('emits back event when back button is clicked', async () => {
    const wrapper = mountComponent()

    await wrapper.findComponent(Button).trigger('click')
    expect(wrapper.emitted('back')).toBeTruthy()
  })

  it('shows loading state when submitting', async () => {
    mockLoading.mockReturnValue(true)
    const wrapper = mountComponent()
    const input = wrapper.findComponent(InputText)

    await input.setValue(
      'comfyui-123456789012345678901234567890123456789012345678901234567890123456789012'
    )
    await wrapper.find('form').trigger('submit')

    const buttons = wrapper.findAllComponents(Button)
    const submitButton = buttons.find(
      (btn) => btn.attributes('type') === 'submit'
    )
    expect(submitButton?.props('loading')).toBe(true)
  })

  it('displays help text and links correctly', () => {
    const wrapper = mountComponent()

    const helpText = wrapper.find('small')
    expect(helpText.text()).toContain('Need an API key?')
    expect(helpText.find('a').attributes('href')).toBe(
      `${getComfyPlatformBaseUrl()}/login`
    )
  })
})
