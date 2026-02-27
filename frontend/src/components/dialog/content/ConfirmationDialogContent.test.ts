import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import PrimeVue from 'primevue/config'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ComponentProps } from 'vue-component-type-helpers'
import { createI18n } from 'vue-i18n'

import ConfirmationDialogContent from './ConfirmationDialogContent.vue'

type Props = ComponentProps<typeof ConfirmationDialogContent>

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en: {} },
  missingWarn: false,
  fallbackWarn: false
})

describe('ConfirmationDialogContent', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  function mountComponent(props: Partial<Props> = {}) {
    return mount(ConfirmationDialogContent, {
      global: {
        plugins: [PrimeVue, i18n]
      },
      props: {
        message: 'Test message',
        type: 'default',
        onConfirm: vi.fn(),
        ...props
      } as Props
    })
  }

  it('renders long messages without breaking layout', () => {
    const longFilename =
      'workflow_checkpoint_' + 'a'.repeat(200) + '.safetensors'
    const wrapper = mountComponent({ message: longFilename })
    expect(wrapper.text()).toContain(longFilename)
  })
})
