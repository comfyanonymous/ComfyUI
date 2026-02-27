import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import { describe, expect, it } from 'vitest'

import CreditTopUpOption from '@/components/dialog/content/credit/CreditTopUpOption.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en: {} }
})

const mountOption = (
  props?: Partial<{ credits: number; description: string; selected: boolean }>
) =>
  mount(CreditTopUpOption, {
    props: {
      credits: 1000,
      description: '~100 videos*',
      selected: false,
      ...props
    },
    global: {
      plugins: [i18n]
    }
  })

describe('CreditTopUpOption', () => {
  it('renders credit amount and description', () => {
    const wrapper = mountOption({ credits: 5000, description: '~500 videos*' })
    expect(wrapper.text()).toContain('5,000')
    expect(wrapper.text()).toContain('~500 videos*')
  })

  it('applies unselected styling when not selected', () => {
    const wrapper = mountOption({ selected: false })
    expect(wrapper.find('div').classes()).toContain(
      'bg-component-node-disabled'
    )
    expect(wrapper.find('div').classes()).toContain('border-transparent')
  })

  it('emits select event when clicked', async () => {
    const wrapper = mountOption()
    await wrapper.find('div').trigger('click')
    expect(wrapper.emitted('select')).toHaveLength(1)
  })
})
