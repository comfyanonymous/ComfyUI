import type { VueWrapper } from '@vue/test-utils'
import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import PrimeVue from 'primevue/config'
import { describe, expect, it } from 'vitest'
import { nextTick } from 'vue'
import { createI18n } from 'vue-i18n'
import type { ComponentProps } from 'vue-component-type-helpers'

import enMessages from '@/locales/en/main.json' with { type: 'json' }

import GridSkeleton from './GridSkeleton.vue'
import PackCardSkeleton from './PackCardSkeleton.vue'

describe('GridSkeleton', () => {
  function mountComponent({
    props = {}
  }: {
    props?: Partial<ComponentProps<typeof GridSkeleton>>
  } = {}): VueWrapper {
    const i18n = createI18n({
      legacy: false,
      locale: 'en',
      messages: { en: enMessages }
    })

    return mount(GridSkeleton, {
      props: {
        gridStyle: {
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(19rem, 1fr))',
          padding: '0.5rem',
          gap: '1.5rem'
        },
        ...props
      },
      global: {
        plugins: [PrimeVue, createTestingPinia({ stubActions: false }), i18n],
        stubs: {
          PackCardSkeleton: true
        }
      }
    })
  }

  it('renders with default props', () => {
    const wrapper = mountComponent()
    expect(wrapper.exists()).toBe(true)
  })

  it('applies the provided grid style', () => {
    const customGridStyle = {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(15rem, 1fr))',
      padding: '1rem',
      gap: '1rem'
    }

    const wrapper = mountComponent({
      props: { gridStyle: customGridStyle }
    })

    const gridElement = wrapper.element
    expect(gridElement.style.display).toBe('grid')
    expect(gridElement.style.gridTemplateColumns).toBe(
      'repeat(auto-fill, minmax(15rem, 1fr))'
    )
    expect(gridElement.style.padding).toBe('1rem')
    expect(gridElement.style.gap).toBe('1rem')
  })

  it('renders the specified number of skeleton cards', async () => {
    const cardCount = 5
    const wrapper = mountComponent({
      props: { skeletonCardCount: cardCount }
    })

    await nextTick()

    const skeletonCards = wrapper.findAllComponents(PackCardSkeleton)
    expect(skeletonCards.length).toBe(5)
  })
})
