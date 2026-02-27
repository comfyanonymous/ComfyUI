import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import PrimeVue from 'primevue/config'
import Tooltip from 'primevue/tooltip'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import InfoButton from '@/components/graph/selectionToolbox/InfoButton.vue'
import Button from '@/components/ui/button/Button.vue'

const { openPanelMock } = vi.hoisted(() => ({
  openPanelMock: vi.fn()
}))

vi.mock('@/stores/workspace/rightSidePanelStore', () => ({
  useRightSidePanelStore: () => ({
    openPanel: openPanelMock
  })
}))

describe('InfoButton', () => {
  const i18n = createI18n({
    legacy: false,
    locale: 'en',
    messages: {
      en: {
        g: {
          info: 'Node Info'
        }
      }
    }
  })

  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  const mountComponent = () => {
    return mount(InfoButton, {
      global: {
        plugins: [i18n, PrimeVue],
        directives: { tooltip: Tooltip },
        components: { Button }
      }
    })
  }

  it('should open the info panel on click', async () => {
    const wrapper = mountComponent()
    const button = wrapper.find('[data-testid="info-button"]')
    await button.trigger('click')
    expect(openPanelMock).toHaveBeenCalledWith('info')
  })
})
