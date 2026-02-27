import { createTestingPinia } from '@pinia/testing'
import { flushPromises, mount } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import type { SimplifiedWidget } from '@/types/simplifiedWidget'

import WidgetSelect from '@/renderer/extensions/vueNodes/widgets/components/WidgetSelect.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en: {} }
})

// Mock modules
vi.mock('@/platform/assets/services/assetService', () => ({
  assetService: {
    shouldUseAssetBrowser: vi.fn(() => true),
    isAssetAPIEnabled: vi.fn(() => true)
  }
}))

// Import after mocks are defined
import { assetService } from '@/platform/assets/services/assetService'
const mockShouldUseAssetBrowser = vi.mocked(assetService.shouldUseAssetBrowser)

describe('WidgetSelect asset mode', () => {
  const createWidget = (): SimplifiedWidget<string | undefined> => ({
    name: 'ckpt_name',
    type: 'combo',
    value: undefined,
    options: {
      values: []
    }
  })

  beforeEach(() => {
    vi.clearAllMocks()
    mockShouldUseAssetBrowser.mockReturnValue(true)
  })

  // Helper to mount with common setup
  const mountWidget = () => {
    return mount(WidgetSelect, {
      props: {
        widget: createWidget(),
        modelValue: undefined,
        nodeType: 'CheckpointLoaderSimple'
      },
      global: {
        plugins: [PrimeVue, createTestingPinia(), i18n]
      }
    })
  }

  it('uses dropdown when isCloud && UseAssetAPI && isEligible', async () => {
    const wrapper = mountWidget()
    await flushPromises()

    expect(
      wrapper.findComponent({ name: 'WidgetSelectDropdown' }).exists()
    ).toBe(true)
  })

  it('uses default widget when shouldUseAssetBrowser returns false', () => {
    mockShouldUseAssetBrowser.mockReturnValue(false)
    const wrapper = mountWidget()

    expect(
      wrapper.findComponent({ name: 'WidgetSelectDefault' }).exists()
    ).toBe(true)
  })
})
