import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import type { VueWrapper } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import { computed } from 'vue'
import type { ComponentPublicInstance } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import type { FormDropdownItem } from '@/renderer/extensions/vueNodes/widgets/components/form/dropdown/types'
import type { ComboInputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { SimplifiedWidget } from '@/types/simplifiedWidget'

import WidgetSelectDropdown from '@/renderer/extensions/vueNodes/widgets/components/WidgetSelectDropdown.vue'

const mockAssetsData = vi.hoisted(() => ({ items: [] as AssetItem[] }))
vi.mock(
  '@/renderer/extensions/vueNodes/widgets/composables/useAssetWidgetData',
  () => ({
    useAssetWidgetData: () => ({
      category: computed(() => 'checkpoints'),
      assets: computed(() => mockAssetsData.items),
      isLoading: computed(() => false),
      error: computed(() => null)
    })
  })
)

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en: {} }
})

interface WidgetSelectDropdownInstance extends ComponentPublicInstance {
  inputItems: FormDropdownItem[]
  outputItems: FormDropdownItem[]
  dropdownItems: FormDropdownItem[]
  filterSelected: string
  updateSelectedItems: (selectedSet: Set<string>) => void
}

describe('WidgetSelectDropdown custom label mapping', () => {
  const createMockWidget = (
    value: string = 'img_001.png',
    options: {
      values?: string[]
      getOptionLabel?: (value?: string | null) => string
    } = {},
    spec?: ComboInputSpec
  ): SimplifiedWidget<string | undefined> => ({
    name: 'test_image_select',
    type: 'combo',
    value,
    options: {
      values: ['img_001.png', 'photo_abc.jpg', 'hash789.png'],
      ...options
    },
    spec
  })

  const mountComponent = (
    widget: SimplifiedWidget<string | undefined>,
    modelValue: string | undefined,
    assetKind: 'image' | 'video' | 'audio' = 'image'
  ): VueWrapper<WidgetSelectDropdownInstance> => {
    return mount(WidgetSelectDropdown, {
      props: {
        widget,
        modelValue,
        assetKind,
        allowUpload: true,
        uploadFolder: 'input'
      },
      global: {
        plugins: [PrimeVue, createTestingPinia(), i18n]
      }
    }) as unknown as VueWrapper<WidgetSelectDropdownInstance>
  }

  describe('when custom labels are not provided', () => {
    it('uses values as labels when no mapping provided', () => {
      const widget = createMockWidget('img_001.png')
      const wrapper = mountComponent(widget, 'img_001.png')

      const inputItems = wrapper.vm.inputItems
      expect(inputItems).toHaveLength(3)
      expect(inputItems[0].name).toBe('img_001.png')
      expect(inputItems[0].label).toBe('img_001.png')
      expect(inputItems[1].name).toBe('photo_abc.jpg')
      expect(inputItems[1].label).toBe('photo_abc.jpg')
      expect(inputItems[2].name).toBe('hash789.png')
      expect(inputItems[2].label).toBe('hash789.png')
    })
  })

  describe('when custom labels are provided via getOptionLabel', () => {
    it('displays custom labels while preserving original values', () => {
      const getOptionLabel = vi.fn((value?: string | null) => {
        if (!value) return 'No file'
        const mapping: Record<string, string> = {
          'img_001.png': 'Vacation Photo',
          'photo_abc.jpg': 'Family Portrait',
          'hash789.png': 'Sunset Beach'
        }
        return mapping[value] || value
      })

      const widget = createMockWidget('img_001.png', {
        getOptionLabel
      })
      const wrapper = mountComponent(widget, 'img_001.png')

      const inputItems = wrapper.vm.inputItems
      expect(inputItems).toHaveLength(3)
      expect(inputItems[0].name).toBe('img_001.png')
      expect(inputItems[0].label).toBe('Vacation Photo')
      expect(inputItems[1].name).toBe('photo_abc.jpg')
      expect(inputItems[1].label).toBe('Family Portrait')
      expect(inputItems[2].name).toBe('hash789.png')
      expect(inputItems[2].label).toBe('Sunset Beach')

      expect(getOptionLabel).toHaveBeenCalledWith('img_001.png')
      expect(getOptionLabel).toHaveBeenCalledWith('photo_abc.jpg')
      expect(getOptionLabel).toHaveBeenCalledWith('hash789.png')
    })

    it('emits original values when items with custom labels are selected', async () => {
      const getOptionLabel = vi.fn((value?: string | null) => {
        if (!value) return 'No file'
        return `Custom: ${value}`
      })

      const widget = createMockWidget('img_001.png', {
        getOptionLabel
      })
      const wrapper = mountComponent(widget, 'img_001.png')

      // Simulate selecting an item
      const selectedSet = new Set(['input-1']) // index 1 = photo_abc.jpg
      wrapper.vm.updateSelectedItems(selectedSet)

      // Should emit the original value, not the custom label
      expect(wrapper.emitted('update:modelValue')).toBeDefined()
      expect(wrapper.emitted('update:modelValue')![0]).toEqual([
        'photo_abc.jpg'
      ])
    })

    it('falls back to original value when label mapping fails', () => {
      const getOptionLabel = vi.fn((value?: string | null) => {
        if (value === 'photo_abc.jpg') {
          throw new Error('Mapping failed')
        }
        return `Labeled: ${value}`
      })

      const consoleErrorSpy = vi
        .spyOn(console, 'error')
        .mockImplementation(() => {})

      const widget = createMockWidget('img_001.png', {
        getOptionLabel
      })
      const wrapper = mountComponent(widget, 'img_001.png')

      const inputItems = wrapper.vm.inputItems
      expect(inputItems[0].name).toBe('img_001.png')
      expect(inputItems[0].label).toBe('Labeled: img_001.png')
      expect(inputItems[1].name).toBe('photo_abc.jpg')
      expect(inputItems[1].label).toBe('photo_abc.jpg')
      expect(inputItems[2].name).toBe('hash789.png')
      expect(inputItems[2].label).toBe('Labeled: hash789.png')

      expect(consoleErrorSpy).toHaveBeenCalled()
      consoleErrorSpy.mockRestore()
    })

    it('falls back to original value when label mapping returns empty string', () => {
      const getOptionLabel = vi.fn((value?: string | null) => {
        if (value === 'photo_abc.jpg') {
          return ''
        }
        return `Labeled: ${value}`
      })

      const widget = createMockWidget('img_001.png', {
        getOptionLabel
      })
      const wrapper = mountComponent(widget, 'img_001.png')

      const inputItems = wrapper.vm.inputItems
      expect(inputItems[0].name).toBe('img_001.png')
      expect(inputItems[0].label).toBe('Labeled: img_001.png')
      expect(inputItems[1].name).toBe('photo_abc.jpg')
      expect(inputItems[1].label).toBe('photo_abc.jpg')
      expect(inputItems[2].name).toBe('hash789.png')
      expect(inputItems[2].label).toBe('Labeled: hash789.png')
    })

    it('falls back to original value when label mapping returns undefined', () => {
      const getOptionLabel = vi.fn((value?: string | null) => {
        if (value === 'hash789.png') {
          return undefined as unknown as string
        }
        return `Labeled: ${value}`
      })

      const widget = createMockWidget('img_001.png', {
        getOptionLabel
      })
      const wrapper = mountComponent(widget, 'img_001.png')

      const inputItems = wrapper.vm.inputItems
      expect(inputItems[0].name).toBe('img_001.png')
      expect(inputItems[0].label).toBe('Labeled: img_001.png')
      expect(inputItems[1].name).toBe('photo_abc.jpg')
      expect(inputItems[1].label).toBe('Labeled: photo_abc.jpg')
      expect(inputItems[2].name).toBe('hash789.png')
      expect(inputItems[2].label).toBe('hash789.png')
    })
  })

  describe('output items with custom label mapping', () => {
    it('applies custom label mapping to output items from queue history', () => {
      const getOptionLabel = vi.fn((value?: string | null) => {
        if (!value) return 'No file'
        return `Output: ${value}`
      })

      const widget = createMockWidget('img_001.png', {
        getOptionLabel
      })
      const wrapper = mountComponent(widget, 'img_001.png')

      const outputItems = wrapper.vm.outputItems
      expect(outputItems).toBeDefined()
      expect(Array.isArray(outputItems)).toBe(true)
    })
  })

  describe('missing value handling for template-loaded nodes', () => {
    it('creates a fallback item in "all" filter when modelValue is not in available items', () => {
      const widget = createMockWidget('template_image.png', {
        values: ['img_001.png', 'photo_abc.jpg']
      })
      const wrapper = mountComponent(widget, 'template_image.png')

      const inputItems = wrapper.vm.inputItems
      expect(inputItems).toHaveLength(2)
      expect(
        inputItems.some((item) => item.name === 'template_image.png')
      ).toBe(false)

      // The missing value should be accessible via dropdownItems when filter is 'all' (default)
      const dropdownItems = wrapper.vm.dropdownItems
      expect(
        dropdownItems.some((item) => item.name === 'template_image.png')
      ).toBe(true)
      expect(dropdownItems[0].name).toBe('template_image.png')
      expect(dropdownItems[0].id).toBe('missing-template_image.png')
    })

    it('does not include fallback item when filter is "inputs"', async () => {
      const widget = createMockWidget('template_image.png', {
        values: ['img_001.png', 'photo_abc.jpg']
      })
      const wrapper = mountComponent(widget, 'template_image.png')

      wrapper.vm.filterSelected = 'inputs'
      await wrapper.vm.$nextTick()

      const dropdownItems = wrapper.vm.dropdownItems
      expect(dropdownItems).toHaveLength(2)
      expect(
        dropdownItems.every((item) => !String(item.id).startsWith('missing-'))
      ).toBe(true)
    })

    it('does not include fallback item when filter is "outputs"', async () => {
      const widget = createMockWidget('template_image.png', {
        values: ['img_001.png', 'photo_abc.jpg']
      })
      const wrapper = mountComponent(widget, 'template_image.png')

      wrapper.vm.filterSelected = 'outputs'
      await wrapper.vm.$nextTick()

      const dropdownItems = wrapper.vm.dropdownItems
      expect(dropdownItems).toHaveLength(wrapper.vm.outputItems.length)
      expect(
        dropdownItems.every((item) => !String(item.id).startsWith('missing-'))
      ).toBe(true)
    })

    it('does not create a fallback item when modelValue exists in available items', () => {
      const widget = createMockWidget('img_001.png', {
        values: ['img_001.png', 'photo_abc.jpg']
      })
      const wrapper = mountComponent(widget, 'img_001.png')

      const dropdownItems = wrapper.vm.dropdownItems
      expect(dropdownItems).toHaveLength(2)
      expect(
        dropdownItems.every((item) => !String(item.id).startsWith('missing-'))
      ).toBe(true)
    })

    it('does not create a fallback item when modelValue is undefined', () => {
      const widget = createMockWidget(undefined as unknown as string, {
        values: ['img_001.png', 'photo_abc.jpg']
      })
      const wrapper = mountComponent(widget, undefined)

      const dropdownItems = wrapper.vm.dropdownItems
      expect(dropdownItems).toHaveLength(2)
      expect(
        dropdownItems.every((item) => !String(item.id).startsWith('missing-'))
      ).toBe(true)
    })
  })
})

describe('WidgetSelectDropdown cloud asset mode (COM-14333)', () => {
  interface CloudModeInstance extends ComponentPublicInstance {
    dropdownItems: FormDropdownItem[]
    displayItems: FormDropdownItem[]
    selectedSet: Set<string>
  }

  const createTestAsset = (
    id: string,
    name: string,
    preview_url: string
  ): AssetItem => ({
    id,
    name,
    preview_url,
    tags: []
  })

  const createCloudModeWidget = (
    value: string = 'model.safetensors'
  ): SimplifiedWidget<string | undefined> => ({
    name: 'test_model_select',
    type: 'combo',
    value,
    options: {
      values: [],
      nodeType: 'CheckpointLoaderSimple'
    }
  })

  const mountCloudComponent = (
    widget: SimplifiedWidget<string | undefined>,
    modelValue: string | undefined
  ): VueWrapper<CloudModeInstance> => {
    return mount(WidgetSelectDropdown, {
      props: {
        widget,
        modelValue,
        assetKind: 'model',
        isAssetMode: true,
        nodeType: 'CheckpointLoaderSimple'
      },
      global: {
        plugins: [PrimeVue, createTestingPinia(), i18n]
      }
    }) as unknown as VueWrapper<CloudModeInstance>
  }

  beforeEach(() => {
    mockAssetsData.items = []
  })

  it('does not include missing items in cloud asset mode dropdown', () => {
    mockAssetsData.items = [
      createTestAsset(
        'asset-1',
        'existing_model.safetensors',
        'https://example.com/preview.jpg'
      )
    ]

    const widget = createCloudModeWidget('missing_model.safetensors')
    const wrapper = mountCloudComponent(widget, 'missing_model.safetensors')

    const dropdownItems = wrapper.vm.dropdownItems
    expect(dropdownItems).toHaveLength(1)
    expect(dropdownItems[0].name).toBe('existing_model.safetensors')
    expect(
      dropdownItems.some((item) => item.name === 'missing_model.safetensors')
    ).toBe(false)
  })

  it('shows only available cloud assets in dropdown', () => {
    mockAssetsData.items = [
      createTestAsset(
        'asset-1',
        'model_a.safetensors',
        'https://example.com/a.jpg'
      ),
      createTestAsset(
        'asset-2',
        'model_b.safetensors',
        'https://example.com/b.jpg'
      )
    ]

    const widget = createCloudModeWidget('model_a.safetensors')
    const wrapper = mountCloudComponent(widget, 'model_a.safetensors')

    const dropdownItems = wrapper.vm.dropdownItems
    expect(dropdownItems).toHaveLength(2)
    expect(dropdownItems.map((item) => item.name)).toEqual([
      'model_a.safetensors',
      'model_b.safetensors'
    ])
  })

  it('returns empty dropdown when no cloud assets available', () => {
    mockAssetsData.items = []

    const widget = createCloudModeWidget('missing_model.safetensors')
    const wrapper = mountCloudComponent(widget, 'missing_model.safetensors')

    const dropdownItems = wrapper.vm.dropdownItems
    expect(dropdownItems).toHaveLength(0)
  })

  it('includes missing cloud asset in displayItems for input field visibility', () => {
    mockAssetsData.items = [
      createTestAsset(
        'asset-1',
        'existing_model.safetensors',
        'https://example.com/preview.jpg'
      )
    ]

    const widget = createCloudModeWidget('missing_model.safetensors')
    const wrapper = mountCloudComponent(widget, 'missing_model.safetensors')

    const displayItems = wrapper.vm.displayItems
    expect(displayItems).toHaveLength(2)
    expect(displayItems[0].name).toBe('missing_model.safetensors')
    expect(displayItems[0].id).toBe('missing-missing_model.safetensors')
    expect(displayItems[1].name).toBe('existing_model.safetensors')

    const selectedSet = wrapper.vm.selectedSet
    expect(selectedSet.has('missing-missing_model.safetensors')).toBe(true)
  })
})
