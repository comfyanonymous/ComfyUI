import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import AssetFilterBar from '@/platform/assets/components/AssetFilterBar.vue'
import type { AssetFilterState } from '@/platform/assets/types/filterTypes'
import {
  createAssetWithSpecificBaseModel,
  createAssetWithSpecificExtension,
  createAssetWithoutBaseModel
} from '@/platform/assets/fixtures/ui-mock-assets'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { createI18n } from 'vue-i18n'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {}
  }
})

// Mock components with minimal functionality for business logic testing
vi.mock('@/components/input/MultiSelect.vue', () => ({
  default: {
    name: 'MultiSelect',
    props: {
      modelValue: Array,
      label: String,
      options: Array,
      class: String
    },
    emits: ['update:modelValue'],
    template: `
      <div data-testid="multi-select">
        <select multiple @change="$emit('update:modelValue', Array.from($event.target.selectedOptions).map(o => ({ name: o.text, value: o.value })))">
          <option v-for="option in options" :key="option.value" :value="option.value">
            {{ option.name }}
          </option>
        </select>
      </div>
    `
  }
}))

vi.mock('@/components/input/SingleSelect.vue', () => ({
  default: {
    name: 'SingleSelect',
    props: {
      modelValue: String,
      label: String,
      options: Array,
      class: String
    },
    emits: ['update:modelValue'],
    template: `
      <div data-testid="single-select">
        <select @change="$emit('update:modelValue', $event.target.value)">
          <option v-for="option in options" :key="option.value" :value="option.value">
            {{ option.name }}
          </option>
        </select>
      </div>
    `
  }
}))

// Test factory functions
function mountAssetFilterBar(props = {}) {
  return mount(AssetFilterBar, {
    props,
    global: {
      plugins: [i18n]
    }
  })
}

// Helper functions to find filters by user-facing attributes
function findFileFormatsFilter(
  wrapper: ReturnType<typeof mountAssetFilterBar>
) {
  return wrapper.findComponent(
    '[data-component-id="asset-filter-file-formats"]'
  )
}

function findBaseModelsFilter(wrapper: ReturnType<typeof mountAssetFilterBar>) {
  return wrapper.findComponent('[data-component-id="asset-filter-base-models"]')
}

function findSortFilter(wrapper: ReturnType<typeof mountAssetFilterBar>) {
  return wrapper.findComponent('[data-component-id="asset-filter-sort"]')
}

describe('AssetFilterBar', () => {
  describe('Filter State Management', () => {
    it('handles multiple simultaneous filter changes correctly', async () => {
      // Provide assets with options so filters are visible
      const assets = [
        createAssetWithSpecificExtension('safetensors'),
        createAssetWithSpecificExtension('ckpt'),
        createAssetWithSpecificBaseModel('sd15'),
        createAssetWithSpecificBaseModel('sdxl')
      ]
      const wrapper = mountAssetFilterBar({ assets })

      // Update file formats
      const fileFormatSelect = findFileFormatsFilter(wrapper)
      const fileFormatSelectElement = fileFormatSelect.find('select')
      const options = fileFormatSelectElement.findAll('option')
      const ckptOption = options.find((o) => o.element.value === 'ckpt')!
      const safetensorsOption = options.find(
        (o) => o.element.value === 'safetensors'
      )!
      ckptOption.element.selected = true
      safetensorsOption.element.selected = true
      await fileFormatSelectElement.trigger('change')

      await nextTick()

      // Update base models
      const baseModelSelect = findBaseModelsFilter(wrapper)
      const baseModelSelectElement = baseModelSelect.find('select')
      const sdxlOption = baseModelSelectElement
        .findAll('option')
        .find((o) => o.element.value === 'sdxl')
      sdxlOption!.element.selected = true
      await baseModelSelectElement.trigger('change')

      await nextTick()

      // Update sort
      const sortSelect = findSortFilter(wrapper)
      const sortSelectElement = sortSelect.find('select')
      sortSelectElement.element.value = 'name-desc'
      await sortSelectElement.trigger('change')

      await nextTick()

      const emitted = wrapper.emitted('filterChange')
      expect(emitted).toBeTruthy()
      expect(emitted!.length).toBeGreaterThanOrEqual(3)

      // Check final state
      const finalState: AssetFilterState = emitted![
        emitted!.length - 1
      ][0] as AssetFilterState
      expect(finalState.fileFormats).toEqual(['ckpt', 'safetensors'])
      expect(finalState.baseModels).toEqual(['sdxl'])
      expect(finalState.sortBy).toBe('name-desc')
      expect(finalState.ownership).toBe('all')
    })

    it('ensures AssetFilterState interface compliance', async () => {
      // Provide assets with options so filters are visible
      const assets = [
        createAssetWithSpecificExtension('safetensors'),
        createAssetWithSpecificBaseModel('sd15')
      ]
      const wrapper = mountAssetFilterBar({ assets })

      const fileFormatSelect = findFileFormatsFilter(wrapper)
      const fileFormatSelectElement = fileFormatSelect.find('select')
      const ckptOption = fileFormatSelectElement.findAll('option')[0]
      ckptOption.element.selected = true
      await fileFormatSelectElement.trigger('change')

      await nextTick()

      const emitted = wrapper.emitted('filterChange')
      const filterState = emitted![0][0] as AssetFilterState

      // Type and structure assertions
      expect(Array.isArray(filterState.fileFormats)).toBe(true)
      expect(Array.isArray(filterState.baseModels)).toBe(true)
      expect(typeof filterState.sortBy).toBe('string')

      // Value type assertions
      expect(filterState.fileFormats.every((f) => typeof f === 'string')).toBe(
        true
      )
      expect(filterState.baseModels.every((m) => typeof m === 'string')).toBe(
        true
      )
    })
  })

  describe('Dynamic Filter Options', () => {
    it('should use dynamic file format options based on actual assets', () => {
      const assets = [
        createAssetWithSpecificExtension('safetensors'),
        createAssetWithSpecificExtension('ckpt'),
        createAssetWithSpecificExtension('pt')
      ]

      const wrapper = mountAssetFilterBar({ assets })

      const fileFormatSelect = findFileFormatsFilter(wrapper)
      const options = fileFormatSelect.findAll('option')
      expect(
        options.map((o) => ({ name: o.text(), value: o.element.value }))
      ).toEqual([
        { name: '.ckpt', value: 'ckpt' },
        { name: '.pt', value: 'pt' },
        { name: '.safetensors', value: 'safetensors' }
      ])
    })

    it('should use dynamic base model options based on actual assets', () => {
      const assets = [
        createAssetWithSpecificBaseModel('sd15'),
        createAssetWithSpecificBaseModel('sdxl'),
        createAssetWithSpecificBaseModel('sd35')
      ]

      const wrapper = mountAssetFilterBar({ assets })

      const baseModelSelect = findBaseModelsFilter(wrapper)
      const options = baseModelSelect.findAll('option')
      expect(
        options.map((o) => ({ name: o.text(), value: o.element.value }))
      ).toEqual([
        { name: 'sd15', value: 'sd15' },
        { name: 'sd35', value: 'sd35' },
        { name: 'sdxl', value: 'sdxl' }
      ])
    })
  })

  describe('Conditional Filter Visibility', () => {
    it('hides file format filter when no options available', () => {
      const assets: AssetItem[] = [] // No assets = no file format options
      const wrapper = mountAssetFilterBar({ assets })

      const fileFormatSelect = findFileFormatsFilter(wrapper)
      expect(fileFormatSelect.exists()).toBe(false)
    })

    it('hides base model filter when no options available', () => {
      const assets = [createAssetWithoutBaseModel()] // Asset without base model = no base model options
      const wrapper = mountAssetFilterBar({ assets })

      const baseModelSelect = findBaseModelsFilter(wrapper)
      expect(baseModelSelect.exists()).toBe(false)
    })

    it('shows both filters when options are available', () => {
      const assets = [
        createAssetWithSpecificExtension('safetensors'),
        createAssetWithSpecificBaseModel('sd15')
      ]
      const wrapper = mountAssetFilterBar({ assets })

      const fileFormatSelect = findFileFormatsFilter(wrapper)
      const baseModelSelect = findBaseModelsFilter(wrapper)

      expect(fileFormatSelect.exists()).toBe(true)
      expect(baseModelSelect.exists()).toBe(true)
    })

    it('hides both filters when no assets provided', () => {
      const wrapper = mountAssetFilterBar()

      const fileFormatSelect = findFileFormatsFilter(wrapper)
      const baseModelSelect = findBaseModelsFilter(wrapper)

      expect(fileFormatSelect.exists()).toBe(false)
      expect(baseModelSelect.exists()).toBe(false)
    })
  })
})
