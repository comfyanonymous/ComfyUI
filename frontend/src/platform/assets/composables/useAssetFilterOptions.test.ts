import { describe, expect, it, vi } from 'vitest'

import { useAssetFilterOptions } from '@/platform/assets/composables/useAssetFilterOptions'

import {
  createAssetWithSpecificBaseModel,
  createAssetWithSpecificExtension,
  createAssetWithoutBaseModel,
  createAssetWithoutExtension,
  createAssetWithoutUserMetadata
} from '@/platform/assets/fixtures/ui-mock-assets'

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: (key: string) => key
  })
}))

describe('useAssetFilterOptions', () => {
  describe('File Format Extraction', () => {
    it('extracts file formats from asset names', () => {
      const assets = [
        createAssetWithSpecificExtension('safetensors'),
        createAssetWithSpecificExtension('ckpt'),
        createAssetWithSpecificExtension('pt')
      ]

      const { availableFileFormats } = useAssetFilterOptions(() => assets)

      expect(availableFileFormats.value).toEqual([
        { name: '.ckpt', value: 'ckpt' },
        { name: '.pt', value: 'pt' },
        { name: '.safetensors', value: 'safetensors' }
      ])
    })

    it('handles duplicate file formats', () => {
      const assets = [
        createAssetWithSpecificExtension('safetensors'),
        createAssetWithSpecificExtension('safetensors'),
        createAssetWithSpecificExtension('ckpt')
      ]

      const { availableFileFormats } = useAssetFilterOptions(() => assets)

      expect(availableFileFormats.value).toEqual([
        { name: '.ckpt', value: 'ckpt' },
        { name: '.safetensors', value: 'safetensors' }
      ])
    })

    it('handles assets with no file extension', () => {
      const assets = [
        createAssetWithoutExtension(),
        createAssetWithSpecificExtension('safetensors')
      ]

      const { availableFileFormats } = useAssetFilterOptions(() => assets)

      expect(availableFileFormats.value).toEqual([
        { name: '.safetensors', value: 'safetensors' }
      ])
    })

    it('handles empty asset list', () => {
      const { availableFileFormats } = useAssetFilterOptions(() => [])

      expect(availableFileFormats.value).toEqual([])
    })
  })

  describe('Base Model Extraction', () => {
    it('extracts base models from user metadata', () => {
      const assets = [
        createAssetWithSpecificBaseModel('sd15'),
        createAssetWithSpecificBaseModel('sdxl'),
        createAssetWithSpecificBaseModel('sd35')
      ]

      const { availableBaseModels } = useAssetFilterOptions(() => assets)

      expect(availableBaseModels.value).toEqual([
        { name: 'sd15', value: 'sd15' },
        { name: 'sd35', value: 'sd35' },
        { name: 'sdxl', value: 'sdxl' }
      ])
    })

    it('handles duplicate base models', () => {
      const assets = [
        createAssetWithSpecificBaseModel('sd15'),
        createAssetWithSpecificBaseModel('sd15'),
        createAssetWithSpecificBaseModel('sdxl')
      ]

      const { availableBaseModels } = useAssetFilterOptions(() => assets)

      expect(availableBaseModels.value).toEqual([
        { name: 'sd15', value: 'sd15' },
        { name: 'sdxl', value: 'sdxl' }
      ])
    })

    it('handles assets with missing user_metadata', () => {
      const assets = [
        createAssetWithoutUserMetadata(),
        createAssetWithSpecificBaseModel('sd15')
      ]

      const { availableBaseModels } = useAssetFilterOptions(() => assets)

      expect(availableBaseModels.value).toEqual([
        { name: 'sd15', value: 'sd15' }
      ])
    })

    it('handles assets with missing base_model field', () => {
      const assets = [
        createAssetWithoutBaseModel(),
        createAssetWithSpecificBaseModel('sdxl')
      ]

      const { availableBaseModels } = useAssetFilterOptions(() => assets)

      expect(availableBaseModels.value).toEqual([
        { name: 'sdxl', value: 'sdxl' }
      ])
    })

    it('handles empty asset list', () => {
      const { availableBaseModels } = useAssetFilterOptions(() => [])

      expect(availableBaseModels.value).toEqual([])
    })
  })

  describe('Reactivity', () => {
    it('returns computed properties that can be reactive', () => {
      const assets = [createAssetWithSpecificExtension('safetensors')]

      const { availableFileFormats, availableBaseModels } =
        useAssetFilterOptions(() => assets)

      expect(availableFileFormats.value).toBeDefined()
      expect(availableBaseModels.value).toBeDefined()
      expect(typeof availableFileFormats.value).toBe('object')
      expect(typeof availableBaseModels.value).toBe('object')
      expect(Array.isArray(availableFileFormats.value)).toBe(true)
      expect(Array.isArray(availableBaseModels.value)).toBe(true)
    })
  })
})
