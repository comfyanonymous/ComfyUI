import { describe, expect, it } from 'vitest'

import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import {
  getAssetAdditionalTags,
  getAssetBaseModel,
  getAssetBaseModels,
  getAssetDescription,
  getAssetDisplayName,
  getAssetModelType,
  getAssetSourceUrl,
  getAssetTriggerPhrases,
  getAssetUserDescription,
  getSourceName
} from '@/platform/assets/utils/assetMetadataUtils'

describe('assetMetadataUtils', () => {
  const mockAsset: AssetItem = {
    id: 'test-id',
    name: 'test-model',
    asset_hash: 'hash123',
    size: 1024,
    mime_type: 'application/octet-stream',
    tags: ['models', 'checkpoints'],
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    last_access_time: '2024-01-01T00:00:00Z'
  }

  describe('getAssetDescription', () => {
    it.for([
      {
        name: 'returns string description when present',
        description: 'A test model',
        expected: 'A test model'
      },
      { name: 'returns null for non-string', description: 123, expected: null },
      { name: 'returns null for null', description: null, expected: null }
    ])('$name', ({ description, expected }) => {
      const asset = { ...mockAsset, user_metadata: { description } }
      expect(getAssetDescription(asset)).toBe(expected)
    })

    it('should return null when no metadata', () => {
      expect(getAssetDescription(mockAsset)).toBeNull()
    })
  })

  describe('getAssetBaseModel', () => {
    it.for([
      {
        name: 'returns string base_model when present',
        base_model: 'SDXL',
        expected: 'SDXL'
      },
      { name: 'returns null for non-string', base_model: 123, expected: null },
      { name: 'returns null for null', base_model: null, expected: null }
    ])('$name', ({ base_model, expected }) => {
      const asset = { ...mockAsset, user_metadata: { base_model } }
      expect(getAssetBaseModel(asset)).toBe(expected)
    })

    it('should return null when no metadata', () => {
      expect(getAssetBaseModel(mockAsset)).toBeNull()
    })
  })

  describe('getAssetDisplayName', () => {
    it.for([
      {
        name: 'returns name from user_metadata when present',
        user_metadata: { name: 'My Custom Name' },
        expected: 'My Custom Name'
      },
      {
        name: 'falls back to asset name for non-string',
        user_metadata: { name: 123 },
        expected: 'test-model'
      },
      {
        name: 'falls back to asset name for undefined',
        user_metadata: undefined,
        expected: 'test-model'
      }
    ])('$name', ({ user_metadata, expected }) => {
      const asset = { ...mockAsset, user_metadata }
      expect(getAssetDisplayName(asset)).toBe(expected)
    })
  })

  describe('getAssetSourceUrl', () => {
    it.for([
      {
        name: 'constructs URL from civitai format',
        source_arn: 'civitai:model:123:version:456',
        expected: 'https://civitai.com/models/123?modelVersionId=456'
      },
      { name: 'returns null for non-string', source_arn: 123, expected: null },
      {
        name: 'returns null for unrecognized format',
        source_arn: 'unknown:format',
        expected: null
      }
    ])('$name', ({ source_arn, expected }) => {
      const asset = { ...mockAsset, user_metadata: { source_arn } }
      expect(getAssetSourceUrl(asset)).toBe(expected)
    })

    it('should return null when no metadata', () => {
      expect(getAssetSourceUrl(mockAsset)).toBeNull()
    })
  })

  describe('getAssetTriggerPhrases', () => {
    it.for([
      {
        name: 'returns array when array present',
        trained_words: ['phrase1', 'phrase2'],
        expected: ['phrase1', 'phrase2']
      },
      {
        name: 'wraps single string in array',
        trained_words: 'single phrase',
        expected: ['single phrase']
      },
      {
        name: 'filters non-string values from array',
        trained_words: ['valid', 123, 'also valid', null],
        expected: ['valid', 'also valid']
      }
    ])('$name', ({ trained_words, expected }) => {
      const asset = { ...mockAsset, user_metadata: { trained_words } }
      expect(getAssetTriggerPhrases(asset)).toEqual(expected)
    })

    it('should return empty array when no metadata', () => {
      expect(getAssetTriggerPhrases(mockAsset)).toEqual([])
    })
  })

  describe('getAssetAdditionalTags', () => {
    it.for([
      {
        name: 'returns array of tags when present',
        additional_tags: ['tag1', 'tag2'],
        expected: ['tag1', 'tag2']
      },
      {
        name: 'filters non-string values from array',
        additional_tags: ['valid', 123, 'also valid'],
        expected: ['valid', 'also valid']
      },
      {
        name: 'returns empty array for non-array',
        additional_tags: 'not an array',
        expected: []
      }
    ])('$name', ({ additional_tags, expected }) => {
      const asset = { ...mockAsset, user_metadata: { additional_tags } }
      expect(getAssetAdditionalTags(asset)).toEqual(expected)
    })

    it('should return empty array when no metadata', () => {
      expect(getAssetAdditionalTags(mockAsset)).toEqual([])
    })
  })

  describe('getSourceName', () => {
    it.for([
      {
        name: 'returns Civitai for civitai.com',
        url: 'https://civitai.com/models/123',
        expected: 'Civitai'
      },
      {
        name: 'returns Hugging Face for huggingface.co',
        url: 'https://huggingface.co/org/model',
        expected: 'Hugging Face'
      },
      {
        name: 'returns Source for unknown URLs',
        url: 'https://example.com/model',
        expected: 'Source'
      }
    ])('$name', ({ url, expected }) => {
      expect(getSourceName(url)).toBe(expected)
    })
  })

  describe('getAssetBaseModels', () => {
    it.for([
      {
        name: 'array of strings',
        base_model: ['SDXL', 'SD1.5', 'Flux'],
        expected: ['SDXL', 'SD1.5', 'Flux']
      },
      {
        name: 'filters non-string entries',
        base_model: ['SDXL', 123, 'SD1.5', null, undefined],
        expected: ['SDXL', 'SD1.5']
      },
      {
        name: 'single string wrapped in array',
        base_model: 'SDXL',
        expected: ['SDXL']
      },
      {
        name: 'non-array/string returns empty',
        base_model: 123,
        expected: []
      },
      { name: 'undefined returns empty', base_model: undefined, expected: [] }
    ])('$name', ({ base_model, expected }) => {
      const asset = { ...mockAsset, user_metadata: { base_model } }
      expect(getAssetBaseModels(asset)).toEqual(expected)
    })

    it('should return empty array when no metadata', () => {
      expect(getAssetBaseModels(mockAsset)).toEqual([])
    })
  })

  describe('getAssetModelType', () => {
    it.for([
      {
        name: 'returns model type from tags',
        tags: ['models', 'checkpoints'],
        expected: 'checkpoints'
      },
      {
        name: 'extracts last segment from path-style tags',
        tags: ['models', 'models/loras'],
        expected: 'loras'
      },
      {
        name: 'returns null when only models tag',
        tags: ['models'],
        expected: null
      },
      { name: 'returns null when tags empty', tags: [], expected: null }
    ])('$name', ({ tags, expected }) => {
      const asset = { ...mockAsset, tags }
      expect(getAssetModelType(asset)).toBe(expected)
    })
  })

  describe('getAssetUserDescription', () => {
    it.for([
      {
        name: 'returns description when present',
        user_description: 'A custom user description',
        expected: 'A custom user description'
      },
      {
        name: 'returns empty for non-string',
        user_description: 123,
        expected: ''
      },
      { name: 'returns empty for null', user_description: null, expected: '' },
      {
        name: 'returns empty for undefined',
        user_description: undefined,
        expected: ''
      }
    ])('$name', ({ user_description, expected }) => {
      const asset = { ...mockAsset, user_metadata: { user_description } }
      expect(getAssetUserDescription(asset)).toBe(expected)
    })

    it('should return empty string when no metadata', () => {
      expect(getAssetUserDescription(mockAsset)).toBe('')
    })
  })
})
