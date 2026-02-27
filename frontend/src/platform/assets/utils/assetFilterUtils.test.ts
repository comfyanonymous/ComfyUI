import { describe, expect, it } from 'vitest'

import type { AssetItem } from '@/platform/assets/schemas/assetSchema'

import {
  filterByBaseModels,
  filterByCategory,
  filterByFileFormats,
  filterByOwnership,
  filterItemByBaseModels,
  filterItemByOwnership
} from './assetFilterUtils'

function createAsset(
  name: string,
  options: Partial<AssetItem> = {}
): AssetItem {
  return {
    id: `asset-${name}`,
    name,
    tags: [],
    is_immutable: false,
    ...options
  } satisfies AssetItem
}

describe('filterByCategory', () => {
  it.for([
    { category: 'all', tags: ['checkpoint'], expected: true },
    { category: 'checkpoint', tags: ['checkpoint'], expected: true },
    { category: 'lora', tags: ['checkpoint'], expected: false },
    {
      category: 'checkpoint',
      tags: ['models', 'checkpoint/xl'],
      expected: true
    },
    { category: 'xl', tags: ['models', 'checkpoint/xl'], expected: false }
  ])(
    'category=$category with tags=$tags returns $expected',
    ({ category, tags, expected }) => {
      const filter = filterByCategory(category)
      const asset = createAsset('model.safetensors', { tags })
      expect(filter(asset)).toBe(expected)
    }
  )
})

describe('filterByFileFormats', () => {
  it.for([
    { formats: [], name: 'model.safetensors', expected: true },
    { formats: ['safetensors'], name: 'model.safetensors', expected: true },
    { formats: ['ckpt'], name: 'model.safetensors', expected: false },
    { formats: ['safetensors'], name: 'MODEL.SAFETENSORS', expected: true },
    { formats: ['safetensors'], name: 'model', expected: false }
  ])(
    'formats=$formats with name=$name returns $expected',
    ({ formats, name, expected }) => {
      const filter = filterByFileFormats(formats)
      const asset = createAsset(name)
      expect(filter(asset)).toBe(expected)
    }
  )

  it('matches any of multiple formats', () => {
    const filter = filterByFileFormats(['safetensors', 'ckpt', 'bin'])
    expect(filter(createAsset('model.safetensors'))).toBe(true)
    expect(filter(createAsset('model.ckpt'))).toBe(true)
    expect(filter(createAsset('model.bin'))).toBe(true)
  })
})

describe('filterByBaseModels', () => {
  it.for([
    { models: [], expected: true },
    { models: new Set<string>(), expected: true }
  ])('empty models ($models) returns true', ({ models }) => {
    const filter = filterByBaseModels(models)
    const asset = createAsset('model.safetensors')
    expect(filter(asset)).toBe(true)
  })

  it.for([
    {
      models: ['SDXL'],
      metadata: { base_model: ['SDXL'] },
      expected: true
    },
    {
      models: ['SDXL'],
      metadata: { base_model: ['SD1.5'] },
      expected: false
    },
    {
      models: new Set(['SDXL', 'SD1.5']),
      metadata: { base_model: ['SDXL'] },
      expected: true
    }
  ])(
    'models=$models with metadata.base_model returns $expected',
    ({ models, metadata, expected }) => {
      const filter = filterByBaseModels(models)
      const asset = createAsset('model.safetensors', { metadata })
      expect(filter(asset)).toBe(expected)
    }
  )

  it('matches base model in user_metadata', () => {
    const filter = filterByBaseModels(['SD1.5'])
    const asset = createAsset('model.safetensors', {
      user_metadata: { base_model: ['SD1.5'] }
    })
    expect(filter(asset)).toBe(true)
  })
})

describe('filterByOwnership', () => {
  it.for([
    { ownership: 'all' as const, is_immutable: true, expected: true },
    { ownership: 'all' as const, is_immutable: false, expected: true },
    { ownership: 'my-models' as const, is_immutable: false, expected: true },
    { ownership: 'my-models' as const, is_immutable: true, expected: false },
    {
      ownership: 'public-models' as const,
      is_immutable: true,
      expected: true
    },
    {
      ownership: 'public-models' as const,
      is_immutable: false,
      expected: false
    }
  ])(
    'ownership=$ownership with is_immutable=$is_immutable returns $expected',
    ({ ownership, is_immutable, expected }) => {
      const filter = filterByOwnership(ownership)
      const asset = createAsset('model', { is_immutable })
      expect(filter(asset)).toBe(expected)
    }
  )
})

describe('filterItemByOwnership', () => {
  const items = [
    { id: '1', is_immutable: true },
    { id: '2', is_immutable: false },
    { id: '3', is_immutable: true }
  ]

  it.for([
    { ownership: 'all' as const, expectedIds: ['1', '2', '3'] },
    { ownership: 'my-models' as const, expectedIds: ['2'] },
    { ownership: 'public-models' as const, expectedIds: ['1', '3'] }
  ])(
    'ownership=$ownership returns items with ids=$expectedIds',
    ({ ownership, expectedIds }) => {
      const result = filterItemByOwnership(items, ownership)
      expect(result.map((i) => i.id)).toEqual(expectedIds)
    }
  )
})

describe('filterItemByBaseModels', () => {
  const items = [
    { id: '1', base_models: ['SDXL'] },
    { id: '2', base_models: ['SD1.5'] },
    { id: '3', base_models: ['SDXL', 'SD1.5'] },
    { id: '4' }
  ]

  it.for([
    { selectedModels: new Set<string>(), expectedIds: ['1', '2', '3', '4'] },
    { selectedModels: new Set(['SDXL']), expectedIds: ['1', '3'] },
    { selectedModels: new Set(['SD1.5']), expectedIds: ['2', '3'] }
  ])(
    'selectedModels=$selectedModels returns items with ids=$expectedIds',
    ({ selectedModels, expectedIds }) => {
      const result = filterItemByBaseModels(items, selectedModels)
      expect(result.map((i) => i.id)).toEqual(expectedIds)
    }
  )
})
