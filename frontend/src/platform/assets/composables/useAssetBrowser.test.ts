import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, ref } from 'vue'

import { useAssetBrowser } from '@/platform/assets/composables/useAssetBrowser'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: (key: string) => key
  })
}))

vi.mock('@/i18n', () => ({
  t: (key: string) => {
    const translations: Record<string, string> = {
      'assetBrowser.allModels': 'All Models',
      'assetBrowser.imported': 'Imported',
      'assetBrowser.byType': 'By type',
      'assetBrowser.assets': 'Assets',
      'assetBrowser.unknown': 'unknown'
    }
    return translations[key] || key
  },
  d: (date: Date) => date.toLocaleDateString()
}))

describe('useAssetBrowser', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.restoreAllMocks()
  })

  // Test fixtures - minimal data focused on functionality being tested
  const createApiAsset = (overrides: Partial<AssetItem> = {}): AssetItem => ({
    id: 'test-id',
    name: 'test-asset.safetensors',
    asset_hash: 'blake3:abc123',
    size: 1024,
    mime_type: 'application/octet-stream',
    tags: ['models', 'checkpoints'],
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    last_access_time: '2024-01-01T00:00:00Z',
    is_immutable: false,
    ...overrides
  })

  describe('Category Filtering', () => {
    it('exposes category-filtered assets for filter options', () => {
      const checkpointAsset = createApiAsset({
        id: 'checkpoint-1',
        name: 'model.safetensors',
        tags: ['models', 'checkpoints']
      })
      const loraAsset = createApiAsset({
        id: 'lora-1',
        name: 'lora.pt',
        tags: ['models', 'loras']
      })

      const { selectedNavItem, categoryFilteredAssets } = useAssetBrowser(
        ref([checkpointAsset, loraAsset])
      )

      // Initially should show all assets
      expect(categoryFilteredAssets.value).toHaveLength(2)

      // When category selected, should only show that category
      selectedNavItem.value = 'checkpoints'
      expect(categoryFilteredAssets.value).toHaveLength(1)
      expect(categoryFilteredAssets.value[0].id).toBe('checkpoint-1')

      selectedNavItem.value = 'loras'
      expect(categoryFilteredAssets.value).toHaveLength(1)
      expect(categoryFilteredAssets.value[0].id).toBe('lora-1')
    })
  })

  describe('Asset Transformation', () => {
    it('transforms API asset to include display properties', () => {
      const apiAsset = createApiAsset({
        user_metadata: { description: 'Test model' }
      })

      const { filteredAssets } = useAssetBrowser(ref([apiAsset]))
      const result = filteredAssets.value[0] // Get the transformed asset from filteredAssets

      // Preserves API properties
      expect(result.id).toBe(apiAsset.id)
      expect(result.name).toBe(apiAsset.name)

      // Adds display properties
      expect(result.secondaryText).toBe('test-asset.safetensors')
      expect(result.badges).toContainEqual({
        label: 'checkpoints',
        type: 'type'
      })
    })

    it('creates secondaryText from filename when metadata missing', () => {
      const apiAsset = createApiAsset({
        tags: ['models', 'loras'],
        user_metadata: undefined
      })

      const { filteredAssets } = useAssetBrowser(ref([apiAsset]))
      const result = filteredAssets.value[0]

      expect(result.secondaryText).toBe('test-asset.safetensors')
    })

    it('removes category prefix from badge labels', () => {
      const apiAsset = createApiAsset({
        tags: ['models', 'checkpoint/stable-diffusion-v1-5']
      })

      const { filteredAssets } = useAssetBrowser(ref([apiAsset]))
      const result = filteredAssets.value[0]

      expect(result.badges).toContainEqual({
        label: 'stable-diffusion-v1-5',
        type: 'type'
      })
    })

    it('handles tags without slash for badges', () => {
      const apiAsset = createApiAsset({
        tags: ['models', 'checkpoints']
      })

      const { filteredAssets } = useAssetBrowser(ref([apiAsset]))
      const result = filteredAssets.value[0]

      expect(result.badges).toContainEqual({
        label: 'checkpoints',
        type: 'type'
      })
    })

    it('handles tags with multiple slashes in badges', () => {
      const apiAsset = createApiAsset({
        tags: ['models', 'checkpoint/subfolder/model-name']
      })

      const { filteredAssets } = useAssetBrowser(ref([apiAsset]))
      const result = filteredAssets.value[0]

      expect(result.badges).toContainEqual({
        label: 'subfolder/model-name',
        type: 'type'
      })
    })
  })

  describe('Tag-Based Filtering', () => {
    it('filters assets by category tag', async () => {
      const assets = [
        createApiAsset({ id: '1', tags: ['models', 'checkpoints'] }),
        createApiAsset({ id: '2', tags: ['models', 'loras'] }),
        createApiAsset({ id: '3', tags: ['models', 'checkpoints'] })
      ]

      const { selectedNavItem, filteredAssets } = useAssetBrowser(ref(assets))

      selectedNavItem.value = 'checkpoints'
      await nextTick()

      expect(filteredAssets.value).toHaveLength(2)
      expect(
        filteredAssets.value.every((asset) =>
          asset.tags.includes('checkpoints')
        )
      ).toBe(true)
    })

    it('returns all assets when category is "all"', async () => {
      const assets = [
        createApiAsset({ id: '1', tags: ['models', 'checkpoints'] }),
        createApiAsset({ id: '2', tags: ['models', 'loras'] })
      ]

      const { selectedNavItem, filteredAssets } = useAssetBrowser(ref(assets))

      selectedNavItem.value = 'all'
      await nextTick()

      expect(filteredAssets.value).toHaveLength(2)
    })
  })

  describe('Fuzzy Search Functionality', () => {
    it('searches across asset name with exact match', async () => {
      const assets = [
        createApiAsset({ name: 'realistic_vision.safetensors' }),
        createApiAsset({ name: 'anime_style.ckpt' }),
        createApiAsset({ name: 'photorealistic_v2.safetensors' })
      ]

      const { searchQuery, filteredAssets } = useAssetBrowser(ref(assets))

      searchQuery.value = 'realistic'
      await nextTick()

      expect(filteredAssets.value.length).toBeGreaterThanOrEqual(1)
      expect(
        filteredAssets.value.some((asset) =>
          asset.name.toLowerCase().includes('realistic')
        )
      ).toBe(true)
    })

    it('searches across asset tags', async () => {
      const assets = [
        createApiAsset({
          name: 'model1.safetensors',
          tags: ['models', 'checkpoints']
        }),
        createApiAsset({
          name: 'model2.safetensors',
          tags: ['models', 'loras']
        })
      ]

      const { searchQuery, filteredAssets } = useAssetBrowser(ref(assets))

      searchQuery.value = 'checkpoints'
      await nextTick()

      expect(filteredAssets.value.length).toBeGreaterThanOrEqual(1)
      expect(filteredAssets.value[0].tags).toContain('checkpoints')
    })

    it('supports fuzzy matching with typos', async () => {
      const assets = [
        createApiAsset({ name: 'checkpoint_model.safetensors' }),
        createApiAsset({ name: 'lora_model.safetensors' })
      ]

      const { searchQuery, filteredAssets } = useAssetBrowser(ref(assets))

      // Intentional typo - fuzzy search should still find it
      searchQuery.value = 'chckpoint'
      await nextTick()

      expect(filteredAssets.value.length).toBeGreaterThanOrEqual(1)
      expect(filteredAssets.value[0].name).toContain('checkpoint')
    })

    it('handles empty search by returning all assets', async () => {
      const assets = [
        createApiAsset({ name: 'test1.safetensors' }),
        createApiAsset({ name: 'test2.safetensors' })
      ]

      const { searchQuery, filteredAssets } = useAssetBrowser(ref(assets))

      searchQuery.value = ''
      await nextTick()

      expect(filteredAssets.value).toHaveLength(2)
    })

    it('handles no search results', async () => {
      const assets = [createApiAsset({ name: 'test.safetensors' })]

      const { searchQuery, filteredAssets } = useAssetBrowser(ref(assets))

      searchQuery.value = 'completelydifferentstring123'
      await nextTick()

      expect(filteredAssets.value).toHaveLength(0)
    })

    it('performs case-insensitive search', async () => {
      const assets = [
        createApiAsset({ name: 'RealisticVision.safetensors' }),
        createApiAsset({ name: 'anime_style.ckpt' })
      ]

      const { searchQuery, filteredAssets } = useAssetBrowser(ref(assets))

      searchQuery.value = 'REALISTIC'
      await nextTick()

      expect(filteredAssets.value.length).toBeGreaterThanOrEqual(1)
      expect(filteredAssets.value[0].name).toContain('Realistic')
    })

    it('combines fuzzy search with format filter', async () => {
      const assets = [
        createApiAsset({ name: 'my_checkpoint_model.safetensors' }),
        createApiAsset({ name: 'my_checkpoint_model.ckpt' }),
        createApiAsset({ name: 'different_lora.safetensors' })
      ]

      const { searchQuery, updateFilters, filteredAssets } = useAssetBrowser(
        ref(assets)
      )

      searchQuery.value = 'checkpoint'
      updateFilters({
        sortBy: 'name-asc',
        fileFormats: ['safetensors'],
        baseModels: [],
        ownership: 'all'
      })
      await nextTick()

      expect(filteredAssets.value.length).toBeGreaterThanOrEqual(1)
      expect(
        filteredAssets.value.every((asset) =>
          asset.name.endsWith('.safetensors')
        )
      ).toBe(true)
      expect(
        filteredAssets.value.some((asset) => asset.name.includes('checkpoint'))
      ).toBe(true)
    })

    it('combines fuzzy search with base model filter', async () => {
      const assets = [
        createApiAsset({
          name: 'realistic_sd15.safetensors',
          user_metadata: { base_model: 'SD1.5' }
        }),
        createApiAsset({
          name: 'realistic_sdxl.safetensors',
          user_metadata: { base_model: 'SDXL' }
        })
      ]

      const { searchQuery, updateFilters, filteredAssets } = useAssetBrowser(
        ref(assets)
      )

      searchQuery.value = 'realistic'
      updateFilters({
        sortBy: 'name-asc',
        fileFormats: [],
        baseModels: ['SDXL'],
        ownership: 'all'
      })
      await nextTick()

      expect(filteredAssets.value).toHaveLength(1)
      expect(filteredAssets.value[0].name).toBe('realistic_sdxl.safetensors')
    })
  })

  describe('Combined Search and Filtering', () => {
    it('applies both search and category filter', async () => {
      const assets = [
        createApiAsset({
          name: 'realistic_checkpoint.safetensors',
          tags: ['models', 'checkpoints']
        }),
        createApiAsset({
          name: 'realistic_lora.safetensors',
          tags: ['models', 'loras']
        }),
        createApiAsset({
          name: 'anime_checkpoint.safetensors',
          tags: ['models', 'checkpoints']
        })
      ]

      const { searchQuery, selectedNavItem, filteredAssets } = useAssetBrowser(
        ref(assets)
      )

      searchQuery.value = 'realistic'
      selectedNavItem.value = 'checkpoints'
      await nextTick()

      expect(filteredAssets.value).toHaveLength(1)
      expect(filteredAssets.value[0].name).toBe(
        'realistic_checkpoint.safetensors'
      )
    })
  })

  describe('Sorting', () => {
    it('sorts assets by name', async () => {
      const assets = [
        createApiAsset({ name: 'zebra.safetensors' }),
        createApiAsset({ name: 'alpha.safetensors' }),
        createApiAsset({ name: 'beta.safetensors' })
      ]

      const { updateFilters, filteredAssets } = useAssetBrowser(ref(assets))

      updateFilters({
        sortBy: 'name-asc',
        fileFormats: [],
        baseModels: [],
        ownership: 'all'
      })
      await nextTick()

      const names = filteredAssets.value.map((asset) => asset.name)
      expect(names).toEqual([
        'alpha.safetensors',
        'beta.safetensors',
        'zebra.safetensors'
      ])
    })

    it('sorts assets by creation date', async () => {
      const assets = [
        createApiAsset({ created_at: '2024-03-01T00:00:00Z' }),
        createApiAsset({ created_at: '2024-01-01T00:00:00Z' }),
        createApiAsset({ created_at: '2024-02-01T00:00:00Z' })
      ]

      const { updateFilters, filteredAssets } = useAssetBrowser(ref(assets))

      updateFilters({
        sortBy: 'recent',
        fileFormats: [],
        baseModels: [],
        ownership: 'all'
      })
      await nextTick()

      const dates = filteredAssets.value.map((asset) => asset.created_at)
      expect(dates).toEqual([
        '2024-03-01T00:00:00Z',
        '2024-02-01T00:00:00Z',
        '2024-01-01T00:00:00Z'
      ])
    })
  })

  describe('Ownership filtering', () => {
    it('filters by ownership - all', async () => {
      const assets = [
        createApiAsset({ name: 'my-model.safetensors', is_immutable: false }),
        createApiAsset({
          name: 'public-model.safetensors',
          is_immutable: true
        }),
        createApiAsset({
          name: 'another-my-model.safetensors',
          is_immutable: false
        })
      ]

      const { updateFilters, filteredAssets } = useAssetBrowser(ref(assets))

      updateFilters({
        sortBy: 'name-asc',
        fileFormats: [],
        baseModels: [],
        ownership: 'all'
      })
      await nextTick()

      expect(filteredAssets.value).toHaveLength(3)
    })

    it('filters by ownership - imported models only via nav selection', async () => {
      const assets = [
        createApiAsset({ name: 'my-model.safetensors', is_immutable: false }),
        createApiAsset({
          name: 'public-model.safetensors',
          is_immutable: true
        }),
        createApiAsset({
          name: 'another-my-model.safetensors',
          is_immutable: false
        }),
        // Need a second category so typeCategories.length > 1
        createApiAsset({
          name: 'lora.safetensors',
          is_immutable: true,
          tags: ['models', 'loras']
        })
      ]

      const { selectedNavItem, filteredAssets } = useAssetBrowser(ref(assets))

      // Selecting 'imported' nav item filters to my-models (non-immutable)
      selectedNavItem.value = 'imported'
      await nextTick()

      expect(filteredAssets.value).toHaveLength(2)
      expect(filteredAssets.value.every((asset) => !asset.is_immutable)).toBe(
        true
      )
    })

    it('shows all models when nav is "all"', async () => {
      const assets = [
        createApiAsset({ name: 'my-model.safetensors', is_immutable: false }),
        createApiAsset({
          name: 'public-model.safetensors',
          is_immutable: true
        }),
        createApiAsset({
          name: 'another-public-model.safetensors',
          is_immutable: true
        })
      ]

      const { selectedNavItem, filteredAssets } = useAssetBrowser(ref(assets))

      // Selecting 'all' nav item shows all models
      selectedNavItem.value = 'all'
      await nextTick()

      expect(filteredAssets.value).toHaveLength(3)
    })

    it('filters by ownership via filter bar - my-models', async () => {
      const assets = [
        createApiAsset({
          name: 'my-model.safetensors',
          is_immutable: false,
          tags: ['models', 'checkpoints']
        }),
        createApiAsset({
          name: 'public-model.safetensors',
          is_immutable: true,
          tags: ['models', 'checkpoints']
        }),
        createApiAsset({
          name: 'another-my-model.safetensors',
          is_immutable: false,
          tags: ['models', 'checkpoints']
        })
      ]

      const { selectedNavItem, updateFilters, filteredAssets } =
        useAssetBrowser(ref(assets))

      // Must select a specific category for ownership filter to apply
      selectedNavItem.value = 'checkpoints'
      updateFilters({
        sortBy: 'name-asc',
        fileFormats: [],
        baseModels: [],
        ownership: 'my-models'
      })
      await nextTick()

      expect(filteredAssets.value).toHaveLength(2)
      expect(filteredAssets.value.every((asset) => !asset.is_immutable)).toBe(
        true
      )
    })

    it('filters by ownership via filter bar - public-models', async () => {
      const assets = [
        createApiAsset({
          name: 'my-model.safetensors',
          is_immutable: false,
          tags: ['models', 'loras']
        }),
        createApiAsset({
          name: 'public-model.safetensors',
          is_immutable: true,
          tags: ['models', 'loras']
        }),
        createApiAsset({
          name: 'another-public-model.safetensors',
          is_immutable: true,
          tags: ['models', 'loras']
        })
      ]

      const { selectedNavItem, updateFilters, filteredAssets } =
        useAssetBrowser(ref(assets))

      // Must select a specific category for ownership filter to apply
      selectedNavItem.value = 'loras'
      updateFilters({
        sortBy: 'name-asc',
        fileFormats: [],
        baseModels: [],
        ownership: 'public-models'
      })
      await nextTick()

      expect(filteredAssets.value).toHaveLength(2)
      expect(filteredAssets.value.every((asset) => asset.is_immutable)).toBe(
        true
      )
    })

    it('nav imported selection overrides filter bar ownership', async () => {
      const assets = [
        createApiAsset({
          name: 'my-model.safetensors',
          is_immutable: false,
          tags: ['models', 'checkpoints']
        }),
        createApiAsset({
          name: 'public-model.safetensors',
          is_immutable: true,
          tags: ['models', 'checkpoints']
        }),
        // Need a second category so typeCategories.length > 1
        createApiAsset({
          name: 'lora.safetensors',
          is_immutable: true,
          tags: ['models', 'loras']
        })
      ]

      const { selectedNavItem, updateFilters, filteredAssets } =
        useAssetBrowser(ref(assets))

      // Must select a specific category for ownership filter to apply
      selectedNavItem.value = 'checkpoints'
      // Set filter bar to public-models
      updateFilters({
        sortBy: 'name-asc',
        fileFormats: [],
        baseModels: [],
        ownership: 'public-models'
      })
      await nextTick()

      expect(filteredAssets.value).toHaveLength(1)
      expect(filteredAssets.value[0].is_immutable).toBe(true)

      // Nav selection to 'imported' should override filter bar
      selectedNavItem.value = 'imported'
      await nextTick()

      expect(filteredAssets.value).toHaveLength(1)
      expect(filteredAssets.value[0].is_immutable).toBe(false)
    })
  })

  describe('Dynamic Category Extraction', () => {
    it('extracts categories from asset tags into navItems', () => {
      const assets = [
        createApiAsset({ tags: ['models', 'checkpoints'] }),
        createApiAsset({ tags: ['models', 'loras'] }),
        createApiAsset({ tags: ['models', 'checkpoints'] }) // duplicate
      ]

      const { navItems } = useAssetBrowser(ref(assets))

      // navItems includes quick filters plus a "By type" group
      expect(navItems.value).toEqual([
        { id: 'all', label: 'All Models', icon: 'icon-[lucide--list]' },
        {
          id: 'imported',
          label: 'Imported',
          icon: 'icon-[lucide--folder-input]',
          badge: undefined
        },
        {
          title: 'By type',
          collapsible: false,
          items: [
            {
              id: 'checkpoints',
              label: 'Checkpoints',
              icon: 'icon-[lucide--folder]'
            },
            { id: 'loras', label: 'Loras', icon: 'icon-[lucide--folder]' }
          ]
        }
      ])
    })

    it('handles assets with no category tag', () => {
      const assets = [
        createApiAsset({ tags: ['models'] }), // No second tag
        createApiAsset({ tags: ['models', 'vae'] })
      ]

      const { navItems } = useAssetBrowser(ref(assets))

      expect(navItems.value).toEqual([
        { id: 'all', label: 'All Models', icon: 'icon-[lucide--list]' },
        {
          id: 'imported',
          label: 'Imported',
          icon: 'icon-[lucide--folder-input]',
          badge: undefined
        },
        {
          title: 'By type',
          collapsible: false,
          items: [{ id: 'vae', label: 'Vae', icon: 'icon-[lucide--folder]' }]
        }
      ])
    })

    it('ignores non-models root tags', () => {
      const assets = [
        createApiAsset({ tags: ['input', 'images'] }),
        createApiAsset({ tags: ['models', 'checkpoints'] })
      ]

      const { navItems } = useAssetBrowser(ref(assets))

      expect(navItems.value).toEqual([
        { id: 'all', label: 'All Models', icon: 'icon-[lucide--list]' },
        {
          id: 'imported',
          label: 'Imported',
          icon: 'icon-[lucide--folder-input]',
          badge: undefined
        },
        {
          title: 'By type',
          collapsible: false,
          items: [
            {
              id: 'checkpoints',
              label: 'Checkpoints',
              icon: 'icon-[lucide--folder]'
            }
          ]
        }
      ])
    })

    it('computes content title from selected nav item', () => {
      const assets = [createApiAsset({ tags: ['models', 'checkpoints'] })]
      const { selectedNavItem, contentTitle } = useAssetBrowser(ref(assets))

      // Default
      expect(contentTitle.value).toBe('All Models')

      // Set specific category
      selectedNavItem.value = 'checkpoints'
      expect(contentTitle.value).toBe('Checkpoints')

      // Set imported
      selectedNavItem.value = 'imported'
      expect(contentTitle.value).toBe('Imported')

      // Unknown category
      selectedNavItem.value = 'unknown'
      expect(contentTitle.value).toBe('Assets')
    })

    it('ignores stale file format filter when navigating to category without that format', async () => {
      const assets = [
        createApiAsset({
          id: 'ckpt-safetensors',
          name: 'model.safetensors',
          tags: ['models', 'checkpoints']
        }),
        createApiAsset({
          id: 'lora-pt',
          name: 'lora.pt',
          tags: ['models', 'loras']
        }),
        createApiAsset({
          id: 'lora-pt-2',
          name: 'lora2.pt',
          tags: ['models', 'loras']
        })
      ]

      const { selectedNavItem, updateFilters, filteredAssets } =
        useAssetBrowser(ref(assets))

      // Select safetensors filter while viewing checkpoints
      selectedNavItem.value = 'checkpoints'
      updateFilters({
        sortBy: 'recent',
        fileFormats: ['safetensors'],
        baseModels: [],
        ownership: 'all'
      })
      await nextTick()

      expect(filteredAssets.value).toHaveLength(1)
      expect(filteredAssets.value[0].id).toBe('ckpt-safetensors')

      // Navigate to loras category which has no .safetensors files
      selectedNavItem.value = 'loras'
      await nextTick()

      // Should show all loras, not empty (stale filter should be ignored)
      expect(filteredAssets.value).toHaveLength(2)
    })

    it('ignores stale base model filter when navigating to category without that model', async () => {
      const assets = [
        createApiAsset({
          id: 'ckpt-sdxl',
          name: 'model.safetensors',
          tags: ['models', 'checkpoints'],
          user_metadata: { base_model: 'SDXL' }
        }),
        createApiAsset({
          id: 'lora-sd15',
          name: 'lora.pt',
          tags: ['models', 'loras'],
          user_metadata: { base_model: 'SD1.5' }
        })
      ]

      const { selectedNavItem, updateFilters, filteredAssets } =
        useAssetBrowser(ref(assets))

      // Select SDXL base model filter while viewing checkpoints
      selectedNavItem.value = 'checkpoints'
      updateFilters({
        sortBy: 'recent',
        fileFormats: [],
        baseModels: ['SDXL'],
        ownership: 'all'
      })
      await nextTick()

      expect(filteredAssets.value).toHaveLength(1)
      expect(filteredAssets.value[0].id).toBe('ckpt-sdxl')

      // Navigate to loras which has no SDXL models
      selectedNavItem.value = 'loras'
      await nextTick()

      // Should show all loras, not empty
      expect(filteredAssets.value).toHaveLength(1)
      expect(filteredAssets.value[0].id).toBe('lora-sd15')
    })

    it('groups models by top-level folder name', () => {
      const assets = [
        createApiAsset({
          id: 'asset-1',
          tags: ['models', 'Chatterbox/subfolder1/model1']
        }),
        createApiAsset({
          id: 'asset-2',
          tags: ['models', 'Chatterbox/subfolder2/model2']
        }),
        createApiAsset({
          id: 'asset-3',
          tags: ['models', 'Chatterbox/subfolder3/model3']
        }),
        createApiAsset({
          id: 'asset-4',
          tags: ['models', 'OtherFolder/subfolder1/model4']
        })
      ]

      const { navItems, selectedNavItem, categoryFilteredAssets } =
        useAssetBrowser(ref(assets))

      // Should group all Chatterbox subfolders under single category in the type group
      const typeGroup = navItems.value[2] as { items: { id: string }[] }
      expect(typeGroup.items.map((i) => i.id)).toEqual([
        'Chatterbox',
        'OtherFolder'
      ])

      // When selecting Chatterbox category, should include all models from its subfolders
      selectedNavItem.value = 'Chatterbox'
      expect(categoryFilteredAssets.value).toHaveLength(3)
      expect(categoryFilteredAssets.value.map((a) => a.id)).toEqual([
        'asset-1',
        'asset-2',
        'asset-3'
      ])

      // When selecting OtherFolder category, should include only its models
      selectedNavItem.value = 'OtherFolder'
      expect(categoryFilteredAssets.value).toHaveLength(1)
      expect(categoryFilteredAssets.value[0].id).toBe('asset-4')
    })
  })
})
