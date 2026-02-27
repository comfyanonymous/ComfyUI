import type { Mock } from 'vitest'
import { liteClient as algoliasearch } from 'algoliasearch/dist/lite/builds/browser'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { components } from '@/types/comfyRegistryTypes'
import { useAlgoliaSearchProvider } from '@/services/providers/algoliaSearchProvider'
import { SortableAlgoliaField } from '@/workbench/extensions/manager/types/comfyManagerTypes'

type RegistryNodePack = components['schemas']['Node']

type GlobalWithAlgolia = typeof globalThis & {
  __ALGOLIA_APP_ID__: string
  __ALGOLIA_API_KEY__: string
}

// Mock global Algolia constants
const globalWithAlgolia = globalThis as GlobalWithAlgolia
globalWithAlgolia.__ALGOLIA_APP_ID__ = 'test-app-id'
globalWithAlgolia.__ALGOLIA_API_KEY__ = 'test-api-key'

// Mock algoliasearch
vi.mock('algoliasearch/dist/lite/builds/browser', () => ({
  liteClient: vi.fn()
}))

interface MockSearchClient {
  search: Mock
}

describe('useAlgoliaSearchProvider', () => {
  let mockSearchClient: MockSearchClient

  beforeEach(() => {
    vi.clearAllMocks()

    // Create mock search client
    mockSearchClient = {
      search: vi.fn()
    }

    vi.mocked(algoliasearch).mockReturnValue(
      mockSearchClient as Partial<
        ReturnType<typeof algoliasearch>
      > as ReturnType<typeof algoliasearch>
    )
  })

  afterEach(() => {
    // Clear the module-level cache between tests
    const provider = useAlgoliaSearchProvider()
    provider.clearSearchCache()
  })

  describe('searchPacks', () => {
    it('should search for packs and convert results', async () => {
      const mockAlgoliaResults = {
        results: [
          {
            hits: [
              {
                objectID: 'algolia-1',
                id: 'pack-1',
                name: 'Test Pack',
                description: 'A test pack',
                publisher_id: 'publisher-1',
                total_install: 500,
                create_time: '2024-01-01T00:00:00Z',
                update_time: '2024-01-15T00:00:00Z',
                repository_url: 'https://github.com/test/pack',
                license: 'MIT',
                status: 'active',
                latest_version: '1.0.0',
                latest_version_status: 'published',
                icon_url: 'https://example.com/icon.png',
                comfy_nodes: ['LoadImage', 'SaveImage']
              }
            ]
          },
          { hits: [] } // Query suggestions
        ]
      }

      mockSearchClient.search.mockResolvedValue(mockAlgoliaResults)

      const provider = useAlgoliaSearchProvider()
      const result = await provider.searchPacks('test', {
        pageSize: 10,
        pageNumber: 0
      })

      expect(mockSearchClient.search).toHaveBeenCalledWith({
        requests: [
          {
            query: 'test',
            indexName: 'nodes_index',
            attributesToRetrieve: expect.any(Array),
            hitsPerPage: 10,
            page: 0
          },
          {
            query: 'test',
            indexName: 'nodes_index_query_suggestions'
          }
        ],
        strategy: 'none'
      })

      expect(result.nodePacks).toHaveLength(1)
      expect(result.nodePacks[0]).toEqual({
        id: 'pack-1',
        name: 'Test Pack',
        description: 'A test pack',
        repository: 'https://github.com/test/pack',
        license: 'MIT',
        downloads: 500,
        status: 'active',
        icon: 'https://example.com/icon.png',
        latest_version: {
          version: '1.0.0',
          createdAt: '2024-01-15T00:00:00Z',
          status: 'published',
          comfy_node_extract_status: undefined
        },
        publisher: {
          id: 'publisher-1',
          name: 'publisher-1'
        },
        created_at: '2024-01-01T00:00:00Z',
        comfy_nodes: ['LoadImage', 'SaveImage'],
        category: undefined,
        author: undefined,
        tags: undefined,
        github_stars: undefined,
        supported_os: undefined,
        supported_comfyui_version: undefined,
        supported_comfyui_frontend_version: undefined,
        supported_accelerators: undefined,
        banner_url: undefined
      })
    })

    it('should include query suggestions when query is long enough', async () => {
      const mockAlgoliaResults = {
        results: [
          { hits: [] }, // Main results
          {
            hits: [
              { query: 'test query', popularity: 10 },
              { query: 'test pack', popularity: 5 }
            ]
          }
        ]
      }

      mockSearchClient.search.mockResolvedValue(mockAlgoliaResults)

      const provider = useAlgoliaSearchProvider()
      const result = await provider.searchPacks('test', {
        pageSize: 10,
        pageNumber: 0
      })

      // Should make 2 requests (main + suggestions)
      expect(mockSearchClient.search).toHaveBeenCalledWith({
        requests: [
          expect.objectContaining({ indexName: 'nodes_index' }),
          expect.objectContaining({
            indexName: 'nodes_index_query_suggestions'
          })
        ],
        strategy: 'none'
      })

      expect(result.querySuggestions).toEqual([
        { query: 'test query', popularity: 10 },
        { query: 'test pack', popularity: 5 }
      ])
    })

    it('should not query suggestions for short queries', async () => {
      mockSearchClient.search.mockResolvedValue({
        results: [{ hits: [] }]
      })

      const provider = useAlgoliaSearchProvider()
      await provider.searchPacks('a', {
        pageSize: 10,
        pageNumber: 0
      })

      // Should only make 1 request (no suggestions)
      expect(mockSearchClient.search).toHaveBeenCalledWith({
        requests: [expect.objectContaining({ indexName: 'nodes_index' })],
        strategy: 'none'
      })
    })

    it('should cache search results', async () => {
      mockSearchClient.search.mockResolvedValue({
        results: [{ hits: [] }, { hits: [] }]
      })

      const provider = useAlgoliaSearchProvider()
      const params = { pageSize: 10, pageNumber: 0 }

      // First call
      await provider.searchPacks('test', params)
      expect(mockSearchClient.search).toHaveBeenCalledTimes(1)

      // Second call with same params should use cache
      await provider.searchPacks('test', params)
      expect(mockSearchClient.search).toHaveBeenCalledTimes(1)

      // Different params should make new request
      await provider.searchPacks('test', { ...params, pageNumber: 1 })
      expect(mockSearchClient.search).toHaveBeenCalledTimes(2)
    })

    it('should handle missing objectID by using id field', async () => {
      const mockAlgoliaResults = {
        results: [
          {
            hits: [
              {
                id: 'pack-id-only',
                name: 'Pack without objectID',
                // ... other required fields
                publisher_id: 'pub',
                total_install: 0,
                comfy_nodes: []
              }
            ]
          },
          { hits: [] }
        ]
      }

      mockSearchClient.search.mockResolvedValue(mockAlgoliaResults)

      const provider = useAlgoliaSearchProvider()
      const result = await provider.searchPacks('test', {
        pageSize: 10,
        pageNumber: 0
      })

      expect(result.nodePacks[0].id).toBe('pack-id-only')
    })
  })

  describe('clearSearchCache', () => {
    it('should clear the cache', async () => {
      mockSearchClient.search.mockResolvedValue({
        results: [{ hits: [] }, { hits: [] }]
      })

      const provider = useAlgoliaSearchProvider()
      const params = { pageSize: 10, pageNumber: 0 }

      // Populate cache
      await provider.searchPacks('test', params)
      expect(mockSearchClient.search).toHaveBeenCalledTimes(1)

      // Clear cache
      provider.clearSearchCache()

      // Same search should hit API again
      await provider.searchPacks('test', params)
      expect(mockSearchClient.search).toHaveBeenCalledTimes(2)
    })
  })

  describe('getSortValue', () => {
    const testPack: Partial<RegistryNodePack> = {
      id: '1',
      name: 'Test Pack',
      downloads: 100,
      publisher: { id: 'pub1', name: 'Publisher One' },
      latest_version: {
        version: '1.0.0',
        createdAt: '2024-01-15T10:00:00Z'
      },
      created_at: '2024-01-01T10:00:00Z'
    }

    it('should return correct values for each sort field', () => {
      const provider = useAlgoliaSearchProvider()

      expect(
        provider.getSortValue(testPack, SortableAlgoliaField.Downloads)
      ).toBe(100)
      expect(provider.getSortValue(testPack, SortableAlgoliaField.Name)).toBe(
        'Test Pack'
      )
      expect(
        provider.getSortValue(testPack, SortableAlgoliaField.Publisher)
      ).toBe('Publisher One')

      const createdTimestamp = new Date('2024-01-01T10:00:00Z').getTime()
      expect(
        provider.getSortValue(
          testPack as RegistryNodePack,
          SortableAlgoliaField.Created
        )
      ).toBe(createdTimestamp)

      const updatedTimestamp = new Date('2024-01-15T10:00:00Z').getTime()
      expect(
        provider.getSortValue(testPack, SortableAlgoliaField.Updated)
      ).toBe(updatedTimestamp)
    })

    it('should handle missing values', () => {
      const incompletePack: Partial<RegistryNodePack> = {
        id: '1',
        name: 'Incomplete'
      }
      const provider = useAlgoliaSearchProvider()

      expect(
        provider.getSortValue(
          incompletePack as RegistryNodePack,
          SortableAlgoliaField.Downloads
        )
      ).toBe(0)
      expect(
        provider.getSortValue(
          incompletePack as RegistryNodePack,
          SortableAlgoliaField.Publisher
        )
      ).toBe('')
      expect(
        provider.getSortValue(
          incompletePack as RegistryNodePack,
          SortableAlgoliaField.Created
        )
      ).toBe(0)
      expect(
        provider.getSortValue(
          incompletePack as RegistryNodePack,
          SortableAlgoliaField.Updated
        )
      ).toBe(0)
    })
  })

  describe('getSortableFields', () => {
    it('should return all Algolia sort fields', () => {
      const provider = useAlgoliaSearchProvider()
      const fields = provider.getSortableFields()

      expect(fields).toEqual([
        {
          id: SortableAlgoliaField.Downloads,
          label: 'Downloads',
          direction: 'desc'
        },
        {
          id: SortableAlgoliaField.Created,
          label: 'Created',
          direction: 'desc'
        },
        {
          id: SortableAlgoliaField.Updated,
          label: 'Updated',
          direction: 'desc'
        },
        {
          id: SortableAlgoliaField.Publisher,
          label: 'Publisher',
          direction: 'asc'
        },
        { id: SortableAlgoliaField.Name, label: 'Name', direction: 'asc' }
      ])
    })
  })

  describe('memoization', () => {
    it('should memoize toRegistryPack conversions', async () => {
      const mockHit = {
        objectID: 'algolia-1',
        id: 'pack-1',
        name: 'Test Pack',
        publisher_id: 'pub1',
        total_install: 100,
        comfy_nodes: []
      }

      mockSearchClient.search.mockResolvedValue({
        results: [
          { hits: [mockHit, mockHit, mockHit] }, // Same object 3 times
          { hits: [] }
        ]
      })

      const provider = useAlgoliaSearchProvider()
      const result = await provider.searchPacks('test', {
        pageSize: 10,
        pageNumber: 0
      })

      // All 3 results should be the same object reference due to memoization
      expect(result.nodePacks[0]).toBe(result.nodePacks[1])
      expect(result.nodePacks[1]).toBe(result.nodePacks[2])
    })
  })
})
