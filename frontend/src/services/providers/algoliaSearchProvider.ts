import QuickLRU from '@alloc/quick-lru'
import type {
  SearchQuery,
  SearchResponse
} from 'algoliasearch/dist/lite/browser'
import { liteClient as algoliasearch } from 'algoliasearch/dist/lite/builds/browser'
import { memoize, omit } from 'es-toolkit/compat'

import {
  MIN_CHARS_FOR_SUGGESTIONS_ALGOLIA,
  SEARCH_CACHE_MAX_SIZE
} from '@/constants/searchConstants'
import type {
  AlgoliaNodePack,
  NodesIndexSuggestion,
  SearchAttribute,
  SearchNodePacksParams
} from '@/types/algoliaTypes'
import type { components } from '@/types/comfyRegistryTypes'
import type {
  NodePackSearchProvider,
  SearchPacksResult,
  SortableField
} from '@/types/searchServiceTypes'
import { paramsToCacheKey } from '@/utils/formatUtil'
import { SortableAlgoliaField } from '@/workbench/extensions/manager/types/comfyManagerTypes'

type RegistryNodePack = components['schemas']['Node']

const RETRIEVE_ATTRIBUTES: SearchAttribute[] = [
  'comfy_nodes',
  'name',
  'description',
  'latest_version',
  'status',
  'publisher_id',
  'total_install',
  'create_time',
  'update_time',
  'license',
  'repository_url',
  'latest_version_status',
  'comfy_node_extract_status',
  'id',
  'icon_url',
  'github_stars',
  'supported_os',
  'supported_comfyui_version',
  'supported_comfyui_frontend_version',
  'supported_accelerators',
  'banner_url'
]

const searchPacksCache = new QuickLRU<string, SearchPacksResult>({
  maxSize: SEARCH_CACHE_MAX_SIZE
})

const toRegistryLatestVersion = (
  algoliaNode: AlgoliaNodePack
): RegistryNodePack['latest_version'] => {
  return {
    version: algoliaNode.latest_version,
    createdAt: algoliaNode.update_time,
    status: algoliaNode.latest_version_status,
    comfy_node_extract_status:
      algoliaNode.comfy_node_extract_status ?? undefined
  }
}

const toRegistryPublisher = (
  algoliaNode: AlgoliaNodePack
): RegistryNodePack['publisher'] => {
  return {
    id: algoliaNode.publisher_id,
    name: algoliaNode.publisher_id
  }
}

/**
 * Convert from node pack in Algolia format to Comfy Registry format
 */
const toRegistryPack = memoize(
  (
    algoliaNode: AlgoliaNodePack
  ): RegistryNodePack & { comfy_nodes: string[] } => {
    return {
      id: algoliaNode.id ?? algoliaNode.objectID,
      name: algoliaNode.name,
      description: algoliaNode.description,
      repository: algoliaNode.repository_url,
      license: algoliaNode.license,
      downloads: algoliaNode.total_install,
      status: algoliaNode.status,
      icon: algoliaNode.icon_url,
      latest_version: toRegistryLatestVersion(algoliaNode),
      publisher: toRegistryPublisher(algoliaNode),
      created_at: algoliaNode.create_time,
      category: algoliaNode.category,
      author: algoliaNode.author,
      tags: algoliaNode.tags,
      github_stars: algoliaNode.github_stars,
      supported_os: algoliaNode.supported_os,
      supported_comfyui_version: algoliaNode.supported_comfyui_version,
      supported_comfyui_frontend_version:
        algoliaNode.supported_comfyui_frontend_version,
      supported_accelerators: algoliaNode.supported_accelerators,
      banner_url: algoliaNode.banner_url,
      comfy_nodes: algoliaNode.comfy_nodes
    }
  },
  (algoliaNode: AlgoliaNodePack) => algoliaNode.id
)

export const useAlgoliaSearchProvider = (): NodePackSearchProvider => {
  const searchClient = algoliasearch(__ALGOLIA_APP_ID__, __ALGOLIA_API_KEY__)

  /**
   * Search for node packs in Algolia (internal method)
   */
  const searchPacksInternal = async (
    query: string,
    params: SearchNodePacksParams
  ): Promise<SearchPacksResult> => {
    const { pageSize, pageNumber } = params
    const rest = omit(params, ['pageSize', 'pageNumber'])

    const requests: SearchQuery[] = [
      {
        query,
        indexName: 'nodes_index',
        attributesToRetrieve: RETRIEVE_ATTRIBUTES,
        ...rest,
        hitsPerPage: pageSize,
        page: pageNumber
      }
    ]

    const shouldQuerySuggestions =
      query.length >= MIN_CHARS_FOR_SUGGESTIONS_ALGOLIA

    // If the query is long enough, also query the suggestions index
    if (shouldQuerySuggestions) {
      requests.push({
        indexName: 'nodes_index_query_suggestions',
        query
      })
    }

    const { results } = await searchClient.search<
      AlgoliaNodePack | NodesIndexSuggestion
    >({
      requests,
      strategy: 'none'
    })

    const [nodePacks, querySuggestions = { hits: [] }] = results as [
      SearchResponse<AlgoliaNodePack>,
      SearchResponse<NodesIndexSuggestion>
    ]

    // Convert Algolia hits to RegistryNodePack format
    const registryPacks = nodePacks.hits.map(toRegistryPack)

    // Extract query suggestions from search results
    const suggestions = querySuggestions.hits.map((suggestion) => ({
      query: suggestion.query,
      popularity: suggestion.popularity
    }))

    return {
      nodePacks: registryPacks,
      querySuggestions: suggestions
    }
  }

  /**
   * Search for node packs in Algolia with caching.
   */
  const searchPacks = async (
    query: string,
    params: SearchNodePacksParams
  ): Promise<SearchPacksResult> => {
    const cacheKey = paramsToCacheKey({ query, ...params })
    const cachedResult = searchPacksCache.get(cacheKey)
    if (cachedResult !== undefined) return cachedResult

    const result = await searchPacksInternal(query, params)
    searchPacksCache.set(cacheKey, result)
    return result
  }

  const clearSearchCache = () => {
    searchPacksCache.clear()
  }

  const getSortValue = (
    pack: RegistryNodePack,
    sortField: string
  ): string | number => {
    // For Algolia, we rely on the default sorting behavior
    // The results are already sorted by the index configuration
    // This is mainly used for re-sorting after results are fetched
    switch (sortField) {
      case SortableAlgoliaField.Downloads:
        return pack.downloads ?? 0
      case SortableAlgoliaField.Created: {
        const createTime = pack.created_at
        return createTime ? new Date(createTime).getTime() : 0
      }
      case SortableAlgoliaField.Updated:
        return pack.latest_version?.createdAt
          ? new Date(pack.latest_version.createdAt).getTime()
          : 0
      case SortableAlgoliaField.Publisher:
        return pack.publisher?.name ?? ''
      case SortableAlgoliaField.Name:
        return pack.name ?? ''
      default:
        return 0
    }
  }

  const getSortableFields = (): SortableField[] => {
    return [
      {
        id: SortableAlgoliaField.Downloads,
        label: 'Downloads',
        direction: 'desc'
      },
      { id: SortableAlgoliaField.Created, label: 'Created', direction: 'desc' },
      { id: SortableAlgoliaField.Updated, label: 'Updated', direction: 'desc' },
      {
        id: SortableAlgoliaField.Publisher,
        label: 'Publisher',
        direction: 'asc'
      },
      { id: SortableAlgoliaField.Name, label: 'Name', direction: 'asc' }
    ]
  }

  return {
    searchPacks,
    clearSearchCache,
    getSortValue,
    getSortableFields
  }
}
