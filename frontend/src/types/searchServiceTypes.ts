import type { SearchNodePacksParams } from '@/types/algoliaTypes'
import type { components } from '@/types/comfyRegistryTypes'

type RegistryNodePack = components['schemas']['Node']

/**
 * Search mode for filtering results
 */
export type SearchMode = 'nodes' | 'packs'
export type QuerySuggestion = {
  query: string
  popularity: number
}

export interface SearchPacksResult {
  nodePacks: RegistryNodePack[]
  querySuggestions: QuerySuggestion[]
}

export interface SortableField<T = string> {
  id: T
  label: string
  direction: 'asc' | 'desc'
}

export interface NodePackSearchProvider {
  /**
   * Search for node packs
   */
  searchPacks(
    query: string,
    params: SearchNodePacksParams
  ): Promise<SearchPacksResult>

  /**
   * Clear the search cache
   */
  clearSearchCache(): void

  /**
   * Get the sort value for a pack based on the sort field
   */
  getSortValue(pack: RegistryNodePack, sortField: string): string | number

  /**
   * Get the list of sortable fields supported by this provider
   */
  getSortableFields(): SortableField[]
}
