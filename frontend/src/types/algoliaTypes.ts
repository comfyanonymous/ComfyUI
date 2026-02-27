import type { BaseSearchParamsWithoutQuery } from 'algoliasearch/dist/lite/browser'

import type { components } from '@/types/comfyRegistryTypes'

type SafeNestedProperty<
  T,
  K1 extends keyof T,
  K2 extends keyof NonNullable<T[K1]>
> = T[K1] extends undefined | null ? undefined : NonNullable<T[K1]>[K2]

type RegistryNodePack = components['schemas']['Node']

/**
 * Node pack record after it has been mapped to Algolia index format.
 * @see https://github.com/Comfy-Org/comfy-api/blob/main/mapper/algolia.go
 */
export interface AlgoliaNodePack {
  objectID: RegistryNodePack['id']
  name: RegistryNodePack['name']
  publisher_id: SafeNestedProperty<RegistryNodePack, 'publisher', 'id'>
  description: RegistryNodePack['description']
  comfy_nodes: string[]
  total_install: RegistryNodePack['downloads']
  id: RegistryNodePack['id']
  create_time: string
  update_time: SafeNestedProperty<
    RegistryNodePack,
    'latest_version',
    'createdAt'
  >
  license: RegistryNodePack['license']
  repository_url: RegistryNodePack['repository']
  status: RegistryNodePack['status']
  latest_version: SafeNestedProperty<
    RegistryNodePack,
    'latest_version',
    'version'
  >
  latest_version_status: SafeNestedProperty<
    RegistryNodePack,
    'latest_version',
    'status'
  >
  comfy_node_extract_status: SafeNestedProperty<
    RegistryNodePack,
    'latest_version',
    'comfy_node_extract_status'
  >
  icon_url: RegistryNodePack['icon']
  category: RegistryNodePack['category']
  author: RegistryNodePack['author']
  tags: RegistryNodePack['tags']
  github_stars: RegistryNodePack['github_stars']
  supported_os: RegistryNodePack['supported_os']
  supported_comfyui_version: RegistryNodePack['supported_comfyui_version']
  supported_comfyui_frontend_version: RegistryNodePack['supported_comfyui_frontend_version']
  supported_accelerators: RegistryNodePack['supported_accelerators']
  banner_url: RegistryNodePack['banner_url']
}

/**
 * An attribute that can be used to search the Algolia index by.
 */
export type SearchAttribute = keyof AlgoliaNodePack

/**
 * Suggestion for a search query (autocomplete).
 */
export interface NodesIndexSuggestion {
  nb_words: number
  nodes_index: {
    exact_nb_hits: number
    facets: {
      exact_matches: Record<string, number>
      analytics: Record<string, unknown>
    }
  }
  objectID: RegistryNodePack['id']
  popularity: number
  query: string
}

/**
 * Parameters for searching the Algolia index.
 */
export type SearchNodePacksParams = BaseSearchParamsWithoutQuery & {
  pageSize: number
  pageNumber: number
  restrictSearchableAttributes?: SearchAttribute[]
}
