import { useAlgoliaSearchProvider } from '@/services/providers/algoliaSearchProvider'
import { useComfyRegistrySearchProvider } from '@/services/providers/registrySearchProvider'
import type { SearchNodePacksParams } from '@/types/algoliaTypes'
import type { components } from '@/types/comfyRegistryTypes'
import type {
  NodePackSearchProvider,
  SearchPacksResult
} from '@/types/searchServiceTypes'

type RegistryNodePack = components['schemas']['Node']

interface ProviderState {
  provider: NodePackSearchProvider
  name: string
  isHealthy: boolean
  lastError?: Error
  lastAttempt?: Date
  consecutiveFailures: number
}

const CIRCUIT_BREAKER_THRESHOLD = 3 // Number of failures before circuit opens
const CIRCUIT_BREAKER_TIMEOUT = 60000 // 1 minute before retry

/**
 * API Gateway for registry search providers with circuit breaker pattern.
 * Acts as a single entry point that routes search requests to appropriate providers
 * and handles failures gracefully by falling back to alternative providers.
 *
 * Implements:
 * - Gateway pattern: Single entry point for all search requests
 * - Circuit breaker: Prevents repeated calls to failed services
 * - Automatic failover: Cascades through providers on failure
 */
export const useRegistrySearchGateway = (): NodePackSearchProvider => {
  const providers: ProviderState[] = []
  let activeProviderIndex = 0

  // Initialize providers in priority order
  try {
    providers.push({
      provider: useAlgoliaSearchProvider(),
      name: 'Algolia',
      isHealthy: true,
      consecutiveFailures: 0
    })
  } catch (error) {
    console.warn('Failed to initialize Algolia provider:', error)
  }

  providers.push({
    provider: useComfyRegistrySearchProvider(),
    name: 'ComfyRegistry',
    isHealthy: true,
    consecutiveFailures: 0
  })

  // TODO: Add an "offline" provider that operates on a local cache of the registry.

  /**
   * Check if a provider's circuit breaker should be closed (available to try)
   */
  const isCircuitClosed = (providerState: ProviderState): boolean => {
    if (providerState.consecutiveFailures < CIRCUIT_BREAKER_THRESHOLD) {
      return true
    }

    // Check if enough time has passed to retry
    if (providerState.lastAttempt) {
      const timeSinceLastAttempt =
        Date.now() - providerState.lastAttempt.getTime()
      if (timeSinceLastAttempt > CIRCUIT_BREAKER_TIMEOUT) {
        return true
      }
    }

    return false
  }

  /**
   * Record a successful call to a provider
   */
  const recordSuccess = (providerState: ProviderState) => {
    providerState.isHealthy = true
    providerState.consecutiveFailures = 0
    providerState.lastError = undefined
  }

  /**
   * Record a failed call to a provider
   */
  const recordFailure = (providerState: ProviderState, error: Error) => {
    providerState.consecutiveFailures++
    providerState.lastError = error
    providerState.lastAttempt = new Date()

    if (providerState.consecutiveFailures >= CIRCUIT_BREAKER_THRESHOLD) {
      providerState.isHealthy = false
      console.warn(
        `${providerState.name} provider circuit breaker opened after ${providerState.consecutiveFailures} failures`
      )
    }
  }

  /**
   * Get the currently active provider based on circuit breaker states
   */
  const getActiveProvider = (): NodePackSearchProvider => {
    // First, try to use the current active provider if it's healthy
    const currentProvider = providers[activeProviderIndex]
    if (currentProvider && isCircuitClosed(currentProvider)) {
      return currentProvider.provider
    }

    // Otherwise, find the first healthy provider
    for (let i = 0; i < providers.length; i++) {
      const providerState = providers[i]
      if (isCircuitClosed(providerState)) {
        activeProviderIndex = i
        return providerState.provider
      }
    }

    throw new Error('No available search providers')
  }

  /**
   * Update the active provider index after a failure.
   * Move to the next provider if available.
   */
  const updateActiveProviderOnFailure = () => {
    if (activeProviderIndex < providers.length - 1) {
      activeProviderIndex++
    }
  }

  /**
   * Search for node packs.
   */
  const searchPacks = async (
    query: string,
    params: SearchNodePacksParams
  ): Promise<SearchPacksResult> => {
    let lastError: Error | null = null

    // Start with the current active provider
    for (let attempts = 0; attempts < providers.length; attempts++) {
      try {
        const provider = getActiveProvider()
        const providerState = providers[activeProviderIndex]

        const result = await provider.searchPacks(query, params)
        recordSuccess(providerState)
        return result
      } catch (error) {
        lastError = error as Error
        const providerState = providers[activeProviderIndex]
        recordFailure(providerState, lastError)
        console.warn(
          `${providerState.name} search provider failed (${providerState.consecutiveFailures} failures):`,
          error
        )

        // Try the next provider
        updateActiveProviderOnFailure()
      }
    }

    // If we get here, all providers failed
    throw new Error(
      `All search providers failed. Last error: ${lastError?.message || 'Unknown error'}`
    )
  }

  /**
   * Clear the search cache for all providers that implement it.
   */
  const clearSearchCache = () => {
    for (const providerState of providers) {
      try {
        providerState.provider.clearSearchCache()
      } catch (error) {
        console.warn(
          `Failed to clear cache for ${providerState.name} provider:`,
          error
        )
      }
    }
  }

  /**
   * Get the sort value for a pack.
   * @example
   * const pack = {
   *   id: '123',
   *   name: 'Test Pack',
   *   downloads: 100
   * }
   * const sortValue = getSortValue(pack, 'downloads')
   * console.log(sortValue) // 100
   */
  const getSortValue = (
    pack: RegistryNodePack,
    sortField: string
  ): string | number => {
    return getActiveProvider().getSortValue(pack, sortField)
  }

  /**
   * Get the sortable fields for the active provider.
   * @example
   * const sortableFields = getSortableFields()
   * console.log(sortableFields) // ['downloads', 'created', 'updated', 'publisher', 'name']
   */
  const getSortableFields = () => {
    return getActiveProvider().getSortableFields()
  }

  return {
    searchPacks,
    clearSearchCache,
    getSortValue,
    getSortableFields
  }
}
