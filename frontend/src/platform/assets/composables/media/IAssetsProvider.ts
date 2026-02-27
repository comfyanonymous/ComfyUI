import type { Ref } from 'vue'

import type { AssetItem } from '@/platform/assets/schemas/assetSchema'

/**
 * Interface for media assets providers
 * Defines the common API for both cloud and internal file implementations
 */
export interface IAssetsProvider {
  /** Current media assets */
  media: Ref<AssetItem[]>

  /** Loading state indicator */
  loading: Ref<boolean>

  /** Error state */
  error: Ref<unknown>

  /**
   * Fetch list of media assets
   * @returns Promise resolving to array of AssetItem
   */
  fetchMediaList: () => Promise<AssetItem[]>

  /**
   * Refresh the media list (alias for fetchMediaList)
   */
  refresh: () => Promise<AssetItem[]>

  /**
   * Load more items (for pagination)
   */
  loadMore: () => Promise<void>

  /**
   * Whether there are more items to load
   */
  hasMore: Ref<boolean>

  /**
   * Whether currently loading more items
   */
  isLoadingMore: Ref<boolean>
}
