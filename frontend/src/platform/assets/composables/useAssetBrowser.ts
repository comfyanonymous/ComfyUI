import { computed, ref } from 'vue'
import type { Ref } from 'vue'
import { useFuse } from '@vueuse/integrations/useFuse'
import type { UseFuseOptions } from '@vueuse/integrations/useFuse'
import { storeToRefs } from 'pinia'

import { d, t } from '@/i18n'
import type {
  AssetFilterState,
  OwnershipOption
} from '@/platform/assets/types/filterTypes'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { useAssetFilterOptions } from '@/platform/assets/composables/useAssetFilterOptions'
import {
  filterByBaseModels,
  filterByCategory,
  filterByFileFormats,
  filterByOwnership
} from '@/platform/assets/utils/assetFilterUtils'
import {
  getAssetBaseModels,
  getAssetFilename
} from '@/platform/assets/utils/assetMetadataUtils'
import { sortAssets } from '@/platform/assets/utils/assetSortUtils'
import { useAssetDownloadStore } from '@/stores/assetDownloadStore'
import type { NavGroupData, NavItemData } from '@/types/navTypes'

type NavId = 'all' | 'imported' | (string & {})

type AssetBadge = {
  label: string
  type: 'type' | 'base' | 'size'
}

// Display properties for transformed assets
export interface AssetDisplayItem extends AssetItem {
  secondaryText: string
  badges: AssetBadge[]
  stats: {
    formattedDate?: string
    downloadCount?: string
    stars?: string
  }
}

/**
 * Asset Browser composable
 * Manages search, filtering, asset transformation and selection logic
 */
export function useAssetBrowser(
  assetsSource: Ref<AssetItem[] | undefined> = ref<AssetItem[] | undefined>([])
) {
  const assets = computed<AssetItem[]>(() => assetsSource.value ?? [])
  const assetDownloadStore = useAssetDownloadStore()
  const { sessionDownloadCount } = storeToRefs(assetDownloadStore)

  // State
  const searchQuery = ref('')
  const selectedNavItem = ref<NavId>('all')
  const filters = ref<AssetFilterState>({
    sortBy: 'recent',
    fileFormats: [],
    baseModels: [],
    ownership: 'all'
  })

  const selectedOwnership = computed<OwnershipOption>(() => {
    if (typeCategories.value.length <= 1) return filters.value.ownership
    if (selectedNavItem.value === 'imported') return 'my-models'
    if (selectedNavItem.value === 'all') return 'all'
    return filters.value.ownership
  })

  const selectedCategory = computed(() => {
    if (
      selectedNavItem.value === 'all' ||
      selectedNavItem.value === 'imported'
    ) {
      return 'all'
    }
    return selectedNavItem.value
  })

  // Transform API asset to display asset
  function transformAssetForDisplay(asset: AssetItem): AssetDisplayItem {
    const secondaryText = getAssetFilename(asset)

    const badges: AssetBadge[] = []

    const typeTag = asset.tags.find((tag) => tag !== 'models')
    // Type badge from non-root tag
    if (typeTag) {
      // Remove category prefix from badge label (e.g. "checkpoint/model" → "model")
      const badgeLabel = typeTag.includes('/')
        ? typeTag.substring(typeTag.indexOf('/') + 1)
        : typeTag

      badges.push({ label: badgeLabel, type: 'type' })
    }

    // Base model badges from metadata
    const baseModels = getAssetBaseModels(asset)
    for (const model of baseModels) {
      badges.push({ label: model, type: 'base' })
    }

    // Create display stats from API data
    const stats = {
      formattedDate: asset.created_at
        ? d(new Date(asset.created_at), { dateStyle: 'short' })
        : undefined,
      downloadCount: undefined, // Not available in API
      stars: undefined // Not available in API
    }

    return {
      ...asset,
      secondaryText,
      badges,
      stats
    }
  }

  const typeCategories = computed<NavItemData[]>(() => {
    const categories = assets.value
      .filter((asset) => asset.tags[0] === 'models')
      .map((asset) => asset.tags[1])
      .filter((tag): tag is string => typeof tag === 'string' && tag.length > 0)
      .map((tag) => tag.split('/')[0])

    return Array.from(new Set(categories))
      .sort()
      .map((category) => ({
        id: category,
        label: category.charAt(0).toUpperCase() + category.slice(1),
        icon: 'icon-[lucide--folder]'
      }))
  })

  const navItems = computed<(NavItemData | NavGroupData)[]>(() => {
    const quickFilters: NavItemData[] = [
      {
        id: 'all',
        label: t('assetBrowser.allModels'),
        icon: 'icon-[lucide--list]'
      },
      {
        id: 'imported',
        label: t('assetBrowser.imported'),
        icon: 'icon-[lucide--folder-input]',
        badge:
          sessionDownloadCount.value > 0
            ? sessionDownloadCount.value
            : undefined
      }
    ]

    if (typeCategories.value.length === 0) {
      return quickFilters
    }

    return [
      ...quickFilters,
      {
        title: t('assetBrowser.byType'),
        items: typeCategories.value,
        collapsible: false
      }
    ]
  })

  const isImportedSelected = computed(
    () => selectedNavItem.value === 'imported'
  )

  // Compute content title from selected nav item
  const contentTitle = computed(() => {
    if (selectedNavItem.value === 'all') {
      return t('assetBrowser.allModels')
    }
    if (selectedNavItem.value === 'imported') {
      return t('assetBrowser.imported')
    }

    const category = typeCategories.value.find(
      (cat) => cat.id === selectedNavItem.value
    )
    return category?.label || t('assetBrowser.assets')
  })

  // Category-filtered assets for filter options (before search/format/base model filters)
  const categoryFilteredAssets = computed(() => {
    return assets.value.filter(filterByCategory(selectedCategory.value))
  })

  const { availableFileFormats, availableBaseModels } = useAssetFilterOptions(
    categoryFilteredAssets
  )

  const activeFileFormats = computed(() =>
    filters.value.fileFormats.filter((f) =>
      availableFileFormats.value.some((opt) => opt.value === f)
    )
  )

  const activeBaseModels = computed(() =>
    filters.value.baseModels.filter((m) =>
      availableBaseModels.value.some((opt) => opt.value === m)
    )
  )

  const fuseOptions: UseFuseOptions<AssetItem> = {
    fuseOptions: {
      keys: [
        { name: 'name', weight: 0.4 },
        { name: 'tags', weight: 0.3 },
        { name: 'user_metadata.name', weight: 0.4 },
        { name: 'user_metadata.additional_tags', weight: 0.3 },
        { name: 'user_metadata.trained_words', weight: 0.3 },
        { name: 'user_metadata.user_description', weight: 0.3 },
        { name: 'metadata.name', weight: 0.4 },
        { name: 'metadata.trained_words', weight: 0.3 }
      ],
      threshold: 0.4, // Higher threshold for typo tolerance (0.0 = exact, 1.0 = match all)
      ignoreLocation: true, // Search anywhere in the string, not just at the beginning
      includeScore: true
    },
    matchAllWhenSearchEmpty: true
  }

  const { results: fuseResults } = useFuse(
    searchQuery,
    categoryFilteredAssets,
    fuseOptions
  )

  const searchFiltered = computed(() =>
    fuseResults.value.map((result) => result.item)
  )

  const filteredAssets = computed(() => {
    const filtered = searchFiltered.value
      .filter(filterByFileFormats(activeFileFormats.value))
      .filter(filterByBaseModels(activeBaseModels.value))
      .filter(filterByOwnership(selectedOwnership.value))

    const sortedAssets = sortAssets(filtered, filters.value.sortBy)

    // Transform to display format
    return sortedAssets.map(transformAssetForDisplay)
  })

  function updateFilters(newFilters: AssetFilterState) {
    filters.value = { ...newFilters }
  }

  return {
    searchQuery,
    selectedNavItem,
    selectedCategory,
    navItems,
    contentTitle,
    categoryFilteredAssets,
    filteredAssets,
    isImportedSelected,
    updateFilters
  }
}
