import { t } from '@/i18n'
import type { AssetSortOption } from '@/platform/assets/types/filterTypes'
import { sortAssets } from '@/platform/assets/utils/assetSortUtils'

import type { FormDropdownItem, SortOption } from './types'

export async function defaultSearcher(
  query: string,
  items: FormDropdownItem[]
) {
  if (query.trim() === '') return items
  const words = query.trim().toLowerCase().split(' ')
  return items.filter((item) => {
    const name = item.name.toLowerCase()
    return words.every((word) => name.includes(word))
  })
}

/**
 * Create a SortOption that delegates to the shared sortAssets utility
 */
function createSortOption(
  id: AssetSortOption,
  name: string
): SortOption<AssetSortOption> {
  return {
    id,
    name,
    sorter: ({ items }) => sortAssets(items, id)
  }
}

export function getDefaultSortOptions(): SortOption<AssetSortOption>[] {
  return [
    createSortOption('default', t('assetBrowser.sortDefault')),
    createSortOption('name-asc', t('assetBrowser.sortAZ'))
  ]
}
