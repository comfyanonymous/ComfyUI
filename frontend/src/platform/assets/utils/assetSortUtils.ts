/**
 * Shared asset sorting utilities
 * Used by both AssetBrowser and FormDropdown
 */

import type { AssetSortOption } from '../types/filterTypes'

/**
 * Minimal interface for sortable items
 * Works with both AssetItem and FormDropdownItem
 */
export interface SortableItem {
  name: string
  label?: string
  created_at?: string | null
}

function getDisplayName(item: SortableItem): string {
  return item.label ?? item.name
}

/**
 * Sort items by the specified sort option
 * @param items - Array of sortable items
 * @param sortBy - Sort option from AssetSortOption
 * @returns New sorted array (does not mutate input)
 */
export function sortAssets<T extends SortableItem>(
  items: readonly T[],
  sortBy: AssetSortOption
): T[] {
  if (sortBy === 'default') {
    return items.slice()
  }

  const sorted = items.slice()

  switch (sortBy) {
    case 'name-desc':
      return sorted.sort((a, b) =>
        getDisplayName(b).localeCompare(getDisplayName(a), undefined, {
          numeric: true,
          sensitivity: 'base'
        })
      )
    case 'recent':
      return sorted.sort(
        (a, b) =>
          new Date(b.created_at ?? 0).getTime() -
          new Date(a.created_at ?? 0).getTime()
      )
    case 'name-asc':
    default:
      return sorted.sort((a, b) =>
        getDisplayName(a).localeCompare(getDisplayName(b), undefined, {
          numeric: true,
          sensitivity: 'base'
        })
      )
  }
}
