import type { ComputedRef, InjectionKey } from 'vue'

import type { AssetKind } from '@/types/widgetTypes'

/**
 * Minimal interface for items in FormDropdown.
 * Both AssetItem (from cloud API) and local file items satisfy this contract.
 */
export interface FormDropdownItem {
  id: string
  /** Display name shown in the dropdown */
  name: string
  /** Original/alternate label (e.g., original filename) */
  label?: string
  /** Preview image/video URL */
  preview_url?: string
  /** Whether the item is immutable (public model) - used for ownership filtering */
  is_immutable?: boolean
  /** Base models this item is compatible with - used for base model filtering */
  base_models?: string[]
}

export interface SortOption<TId extends string = string> {
  id: TId
  name: string
  sorter: (ctx: { items: readonly FormDropdownItem[] }) => FormDropdownItem[]
}

export type LayoutMode = 'list' | 'grid' | 'list-small'

export const AssetKindKey: InjectionKey<ComputedRef<AssetKind | undefined>> =
  Symbol('assetKind')
