import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import type { OwnershipOption } from '@/platform/assets/types/filterTypes'
import { getAssetBaseModels } from '@/platform/assets/utils/assetMetadataUtils'

export function filterByCategory(category: string) {
  return (asset: AssetItem) => {
    if (category === 'all') return true

    // Check if any tag matches the category (for exact matches)
    if (asset.tags.includes(category)) return true

    // Check if any tag's top-level folder matches the category
    return asset.tags.some((tag) => {
      if (typeof tag === 'string' && tag.includes('/')) {
        return tag.split('/')[0] === category
      }
      return false
    })
  }
}

export function filterByFileFormats(formats: string[]) {
  return (asset: AssetItem) => {
    if (formats.length === 0) return true
    const formatSet = new Set(formats)
    const extension = asset.name.split('.').pop()?.toLowerCase()
    return extension ? formatSet.has(extension) : false
  }
}

export function filterByBaseModels(models: string[] | Set<string>) {
  const modelSet = models instanceof Set ? models : new Set(models)
  return (asset: AssetItem) => {
    if (modelSet.size === 0) return true
    const assetBaseModels = getAssetBaseModels(asset)
    return assetBaseModels.some((model) => modelSet.has(model))
  }
}

export function filterByOwnership(ownership: OwnershipOption) {
  return (asset: AssetItem) => {
    if (ownership === 'all') return true
    if (ownership === 'my-models') return asset.is_immutable === false
    if (ownership === 'public-models') return asset.is_immutable === true
    return true
  }
}

export function filterItemByOwnership<T extends { is_immutable?: boolean }>(
  items: T[],
  ownership: OwnershipOption
): T[] {
  if (ownership === 'all') return items
  const isPublic = ownership === 'public-models'
  return items.filter((item) => item.is_immutable === isPublic)
}

export function filterItemByBaseModels<T extends { base_models?: string[] }>(
  items: T[],
  selectedModels: Set<string>
): T[] {
  if (selectedModels.size === 0) return items
  return items.filter((item) =>
    item.base_models?.some((model) => selectedModels.has(model))
  )
}
