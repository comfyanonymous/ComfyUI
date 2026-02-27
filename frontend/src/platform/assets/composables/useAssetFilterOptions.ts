import { uniqWith } from 'es-toolkit'
import { computed, toValue } from 'vue'
import type { MaybeRefOrGetter } from 'vue'
import { useI18n } from 'vue-i18n'

import type { SelectOption } from '@/components/input/types'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import type { OwnershipFilterOption } from '@/platform/assets/types/filterTypes'
import { getAssetBaseModels } from '@/platform/assets/utils/assetMetadataUtils'

/**
 * Composable that extracts available filter options from asset data
 * Provides reactive computed properties for file formats, base models, and ownership
 */
export function useAssetFilterOptions(assets: MaybeRefOrGetter<AssetItem[]>) {
  const { t } = useI18n()

  const ownershipOptions = computed<OwnershipFilterOption[]>(() => [
    { name: t('assetBrowser.ownershipAll'), value: 'all' },
    { name: t('assetBrowser.ownershipMyModels'), value: 'my-models' },
    { name: t('assetBrowser.ownershipPublicModels'), value: 'public-models' }
  ])
  /**
   * Extract unique file formats from asset names
   * Returns sorted SelectOption array with extensions
   */
  const availableFileFormats = computed<SelectOption[]>(() => {
    const assetList = toValue(assets)
    const extensions = assetList
      .map((asset) => {
        const extension = asset.name.split('.').pop()
        return extension && extension !== asset.name ? extension : null
      })
      .filter((extension): extension is string => extension !== null)

    const uniqueExtensions = uniqWith(extensions, (a, b) => a === b)

    return uniqueExtensions.sort().map((format) => ({
      name: `.${format}`,
      value: format
    }))
  })

  /**
   * Extract unique base models from asset user metadata
   * Returns sorted SelectOption array with base model names
   */
  const availableBaseModels = computed<SelectOption[]>(() => {
    const assetList = toValue(assets)
    const models = assetList.flatMap((asset) => getAssetBaseModels(asset))

    const uniqueModels = uniqWith(models, (a, b) => a === b)

    return uniqueModels.sort().map((model) => ({
      name: model,
      value: model
    }))
  })

  return {
    availableFileFormats,
    availableBaseModels,
    ownershipOptions
  }
}
