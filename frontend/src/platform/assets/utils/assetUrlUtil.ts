/**
 * Utilities for constructing asset URLs
 */

import { api } from '@/scripts/api'
import type { AssetItem } from '../schemas/assetSchema'
import { getAssetType } from './assetTypeUtil'

/**
 * Get the download/view URL for an asset
 * Constructs the proper URL with filename encoding, type, and subfolder parameters
 *
 * @param asset The asset to get URL for
 * @param defaultType Default type if asset doesn't have tags (default: 'output')
 * @returns Full URL for viewing/downloading the asset
 *
 * @example
 * const url = getAssetUrl(asset)
 * downloadFile(url, asset.name)
 */
export function getAssetUrl(
  asset: AssetItem,
  defaultType: 'input' | 'output' = 'output'
): string {
  const assetType = getAssetType(asset, defaultType)
  const subfolder = asset.user_metadata?.subfolder
  const params = new URLSearchParams()
  params.set('filename', asset.name)
  params.set('type', assetType)
  if (typeof subfolder === 'string' && subfolder) {
    params.set('subfolder', subfolder)
  }
  return api.apiURL(`/view?${params}`)
}
