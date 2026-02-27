/**
 * Utilities for working with asset types
 */

import type { AssetItem } from '../schemas/assetSchema'

/**
 * Extract asset type from an asset's tags array
 * Falls back to a default type if tags are not present
 *
 * @param asset The asset to extract type from
 * @param defaultType Default type to use if tags are empty (default: 'output')
 * @returns The asset type ('input', 'output', 'temp', etc.)
 *
 * @example
 * getAssetType(asset) // Returns 'output' or first tag
 * getAssetType(asset, 'input') // Returns 'input' if no tags
 */
export function getAssetType(
  asset: AssetItem,
  defaultType: 'input' | 'output' = 'output'
): string {
  return asset.tags?.[0] || defaultType
}
