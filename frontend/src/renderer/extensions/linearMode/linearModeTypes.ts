import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import type { ResultItemImpl } from '@/stores/queueStore'

export interface InProgressItem {
  id: string
  jobId: string
  state: 'skeleton' | 'latent' | 'image'
  latentPreviewUrl?: string
  output?: ResultItemImpl
}

export interface OutputSelection {
  asset?: AssetItem
  output?: ResultItemImpl
  canShowPreview: boolean
  latentPreviewUrl?: string
}

export type SelectionValue =
  | { id: string; kind: 'inProgress'; itemId: string }
  | { id: string; kind: 'history'; assetId: string; key: number }
