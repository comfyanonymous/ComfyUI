import type { InjectionKey, Ref } from 'vue'
import { z } from 'zod'

import { assetItemSchema } from './assetSchema'

const zMediaKindSchema = z.enum([
  'video',
  'audio',
  'image',
  '3D',
  'text',
  'other'
])
export type MediaKind = z.infer<typeof zMediaKindSchema>

const zDimensionsSchema = z.object({
  width: z.number().positive(),
  height: z.number().positive()
})

// Extend the base asset schema with media-specific fields
const zMediaAssetDisplayItemSchema = assetItemSchema.extend({
  // New required fields
  kind: zMediaKindSchema,
  src: z.string().url(),

  // New optional fields
  duration: z.number().nonnegative().optional(),
  dimensions: zDimensionsSchema.optional()
})

// Asset context schema
const zAssetContextSchema = z.object({
  type: z.enum(['input', 'output']),
  outputCount: z.number().positive().optional() // Only for output context
})

// Export the inferred types
export type AssetMeta = z.infer<typeof zMediaAssetDisplayItemSchema>
export type AssetContext = z.infer<typeof zAssetContextSchema>

// Injection key for MediaAsset provide/inject pattern
interface MediaAssetProviderValue {
  asset: Ref<AssetMeta | undefined>
  context: Ref<AssetContext>
  isVideoPlaying: Ref<boolean>
  showVideoControls: Ref<boolean>
}

export const MediaAssetKey: InjectionKey<MediaAssetProviderValue> =
  Symbol('mediaAsset')
