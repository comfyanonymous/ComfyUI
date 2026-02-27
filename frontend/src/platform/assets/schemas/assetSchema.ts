import { z } from 'zod'

// Zod schemas for asset API validation matching ComfyUI Assets REST API spec
const zAsset = z.object({
  id: z.string(),
  name: z.string(),
  asset_hash: z.string().nullish(),
  size: z.number().optional(), // TBD: Will be provided by history API in the future
  mime_type: z.string().nullish(),
  tags: z.array(z.string()).optional().default([]),
  preview_id: z.string().nullable().optional(),
  preview_url: z.string().optional(),
  created_at: z.string().optional(),
  updated_at: z.string().optional(),
  is_immutable: z.boolean().optional(),
  last_access_time: z.string().optional(),
  metadata: z.record(z.unknown()).optional(), // API allows arbitrary key-value pairs
  user_metadata: z.record(z.unknown()).optional() // API allows arbitrary key-value pairs
})

const zAssetResponse = z.object({
  assets: z.array(zAsset).optional(),
  total: z.number().optional(),
  has_more: z.boolean().optional()
})

const zModelFolder = z.object({
  name: z.string(),
  folders: z.array(z.string())
})

// Zod schema for ModelFile to align with interface
const zModelFile = z.object({
  name: z.string(),
  pathIndex: z.number()
})

const zValidationError = z.object({
  code: z.string(),
  message: z.string(),
  field: z.string()
})

const zValidationResult = z.object({
  is_valid: z.boolean(),
  errors: z.array(zValidationError).optional(),
  warnings: z.array(zValidationError).optional()
})

const zAssetMetadata = z.object({
  content_length: z.number(),
  final_url: z.string(),
  content_type: z.string().optional(),
  filename: z.string().optional(),
  name: z.string().optional(),
  tags: z.array(z.string()).optional(),
  preview_url: z.string().optional(),
  preview_image: z.string().optional(),
  validation: zValidationResult.optional()
})

const zAsyncUploadTask = z.object({
  task_id: z.string(),
  status: z.enum(['created', 'running', 'completed', 'failed']),
  message: z.string().optional()
})

const zAsyncUploadResponse = z.discriminatedUnion('type', [
  z.object({ type: z.literal('sync'), asset: zAsset }),
  z.object({ type: z.literal('async'), task: zAsyncUploadTask })
])

// Filename validation schema
export const assetFilenameSchema = z
  .string()
  .min(1, 'Filename cannot be empty')
  .regex(/^[^\\:*?"<>|]+$/, 'Invalid filename characters') // Allow forward slashes, block backslashes and other unsafe chars
  .regex(/^(?!\/|.*\.\.)/, 'Path must not start with / or contain ..') // Prevent absolute paths and directory traversal
  .trim()

// Export schemas following repository patterns
export const assetItemSchema = zAsset
export const assetResponseSchema = zAssetResponse
export const asyncUploadResponseSchema = zAsyncUploadResponse

// Export types derived from Zod schemas
export type AssetItem = z.infer<typeof zAsset>
export type AssetResponse = z.infer<typeof zAssetResponse>
export type AssetMetadata = z.infer<typeof zAssetMetadata>
export type AsyncUploadResponse = z.infer<typeof zAsyncUploadResponse>
export type ModelFolder = z.infer<typeof zModelFolder>
export type ModelFile = z.infer<typeof zModelFile>

/** Payload for updating an asset via PUT /assets/:id */
export type AssetUpdatePayload = Partial<
  Pick<AssetItem, 'name' | 'tags' | 'user_metadata'>
>

/** User-editable metadata fields for model assets */
const zAssetUserMetadata = z.object({
  name: z.string().optional(),
  base_model: z.array(z.string()).optional(),
  additional_tags: z.array(z.string()).optional(),
  user_description: z.string().optional()
})

export type AssetUserMetadata = z.infer<typeof zAssetUserMetadata>

export const tagsOperationResultSchema = z.object({
  total_tags: z.array(z.string()),
  added: z.array(z.string()).optional(),
  removed: z.array(z.string()).optional(),
  already_present: z.array(z.string()).optional(),
  not_present: z.array(z.string()).optional()
})

export type TagsOperationResult = z.infer<typeof tagsOperationResultSchema>

// Legacy interface for backward compatibility (now aligned with Zod schema)
export interface ModelFolderInfo {
  name: string
  folders: string[]
}
