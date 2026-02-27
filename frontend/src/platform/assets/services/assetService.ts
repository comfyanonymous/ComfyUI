import { fromZodError } from 'zod-validation-error'

import { st } from '@/i18n'

import {
  assetItemSchema,
  assetResponseSchema,
  asyncUploadResponseSchema,
  tagsOperationResultSchema
} from '@/platform/assets/schemas/assetSchema'
import type {
  AssetItem,
  AssetMetadata,
  AssetResponse,
  AssetUpdatePayload,
  AsyncUploadResponse,
  ModelFile,
  ModelFolder,
  TagsOperationResult
} from '@/platform/assets/schemas/assetSchema'
import { isCloud } from '@/platform/distribution/types'
import { useSettingStore } from '@/platform/settings/settingStore'
import { api } from '@/scripts/api'
import { useModelToNodeStore } from '@/stores/modelToNodeStore'

export interface PaginationOptions {
  limit?: number
  offset?: number
}

interface AssetRequestOptions extends PaginationOptions {
  includeTags: string[]
  includePublic?: boolean
}

interface AssetExportOptions {
  job_ids?: string[]
  asset_ids?: string[]
  naming_strategy?:
    | 'group_by_job_id'
    | 'prepend_job_id'
    | 'preserve'
    | 'asset_id'
  job_asset_name_filters?: Record<string, string[]>
}

/**
 * Maps CivitAI validation error codes to localized error messages
 */
function getLocalizedErrorMessage(errorCode: string): string {
  const errorMessages: Record<string, string> = {
    // Validation errors
    FILE_TOO_LARGE: st('assetBrowser.errorFileTooLarge', 'File too large'),
    FORMAT_NOT_ALLOWED: st(
      'assetBrowser.errorFormatNotAllowed',
      'Format not allowed'
    ),
    UNSAFE_PICKLE_SCAN: st(
      'assetBrowser.errorUnsafePickleScan',
      'Unsafe pickle scan'
    ),
    UNSAFE_VIRUS_SCAN: st(
      'assetBrowser.errorUnsafeVirusScan',
      'Unsafe virus scan'
    ),
    MODEL_TYPE_NOT_SUPPORTED: st(
      'assetBrowser.errorModelTypeNotSupported',
      'Model type not supported'
    ),

    // HTTP 400 - Bad Request
    INVALID_URL: st('assetBrowser.errorInvalidUrl', 'Please provide a URL.'),
    INVALID_URL_FORMAT: st(
      'assetBrowser.errorInvalidUrlFormat',
      'The URL format is invalid. Please check and try again.'
    ),
    UNSUPPORTED_SOURCE: st(
      'assetBrowser.errorUnsupportedSource',
      'This URL is not supported. Only Hugging Face and Civitai URLs are allowed.'
    ),

    // HTTP 401 - Unauthorized
    UNAUTHORIZED: st(
      'assetBrowser.errorUnauthorized',
      'Please sign in to continue.'
    ),

    // HTTP 422 - External Source Errors
    USER_TOKEN_INVALID: st(
      'assetBrowser.errorUserTokenInvalid',
      'Your stored API token is invalid or expired. Please update your token in settings.'
    ),
    USER_TOKEN_ACCESS_DENIED: st(
      'assetBrowser.errorUserTokenAccessDenied',
      'Your API token does not have access to this resource. Please check your token permissions.'
    ),
    UNAUTHORIZED_SOURCE: st(
      'assetBrowser.errorUnauthorizedSource',
      'This resource requires authentication. Please add your API token in settings.'
    ),
    ACCESS_FORBIDDEN: st(
      'assetBrowser.errorAccessForbidden',
      'Access to this resource is forbidden.'
    ),
    RESOURCE_NOT_FOUND: st(
      'assetBrowser.errorResourceNotFound',
      'The file was not found. Please check the URL and try again.'
    ),
    RATE_LIMITED: st(
      'assetBrowser.errorRateLimited',
      'Too many requests. Please try again in a few minutes.'
    ),
    SOURCE_SERVER_ERROR: st(
      'assetBrowser.errorSourceServerError',
      'The source server is experiencing issues. Please try again later.'
    ),
    NETWORK_TIMEOUT: st(
      'assetBrowser.errorNetworkTimeout',
      'Request timed out. Please try again.'
    ),
    CONNECTION_REFUSED: st(
      'assetBrowser.errorConnectionRefused',
      'Unable to connect to the source. Please try again later.'
    ),
    INVALID_HOST: st(
      'assetBrowser.errorInvalidHost',
      'The source URL hostname could not be resolved.'
    ),
    NETWORK_ERROR: st(
      'assetBrowser.errorNetworkError',
      'A network error occurred. Please check your connection and try again.'
    ),
    REQUEST_CANCELLED: st(
      'assetBrowser.errorRequestCancelled',
      'Request was cancelled.'
    ),
    DOWNLOAD_CANCELLED: st(
      'assetBrowser.errorDownloadCancelled',
      'Download was cancelled.'
    ),
    METADATA_FETCH_FAILED: st(
      'assetBrowser.errorMetadataFetchFailed',
      'Failed to fetch file information from the source.'
    ),
    HTTP_ERROR: st(
      'assetBrowser.errorHttpError',
      'An error occurred while fetching metadata.'
    ),

    // HTTP 500 - Internal Server Errors
    SERVICE_UNAVAILABLE: st(
      'assetBrowser.errorServiceUnavailable',
      'Service temporarily unavailable. Please try again later.'
    ),
    INTERNAL_ERROR: st(
      'assetBrowser.errorInternalError',
      'An unexpected error occurred. Please try again.'
    )
  }
  return (
    errorMessages[errorCode] ||
    st('assetBrowser.errorUnknown', 'Unknown error') ||
    'Unknown error'
  )
}

const ASSETS_ENDPOINT = '/assets'
const ASSETS_DOWNLOAD_ENDPOINT = '/assets/download'
const ASSETS_EXPORT_ENDPOINT = '/assets/export'
const EXPERIMENTAL_WARNING = `EXPERIMENTAL: If you are seeing this please make sure "Comfy.Assets.UseAssetAPI" is set to "false" in your ComfyUI Settings.\n`
const DEFAULT_LIMIT = 500

export const MODELS_TAG = 'models'
export const MISSING_TAG = 'missing'

/**
 * Validates asset response data using Zod schema
 */
function validateAssetResponse(data: unknown): AssetResponse {
  const result = assetResponseSchema.safeParse(data)
  if (result.success) return result.data

  const error = fromZodError(result.error)
  throw new Error(
    `${EXPERIMENTAL_WARNING}Invalid asset response against zod schema:\n${error}`
  )
}

/**
 * Private service for asset-related network requests
 * Not exposed globally - used internally by ComfyApi
 */
function createAssetService() {
  /**
   * Handles API response with consistent error handling and Zod validation
   */
  async function handleAssetRequest(
    options: AssetRequestOptions,
    context: string
  ): Promise<AssetResponse> {
    const {
      includeTags,
      limit = DEFAULT_LIMIT,
      offset,
      includePublic
    } = options
    const queryParams = new URLSearchParams({
      include_tags: includeTags.join(','),
      limit: limit.toString()
    })
    if (offset !== undefined && offset > 0) {
      queryParams.set('offset', offset.toString())
    }
    if (includePublic !== undefined) {
      queryParams.set('include_public', includePublic ? 'true' : 'false')
    }

    const url = `${ASSETS_ENDPOINT}?${queryParams.toString()}`
    const res = await api.fetchApi(url)
    if (!res.ok) {
      throw new Error(
        `${EXPERIMENTAL_WARNING}Unable to load ${context}: Server returned ${res.status}. Please try again.`
      )
    }
    const data = await res.json()
    return validateAssetResponse(data)
  }
  /**
   * Gets a list of model folder keys from the asset API
   *
   * Logic:
   * 1. Extract directory names directly from asset tags
   * 2. Filter out blacklisted directories
   * 3. Return alphabetically sorted directories with assets
   *
   * @returns The list of model folder keys
   */
  async function getAssetModelFolders(): Promise<ModelFolder[]> {
    const data = await handleAssetRequest(
      { includeTags: [MODELS_TAG] },
      'model folders'
    )

    // Blacklist directories we don't want to show
    const blacklistedDirectories = new Set(['configs'])

    // Extract directory names from assets that actually exist, exclude missing assets
    const discoveredFolders = new Set<string>(
      data?.assets
        ?.filter((asset) => !asset.tags.includes(MISSING_TAG))
        ?.flatMap((asset) => asset.tags)
        ?.filter(
          (tag) => tag !== MODELS_TAG && !blacklistedDirectories.has(tag)
        ) ?? []
    )

    // Return only discovered folders in alphabetical order
    const sortedFolders = Array.from(discoveredFolders).toSorted()
    return sortedFolders.map((name) => ({ name, folders: [] }))
  }

  /**
   * Gets a list of models in the specified folder from the asset API
   * @param folder The folder to list models from, such as 'checkpoints'
   * @returns The list of model filenames within the specified folder
   */
  async function getAssetModels(folder: string): Promise<ModelFile[]> {
    const data = await handleAssetRequest(
      { includeTags: [MODELS_TAG, folder] },
      `models for ${folder}`
    )

    return (
      data?.assets
        ?.filter(
          (asset) =>
            !asset.tags.includes(MISSING_TAG) && asset.tags.includes(folder)
        )
        ?.map((asset) => ({
          name: asset.name,
          pathIndex: 0
        })) ?? []
    )
  }

  /**
   * Checks if a widget input should use the asset browser based on both input name and node comfyClass
   *
   * @param nodeType - The ComfyUI node comfyClass (e.g., 'CheckpointLoaderSimple', 'LoraLoader')
   * @param widgetName - The name of the widget to check (e.g., 'ckpt_name')
   * @returns true if this input should use asset browser
   */
  function isAssetBrowserEligible(
    nodeType: string | undefined,
    widgetName: string
  ): boolean {
    if (!nodeType || !widgetName) return false
    return (
      useModelToNodeStore().getRegisteredNodeTypes()[nodeType] === widgetName
    )
  }

  /**
   * Checks if the asset API is enabled (cloud environment + user setting).
   */
  function isAssetAPIEnabled(): boolean {
    if (!isCloud) return false
    return !!useSettingStore().get('Comfy.Assets.UseAssetAPI')
  }

  /**
   * Checks if the asset browser should be used for a given node input.
   * Combines the cloud environment check, user setting, and eligibility check.
   *
   * @param nodeType - The ComfyUI node comfyClass
   * @param widgetName - The name of the widget to check
   * @returns true if this input should use the asset browser
   */
  function shouldUseAssetBrowser(
    nodeType: string | undefined,
    widgetName: string
  ): boolean {
    return isAssetAPIEnabled() && isAssetBrowserEligible(nodeType, widgetName)
  }

  /**
   * Gets assets for a specific node type by finding the matching category
   * and fetching all assets with that category tag
   *
   * @param nodeType - The ComfyUI node type (e.g., 'CheckpointLoaderSimple')
   * @param options - Pagination options
   * @param options.limit - Maximum number of assets to return (default: 500)
   * @param options.offset - Number of assets to skip (default: 0)
   * @returns Promise<AssetItem[]> - Full asset objects with preserved metadata
   */
  async function getAssetsForNodeType(
    nodeType: string,
    { limit = DEFAULT_LIMIT, offset = 0 }: PaginationOptions = {}
  ): Promise<AssetItem[]> {
    if (!nodeType || typeof nodeType !== 'string') {
      return []
    }

    // Find the category for this node type using efficient O(1) lookup
    const modelToNodeStore = useModelToNodeStore()
    const category = modelToNodeStore.getCategoryForNodeType(nodeType)

    if (!category) {
      return []
    }

    // Fetch assets for this category using same API pattern as getAssetModels
    const data = await handleAssetRequest(
      { includeTags: [MODELS_TAG, category], limit, offset },
      `assets for ${nodeType}`
    )

    // Return full AssetItem[] objects (don't strip like getAssetModels does)
    return (
      data?.assets?.filter(
        (asset) =>
          !asset.tags.includes(MISSING_TAG) && asset.tags.includes(category)
      ) ?? []
    )
  }

  /**
   * Gets complete details for a specific asset by ID
   * Calls the detail endpoint which includes user_metadata and all fields
   *
   * @param id - The asset ID
   * @returns Promise<AssetItem> - Complete asset object with user_metadata
   */
  async function getAssetDetails(id: string): Promise<AssetItem> {
    const res = await api.fetchApi(`${ASSETS_ENDPOINT}/${id}`)
    if (!res.ok) {
      throw new Error(
        `${EXPERIMENTAL_WARNING}Unable to load asset details for ${id}: Server returned ${res.status}. Please try again.`
      )
    }
    const data = await res.json()

    // Validate the single asset response against our schema
    const result = assetResponseSchema.safeParse({ assets: [data] })
    if (result.success && result.data.assets?.[0]) {
      return result.data.assets[0]
    }

    const error = result.error
      ? fromZodError(result.error)
      : 'Unknown validation error'
    throw new Error(
      `${EXPERIMENTAL_WARNING}Invalid asset response against zod schema:\n${error}`
    )
  }

  /**
   * Gets assets filtered by a specific tag
   *
   * @param tag - The tag to filter by (e.g., 'models', 'input')
   * @param includePublic - Whether to include public assets (default: true)
   * @param options - Pagination options
   * @param options.limit - Maximum number of assets to return (default: 500)
   * @param options.offset - Number of assets to skip (default: 0)
   * @returns Promise<AssetItem[]> - Full asset objects filtered by tag, excluding missing assets
   */
  async function getAssetsByTag(
    tag: string,
    includePublic: boolean = true,
    { limit = DEFAULT_LIMIT, offset = 0 }: PaginationOptions = {}
  ): Promise<AssetItem[]> {
    const data = await handleAssetRequest(
      { includeTags: [tag], limit, offset, includePublic },
      `assets for tag ${tag}`
    )

    return (
      data?.assets?.filter((asset) => !asset.tags.includes(MISSING_TAG)) ?? []
    )
  }

  /**
   * Deletes an asset by ID
   * Only available in cloud environment
   *
   * @param id - The asset ID (UUID)
   * @returns Promise<void>
   * @throws Error if deletion fails
   */
  async function deleteAsset(id: string): Promise<void> {
    const res = await api.fetchApi(`${ASSETS_ENDPOINT}/${id}`, {
      method: 'DELETE'
    })

    if (!res.ok) {
      throw new Error(
        `Unable to delete asset ${id}: Server returned ${res.status}`
      )
    }
  }

  /**
   * Update metadata of an asset by ID
   * Only available in cloud environment
   *
   * @param id - The asset ID (UUID)
   * @param newData - The data to update
   * @returns Promise<AssetItem>
   * @throws Error if update fails
   */
  async function updateAsset(
    id: string,
    newData: AssetUpdatePayload
  ): Promise<AssetItem> {
    const res = await api.fetchApi(`${ASSETS_ENDPOINT}/${id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(newData)
    })

    if (!res.ok) {
      throw new Error(
        `Unable to update asset ${id}: Server returned ${res.status}`
      )
    }

    const newAsset = assetItemSchema.safeParse(await res.json())
    if (newAsset.success) {
      return newAsset.data
    }

    throw new Error(
      `Unable to update asset ${id}: Invalid response - ${newAsset.error}`
    )
  }

  /**
   * Retrieves metadata from a download URL without downloading the file
   *
   * @param url - Download URL to retrieve metadata from (will be URL-encoded)
   * @returns Promise with metadata including content_length, final_url, filename, etc.
   * @throws Error if metadata retrieval fails
   */
  async function getAssetMetadata(url: string): Promise<AssetMetadata> {
    const encodedUrl = encodeURIComponent(url)
    const res = await api.fetchApi(
      `${ASSETS_ENDPOINT}/remote-metadata?url=${encodedUrl}`
    )

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}))
      throw new Error(
        getLocalizedErrorMessage(errorData.code || 'UNKNOWN_ERROR')
      )
    }

    const data: AssetMetadata = await res.json()
    if (data.validation?.is_valid === false) {
      throw new Error(
        getLocalizedErrorMessage(
          data.validation?.errors?.[0]?.code || 'UNKNOWN_ERROR'
        )
      )
    }

    return data
  }

  /**
   * Uploads an asset by providing a URL to download from
   *
   * @param params - Upload parameters
   * @param params.url - HTTP/HTTPS URL to download from
   * @param params.name - Display name (determines extension)
   * @param params.tags - Optional freeform tags
   * @param params.user_metadata - Optional custom metadata object
   * @param params.preview_id - Optional UUID for preview asset
   * @returns Promise<AssetItem & { created_new: boolean }> - Asset object with created_new flag
   * @throws Error if upload fails
   */
  async function uploadAssetFromUrl(params: {
    url: string
    name: string
    tags?: string[]
    user_metadata?: Record<string, unknown>
    preview_id?: string
  }): Promise<AssetItem & { created_new: boolean }> {
    const res = await api.fetchApi(ASSETS_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(params)
    })

    if (!res.ok) {
      throw new Error(
        st(
          'assetBrowser.errorUploadFailed',
          'Failed to upload asset. Please try again.'
        )
      )
    }

    return await res.json()
  }

  /**
   * Uploads an asset from base64 data
   *
   * @param params - Upload parameters
   * @param params.data - Base64 data URL (e.g., "data:image/png;base64,...")
   * @param params.name - Display name (determines extension)
   * @param params.tags - Optional freeform tags
   * @param params.user_metadata - Optional custom metadata object
   * @returns Promise<AssetItem & { created_new: boolean }> - Asset object with created_new flag
   * @throws Error if upload fails
   */
  async function uploadAssetFromBase64(params: {
    data: string
    name: string
    tags?: string[]
    user_metadata?: Record<string, unknown>
  }): Promise<AssetItem & { created_new: boolean }> {
    // Validate that data is a data URL
    if (!params.data || !params.data.startsWith('data:')) {
      throw new Error(
        'Invalid data URL: expected a string starting with "data:"'
      )
    }

    // Convert base64 data URL to Blob
    const blob = await fetch(params.data).then((r) => r.blob())

    // Create FormData and append the blob
    const formData = new FormData()
    formData.append('file', blob, params.name)

    if (params.tags) {
      formData.append('tags', JSON.stringify(params.tags))
    }

    if (params.user_metadata) {
      formData.append('user_metadata', JSON.stringify(params.user_metadata))
    }

    const res = await api.fetchApi(ASSETS_ENDPOINT, {
      method: 'POST',
      body: formData
    })

    if (!res.ok) {
      throw new Error(
        `Failed to upload asset from base64: ${res.status} ${res.statusText}`
      )
    }

    return await res.json()
  }

  /**
   * Add tags to an asset
   * @param id - The asset ID (UUID)
   * @param tags - Tags to add
   * @returns Promise<TagsOperationResult>
   */
  async function addAssetTags(
    id: string,
    tags: string[]
  ): Promise<TagsOperationResult> {
    const res = await api.fetchApi(`${ASSETS_ENDPOINT}/${id}/tags`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tags })
    })

    if (!res.ok) {
      throw new Error(
        `Unable to add tags to asset ${id}: Server returned ${res.status}`
      )
    }

    const result = await res.json()
    const parseResult = tagsOperationResultSchema.safeParse(result)
    if (!parseResult.success) {
      throw fromZodError(parseResult.error)
    }
    return parseResult.data
  }

  /**
   * Remove tags from an asset
   * @param id - The asset ID (UUID)
   * @param tags - Tags to remove
   * @returns Promise<TagsOperationResult>
   */
  async function removeAssetTags(
    id: string,
    tags: string[]
  ): Promise<TagsOperationResult> {
    const res = await api.fetchApi(`${ASSETS_ENDPOINT}/${id}/tags`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tags })
    })

    if (!res.ok) {
      throw new Error(
        `Unable to remove tags from asset ${id}: Server returned ${res.status}`
      )
    }

    const result = await res.json()
    const parseResult = tagsOperationResultSchema.safeParse(result)
    if (!parseResult.success) {
      throw fromZodError(parseResult.error)
    }
    return parseResult.data
  }

  /**
   * Uploads an asset asynchronously using the /api/assets/download endpoint
   * Returns immediately with either the asset (if already exists) or a task to track
   *
   * @param params - Upload parameters
   * @param params.source_url - HTTP/HTTPS URL to download from
   * @param params.tags - Optional freeform tags
   * @param params.user_metadata - Optional custom metadata object
   * @param params.preview_id - Optional UUID for preview asset
   * @returns Promise<AsyncUploadResponse> - Either sync asset or async task info
   * @throws Error if upload fails
   */
  async function uploadAssetAsync(params: {
    source_url: string
    tags?: string[]
    user_metadata?: Record<string, unknown>
    preview_id?: string
  }): Promise<AsyncUploadResponse> {
    const res = await api.fetchApi(ASSETS_DOWNLOAD_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })

    if (!res.ok) {
      throw new Error(
        st(
          'assetBrowser.errorUploadFailed',
          'Failed to upload asset. Please try again.'
        )
      )
    }

    const data = await res.json()

    if (res.status === 202) {
      const result = asyncUploadResponseSchema.safeParse({
        type: 'async',
        task: data
      })
      if (!result.success) {
        throw new Error(
          st(
            'assetBrowser.errorUploadFailed',
            'Failed to parse async upload response. Please try again.'
          )
        )
      }
      return result.data
    }

    const result = asyncUploadResponseSchema.safeParse({
      type: 'sync',
      asset: data
    })
    if (!result.success) {
      throw new Error(
        st(
          'assetBrowser.errorUploadFailed',
          'Failed to parse sync upload response. Please try again.'
        )
      )
    }
    return result.data
  }

  async function createAssetExport(
    params: AssetExportOptions
  ): Promise<{ task_id: string; status: string; message?: string }> {
    const res = await api.fetchApi(ASSETS_EXPORT_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })

    if (!res.ok) {
      throw new Error(`Failed to create asset export: ${res.status}`)
    }

    return await res.json()
  }

  async function getExportDownloadUrl(
    exportName: string
  ): Promise<{ url: string; expires_at?: string }> {
    const res = await api.fetchApi(`/assets/exports/${exportName}`)

    if (!res.ok) {
      throw new Error(`Failed to get export download URL: ${res.status}`)
    }

    return await res.json()
  }

  return {
    getAssetModelFolders,
    getAssetModels,
    isAssetAPIEnabled,
    isAssetBrowserEligible,
    shouldUseAssetBrowser,
    getAssetsForNodeType,
    getAssetDetails,
    getAssetsByTag,
    deleteAsset,
    updateAsset,
    addAssetTags,
    removeAssetTags,
    getAssetMetadata,
    uploadAssetFromUrl,
    uploadAssetFromBase64,
    uploadAssetAsync,
    createAssetExport,
    getExportDownloadUrl
  }
}

export const assetService = createAssetService()
