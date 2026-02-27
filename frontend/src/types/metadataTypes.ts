import type {
  ComfyApiWorkflow,
  ComfyWorkflowJSON
} from '@/platform/workflow/validation/schemas/workflowSchema'

/**
 * Tag names used in ComfyUI metadata
 */
export enum ComfyMetadataTags {
  PROMPT = 'PROMPT',
  WORKFLOW = 'WORKFLOW'
}

/**
 * Metadata extracted from ComfyUI output files
 */
export interface ComfyMetadata {
  workflow?: ComfyWorkflowJSON
  prompt?: ComfyApiWorkflow
  [key: string]: string | ComfyWorkflowJSON | ComfyApiWorkflow | undefined
}

export type EbmlElementRange = {
  /** Position in the buffer */
  pos: number
  /** Length of the element in bytes */
  len: number
}

export type EbmlTagPosition = {
  name: EbmlElementRange
  value: EbmlElementRange
}

export type VInt = {
  /** The value of the variable-length integer */
  value: number
  /** The length of the variable-length integer in bytes */
  length: number
}

export type TextRange = {
  start: number
  end: number
}

export enum ASCII {
  GLTF = 0x46546c67,
  JSON = 0x4e4f534a,
  OPEN_BRACE = 0x7b
}

export enum GltfSizeBytes {
  HEADER = 12,
  CHUNK_HEADER = 8
}

export type GltfHeader = {
  magicNumber: number
  gltfFormatVersion: number
  totalLengthBytes: number
}

export type GltfChunkHeader = {
  chunkLengthBytes: number
  chunkTypeIdentifier: number
}

type GltfExtras = {
  workflow?: string | object
  prompt?: string | object
  [key: string]: unknown
}

export type GltfJsonData = {
  asset?: {
    extras?: GltfExtras
    [key: string]: unknown
  }
  [key: string]: unknown
}

/**
 * Represents the content range [start, end) of an ISOBMFF box, excluding its header.
 * Null if the box was not found.
 */
export type IsobmffBoxContentRange = { start: number; end: number } | null

export type AvifInfeBox = {
  box_header: {
    size: number
    type: 'infe'
  }
  version: number
  flags: number
  item_ID: number
  item_protection_index: number
  item_type: string
  item_name: string
  content_type?: string
  content_encoding?: string
}

export type AvifIinfBox = {
  box_header: {
    size: number
    type: 'iinf'
  }
  version: number
  flags: number
  entry_count: number
  entries: AvifInfeBox[]
}

type AvifIlocItemExtent = {
  extent_offset: number
  extent_length: number
}

type AvifIlocItem = {
  item_ID: number
  data_reference_index: number
  base_offset: number
  extent_count: number
  extents: AvifIlocItemExtent[]
}

export type AvifIlocBox = {
  box_header: {
    size: number
    type: 'iloc'
  }
  version: number
  flags: number
  offset_size: number
  length_size: number
  base_offset_size: number
  index_size: number
  item_count: number
  items: AvifIlocItem[]
}
