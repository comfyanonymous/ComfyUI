import {
  type ComfyApiWorkflow,
  type ComfyWorkflowJSON
} from '@/platform/workflow/validation/schemas/workflowSchema'
import {
  ASCII,
  type ComfyMetadata,
  ComfyMetadataTags,
  type GltfChunkHeader,
  type GltfHeader,
  type GltfJsonData,
  GltfSizeBytes
} from '@/types/metadataTypes'

const MAX_READ_BYTES = 1 << 20

const isJsonChunk = (chunk: GltfChunkHeader | null): boolean =>
  !!chunk && chunk.chunkTypeIdentifier === ASCII.JSON

const isValidChunkRange = (
  start: number,
  length: number,
  bufferSize: number
): boolean => start + length <= bufferSize

const byteArrayToString = (bytes: Uint8Array): string =>
  new TextDecoder().decode(bytes)

const parseGltfBinaryHeader = (dataView: DataView): GltfHeader | null => {
  if (dataView.byteLength < GltfSizeBytes.HEADER) return null

  const magicNumber = dataView.getUint32(0, true)
  if (magicNumber !== ASCII.GLTF) return null

  return {
    magicNumber,
    gltfFormatVersion: dataView.getUint32(4, true),
    totalLengthBytes: dataView.getUint32(8, true)
  }
}

const parseChunkHeaderAtOffset = (
  dataView: DataView,
  offset: number
): GltfChunkHeader | null => {
  if (offset + GltfSizeBytes.CHUNK_HEADER > dataView.byteLength) return null

  return {
    chunkLengthBytes: dataView.getUint32(offset, true),
    chunkTypeIdentifier: dataView.getUint32(offset + 4, true)
  }
}

const extractJsonChunk = (
  buffer: ArrayBuffer
): { start: number; length: number } | null => {
  const dataView = new DataView(buffer)

  const header = parseGltfBinaryHeader(dataView)
  if (!header) return null

  const chunkOffset = GltfSizeBytes.HEADER
  const firstChunk = parseChunkHeaderAtOffset(dataView, chunkOffset)
  if (!firstChunk || !isJsonChunk(firstChunk)) return null

  const jsonStart = chunkOffset + GltfSizeBytes.CHUNK_HEADER
  const isValid = isValidChunkRange(
    jsonStart,
    firstChunk.chunkLengthBytes,
    dataView.byteLength
  )
  if (!isValid) return null

  return { start: jsonStart, length: firstChunk.chunkLengthBytes }
}

const extractJsonChunkData = (buffer: ArrayBuffer): Uint8Array | null => {
  const chunkLocation = extractJsonChunk(buffer)
  if (!chunkLocation) return null

  return new Uint8Array(buffer, chunkLocation.start, chunkLocation.length)
}

const parseJson = (text: string): ReturnType<typeof JSON.parse> | null => {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

const parseJsonBytes = (
  bytes: Uint8Array
): ReturnType<typeof JSON.parse> | null => {
  const jsonString = byteArrayToString(bytes)
  return parseJson(jsonString)
}

const parseMetadataValue = (
  value: string | object
): ComfyWorkflowJSON | ComfyApiWorkflow | undefined => {
  if (typeof value !== 'string')
    return value as ComfyWorkflowJSON | ComfyApiWorkflow

  const parsed = parseJson(value)
  if (!parsed) return undefined

  return parsed as ComfyWorkflowJSON | ComfyApiWorkflow
}

const extractComfyMetadata = (jsonData: GltfJsonData): ComfyMetadata => {
  const metadata: ComfyMetadata = {}

  if (!jsonData?.asset?.extras) return metadata

  const { extras } = jsonData.asset

  if (extras.workflow) {
    const parsedValue = parseMetadataValue(extras.workflow)
    if (parsedValue) {
      metadata[ComfyMetadataTags.WORKFLOW.toLowerCase()] = parsedValue
    }
  }

  if (extras.prompt) {
    const parsedValue = parseMetadataValue(extras.prompt)
    if (parsedValue) {
      metadata[ComfyMetadataTags.PROMPT.toLowerCase()] = parsedValue
    }
  }

  return metadata
}

const processGltfFileBuffer = (buffer: ArrayBuffer): ComfyMetadata => {
  const jsonChunk = extractJsonChunkData(buffer)
  if (!jsonChunk) return {}

  const parsedJson = parseJsonBytes(jsonChunk)
  if (!parsedJson) return {}

  return extractComfyMetadata(parsedJson)
}

/**
 * Extract ComfyUI metadata from a GLTF binary file (GLB)
 */
export function getGltfBinaryMetadata(file: File): Promise<ComfyMetadata> {
  return new Promise<ComfyMetadata>((resolve) => {
    if (!file) return Promise.resolve({})

    const bytesToRead = Math.min(file.size, MAX_READ_BYTES)

    const reader = new FileReader()
    reader.onload = (event) => {
      try {
        if (!event.target?.result) {
          resolve({})
          return
        }

        resolve(processGltfFileBuffer(event.target.result as ArrayBuffer))
      } catch {
        resolve({})
      }
    }
    reader.onerror = () => resolve({})
    reader.readAsArrayBuffer(file.slice(0, bytesToRead))
  })
}
