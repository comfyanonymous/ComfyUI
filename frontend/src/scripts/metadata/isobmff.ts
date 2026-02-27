import {
  type ComfyApiWorkflow,
  type ComfyWorkflowJSON
} from '@/platform/workflow/validation/schemas/workflowSchema'
import {
  ASCII,
  type ComfyMetadata,
  ComfyMetadataTags,
  type IsobmffBoxContentRange
} from '@/types/metadataTypes'

// Set max read high, as atoms are stored near end of file
// while search is made to be efficient.
const MAX_READ_BYTES = 64 * 1024 * 1024

const BOX_TYPES = {
  USER_DATA: [0x75, 0x64, 0x74, 0x61],
  META_DATA: [0x6d, 0x65, 0x74, 0x61],
  ITEM_LIST: [0x69, 0x6c, 0x73, 0x74],
  KEYS: [0x6b, 0x65, 0x79, 0x73],
  DATA: [0x64, 0x61, 0x74, 0x61],
  MOVIE: [0x6d, 0x6f, 0x6f, 0x76]
}
const SIZES = {
  HEADER: 8,
  VERSION: 4,
  LOCALE: 4,
  ITEM_MIN: 8
}

const bufferMatchesBoxType = (
  data: Uint8Array,
  pos: number,
  boxType: number[]
): boolean => {
  if (pos + 4 > data.length) return false

  for (let i = 0; i < 4; i++) {
    if (data[pos + i] !== boxType[i]) return false
  }
  return true
}

const readUint32 = (data: Uint8Array, pos: number): number => {
  if (pos + 4 > data.length) return 0
  return (
    (data[pos] << 24) |
    (data[pos + 1] << 16) |
    (data[pos + 2] << 8) |
    data[pos + 3]
  )
}

const findIsobmffBoxByType = (
  data: Uint8Array,
  startPos: number,
  endPos: number,
  boxType: number[]
): IsobmffBoxContentRange => {
  for (let pos = startPos; pos < endPos - 8; pos++) {
    const size = readUint32(data, pos)
    if (size < SIZES.ITEM_MIN) continue // Minimum size is 8 bytes

    if (bufferMatchesBoxType(data, pos + 4, boxType))
      return { start: pos + SIZES.HEADER, end: pos + size } // Skip header

    // If type doesn't match, ensure size is valid before skipping
    if (pos + size > endPos) return null

    pos += size - 1 // Skip to the next potential box start
  }
  return null
}

const extractJson = (
  data: Uint8Array,
  start: number,
  end: number
): ComfyWorkflowJSON | ComfyApiWorkflow | null => {
  let jsonStart = start
  while (jsonStart < end && data[jsonStart] !== ASCII.OPEN_BRACE) {
    jsonStart++
  }
  if (jsonStart >= end) return null

  try {
    const jsonText = new TextDecoder().decode(data.slice(jsonStart, end))
    return JSON.parse(jsonText)
  } catch {
    return null
  }
}

const readUtf8String = (data: Uint8Array, start: number, end: number): string =>
  new TextDecoder().decode(data.slice(start, end))

const parseKeysBox = (
  data: Uint8Array,
  keysBoxStart: number,
  keysBoxEnd: number
): Map<number, string> => {
  const keysMap = new Map<number, string>()
  let pos = keysBoxStart + 4 // Skip version/flags
  if (pos + 4 > keysBoxEnd) return keysMap

  const entryCount = readUint32(data, pos)
  pos += 4

  for (let i = 1; i <= entryCount; i++) {
    // Keys are 1-indexed
    if (pos + SIZES.HEADER > keysBoxEnd) break

    const keySize = readUint32(data, pos)
    pos += SIZES.HEADER

    const keyNameEnd = pos + keySize - SIZES.HEADER
    if (keySize < SIZES.ITEM_MIN || keyNameEnd > keysBoxEnd) break

    const keyName = readUtf8String(data, pos, keyNameEnd)
    keysMap.set(i, keyName)
    pos = keyNameEnd
  }
  return keysMap
}

const extractMetadataValueFromDataBox = (
  data: Uint8Array,
  dataBoxStart: number,
  dataBoxEnd: number,
  keyName: string
): ComfyWorkflowJSON | ComfyApiWorkflow | null => {
  const valueStart = dataBoxStart + SIZES.VERSION + SIZES.LOCALE
  if (valueStart >= dataBoxEnd) return null

  const lowerKeyName = keyName.toLowerCase()
  if (
    lowerKeyName === ComfyMetadataTags.PROMPT.toLowerCase() ||
    lowerKeyName === ComfyMetadataTags.WORKFLOW.toLowerCase()
  ) {
    return extractJson(data, valueStart, dataBoxEnd)
  }
  return null
}

const parseIlstItem = (
  data: Uint8Array,
  itemStart: number,
  itemEnd: number,
  keysMap: Map<number, string>,
  metadata: ComfyMetadata
) => {
  if (itemStart + SIZES.HEADER > itemEnd) return

  const itemIndex = readUint32(data, itemStart + 4)
  const keyName = keysMap.get(itemIndex)
  if (!keyName) return

  const dataBox = findIsobmffBoxByType(
    data,
    itemStart + SIZES.HEADER,
    itemEnd,
    BOX_TYPES.DATA
  )
  if (dataBox) {
    const value = extractMetadataValueFromDataBox(
      data,
      dataBox.start,
      dataBox.end,
      keyName
    )
    if (value !== null) {
      metadata[keyName.toLowerCase() as keyof ComfyMetadata] = value
    }
  }
}

const parseIlstBox = (
  data: Uint8Array,
  ilstStart: number,
  ilstEnd: number,
  keysMap: Map<number, string>,
  metadata: ComfyMetadata
) => {
  let pos = ilstStart
  while (pos < ilstEnd - SIZES.HEADER) {
    const itemSize = readUint32(data, pos)
    if (itemSize <= SIZES.HEADER || pos + itemSize > ilstEnd) break // Invalid item size
    parseIlstItem(data, pos, pos + itemSize, keysMap, metadata)
    pos += itemSize
  }
}

const findUserDataBox = (data: Uint8Array): IsobmffBoxContentRange => {
  let userDataBox: IsobmffBoxContentRange = null

  // Metadata can be in 'udta' at top level or inside 'moov'
  userDataBox = findIsobmffBoxByType(data, 0, data.length, BOX_TYPES.USER_DATA)

  if (!userDataBox) {
    const moovBox = findIsobmffBoxByType(data, 0, data.length, BOX_TYPES.MOVIE)
    if (moovBox) {
      userDataBox = findIsobmffBoxByType(
        data,
        moovBox.start,
        moovBox.end,
        BOX_TYPES.USER_DATA
      )
    }
  }
  return userDataBox
}

const parseIsobmffMetadata = (data: Uint8Array): ComfyMetadata => {
  const metadata: ComfyMetadata = {}
  const userDataBox = findUserDataBox(data)
  if (!userDataBox) return metadata

  const metaBox = findIsobmffBoxByType(
    data,
    userDataBox.start,
    userDataBox.end,
    BOX_TYPES.META_DATA
  )
  if (!metaBox) return metadata

  const metaContentStart = metaBox.start + SIZES.VERSION
  const keysBox = findIsobmffBoxByType(
    data,
    metaContentStart,
    metaBox.end,
    BOX_TYPES.KEYS
  )
  if (!keysBox) return metadata

  const keysMap = parseKeysBox(data, keysBox.start, keysBox.end)
  if (keysMap.size === 0) return metadata // keys box is empty or failed to parse

  const ilstBox = findIsobmffBoxByType(
    data,
    metaContentStart,
    metaBox.end,
    BOX_TYPES.ITEM_LIST
  )
  if (!ilstBox) return metadata

  parseIlstBox(data, ilstBox.start, ilstBox.end, keysMap, metadata)

  return metadata
}

/**
 * Extracts ComfyUI Workflow metadata from an ISO Base Media File Format (ISOBMFF) file
 * (e.g., MP4, MOV) by parsing the `udta.meta.keys` and `udta.meta.ilst` boxes.
 * @param file - The file to extract metadata from.
 */
export function getFromIsobmffFile(file: File): Promise<ComfyMetadata> {
  return new Promise<ComfyMetadata>((resolve) => {
    const reader = new FileReader()
    reader.onload = (event: ProgressEvent<FileReader>) => {
      if (!event.target?.result) {
        resolve({})
        return
      }

      try {
        const data = new Uint8Array(event.target.result as ArrayBuffer)
        resolve(parseIsobmffMetadata(data))
      } catch (e) {
        console.error('Parser: Error parsing ISOBMFF metadata:', e)
        resolve({})
      }
    }
    reader.onerror = (err) => {
      console.error('FileReader: Error reading ISOBMFF file:', err)
      resolve({})
    }
    reader.readAsArrayBuffer(file.slice(0, MAX_READ_BYTES))
  })
}
