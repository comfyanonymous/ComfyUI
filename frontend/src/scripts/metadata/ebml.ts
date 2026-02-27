import {
  type ComfyApiWorkflow,
  type ComfyWorkflowJSON
} from '@/platform/workflow/validation/schemas/workflowSchema'
import {
  type ComfyMetadata,
  ComfyMetadataTags,
  type EbmlElementRange,
  type EbmlTagPosition,
  type TextRange,
  type VInt
} from '@/types/metadataTypes'

const WEBM_SIGNATURE = [0x1a, 0x45, 0xdf, 0xa3]
const MAX_READ_BYTES = 2 * 1024 * 1024
const EBML_ID = {
  SIMPLE_TAG: new Uint8Array([0x67, 0xc8]),
  TAG_NAME: new Uint8Array([0x45, 0xa3]),
  TAG_VALUE: new Uint8Array([0x44, 0x87])
}
const ASCII = {
  OPEN_BRACE: 0x7b,
  NULL: 0,
  PRINTABLE_MIN: 32,
  PRINTABLE_MAX: 126
}
const MASK = {
  MSB: 0x80,
  ALL_BITS_SET: -1
}

const hasWebmSignature = (data: Uint8Array): boolean =>
  WEBM_SIGNATURE.every((byte, index) => data[index] === byte)

const readVint = (data: Uint8Array, pos: number): VInt | null => {
  if (pos >= data.length) return null

  const byte = data[pos]

  // Fast path for common case (1-byte vint)
  if ((byte & MASK.MSB) === MASK.MSB) {
    return { value: byte & ~MASK.MSB, length: 1 }
  }

  const length = findFirstSetBitPosition(byte)
  if (length === 0 || pos + length > data.length) {
    return null
  }

  return {
    value: calculateVintValue(data, pos, length),
    length
  }
}

const calculateVintValue = (
  data: Uint8Array,
  pos: number,
  length: number
): number => {
  let value = data[pos] & (0xff >> length)

  for (let i = 1; i < length; i++) {
    value = (value << 8) | data[pos + i]
  }

  const allBitsSet = Math.pow(2, 7 * length) - 1
  return value === allBitsSet ? MASK.ALL_BITS_SET : value
}

const findFirstSetBitPosition = (byte: number): number => {
  for (let mask = MASK.MSB, position = 1; mask !== 0; mask >>= 1, position++) {
    if ((byte & mask) !== 0) return position
  }
  return 0
}

const matchesId = (data: Uint8Array, pos: number, id: Uint8Array): boolean => {
  if (pos + id.length > data.length) return false

  for (let i = 0; i < id.length; i++) {
    if (data[pos + i] !== id[i]) return false
  }

  return true
}

const findNextTag = (
  data: Uint8Array,
  pos: number
): { tagEnd: number; contents: EbmlTagPosition | null } | null => {
  if (!matchesId(data, pos, EBML_ID.SIMPLE_TAG)) return null

  const tagSize = readVint(data, pos + 2)
  if (!tagSize || tagSize.value <= 0) return null

  const tagEnd = pos + 2 + tagSize.length + tagSize.value
  if (tagEnd > data.length) return null

  const contents = extractTagContents(data, pos + 2 + tagSize.length, tagEnd)
  return { tagEnd, contents }
}

const extractTagContents = (
  data: Uint8Array,
  start: number,
  end: number
): EbmlTagPosition | null => {
  const nameInfo = findElementsInTag(data, start, end, EBML_ID.TAG_NAME)
  const valueInfo = findElementsInTag(data, start, end, EBML_ID.TAG_VALUE)

  if (!nameInfo || !valueInfo) return null

  return {
    name: nameInfo,
    value: valueInfo
  }
}

const findElementsInTag = (
  data: Uint8Array,
  start: number,
  end: number,
  id: Uint8Array
): EbmlElementRange | null => {
  for (let pos = start; pos < end - 1; ) {
    if (matchesId(data, pos, id)) {
      const size = readVint(data, pos + 2)
      if (size && size.value > 0) {
        const elementPos = pos + 2 + size.length
        return { pos: elementPos, len: size.value }
      }
    }
    pos++
  }

  return null
}

const processTagContents = (
  data: Uint8Array,
  contents: EbmlTagPosition,
  meta: ComfyMetadata
) => {
  try {
    const name = extractEbmlTagName(data, contents.name.pos, contents.name.len)
    if (!name) return

    const value = extractEbmlTagValue(
      data,
      name,
      contents.value.pos,
      contents.value.len
    )
    if (value !== null) {
      meta[name.toLowerCase()] = value
    }
  } catch {
    // Silently continue on error
  }
}

const extractEbmlTagValue = (
  data: Uint8Array,
  name: string,
  pos: number,
  len: number
): string | ComfyWorkflowJSON | ComfyApiWorkflow | null => {
  if (
    name === ComfyMetadataTags.PROMPT ||
    name === ComfyMetadataTags.WORKFLOW
  ) {
    return readJson(data, pos, len)
  }

  return ebmlToString(data, pos, len)
}

const extractEbmlTagName = (
  data: Uint8Array,
  start: number,
  length: number
): string | null => {
  if (length <= 0) return null

  const textRange = findReadableTextRange(data, start, length)
  if (!textRange) return null

  return new TextDecoder()
    .decode(data.subarray(textRange.start, textRange.end))
    .trim()
}

const findReadableTextRange = (
  data: Uint8Array,
  start: number,
  length: number
): TextRange | null => {
  const isPrintableAscii = (byte: number) =>
    byte >= ASCII.PRINTABLE_MIN && byte <= ASCII.PRINTABLE_MAX

  let textStart = start
  while (textStart < start + length && !isPrintableAscii(data[textStart])) {
    textStart++
  }

  if (textStart >= start + length) return null

  let textEnd = textStart
  while (textEnd < start + length && data[textEnd] !== ASCII.NULL) {
    textEnd++
  }

  return { start: textStart, end: textEnd }
}

const readJson = (
  data: Uint8Array,
  start: number,
  length: number
): ComfyWorkflowJSON | ComfyApiWorkflow | null => {
  if (length <= 0) return null

  const nullTerminatorPos = findNullTerminator(data, start, length)
  const jsonStartPos = findJsonStart(data, start, nullTerminatorPos - start)

  if (jsonStartPos === null) return null

  const jsonText = decodeJsonText(data, jsonStartPos, nullTerminatorPos)
  return parseJsonText(jsonText)
}

const decodeJsonText = (
  data: Uint8Array,
  start: number,
  end: number
): string => {
  return new TextDecoder().decode(data.subarray(start, end))
}

const parseJsonText = (
  jsonText: string
): ComfyWorkflowJSON | ComfyApiWorkflow | null => {
  const jsonEndPos = findJsonEnd(jsonText)
  if (jsonEndPos === null) return null

  try {
    return JSON.parse(jsonText.substring(0, jsonEndPos))
  } catch {
    return null
  }
}

const findNullTerminator = (
  data: Uint8Array,
  start: number,
  maxLength: number
): number => {
  const end = Math.min(start + maxLength, data.length)

  for (let pos = start; pos < end; pos++) {
    if (data[pos] === ASCII.NULL) return pos
  }

  return end
}

const findJsonStart = (
  data: Uint8Array,
  start: number,
  length: number
): number | null => {
  for (let pos = start; pos < start + length; pos++) {
    if (data[pos] === ASCII.OPEN_BRACE) return pos
  }

  return null
}

const findJsonEnd = (text: string): number | null => {
  let braceCount = 1
  let pos = 1 // Start after the opening brace

  while (braceCount > 0 && pos < text.length) {
    if (text[pos] === '{') braceCount++
    if (text[pos] === '}') braceCount--
    pos++
  }

  return braceCount === 0 ? pos : null
}

const ebmlToString = (
  data: Uint8Array,
  start: number,
  length: number
): string => {
  if (length <= 0) return ''

  const endPos = findNullTerminator(data, start, length)
  return new TextDecoder().decode(data.subarray(start, endPos)).trim()
}

const parseMetadata = (data: Uint8Array): ComfyMetadata => {
  const meta: ComfyMetadata = {}

  for (let pos = 0; pos < data.length - 2; ) {
    const tagInfo = findNextTag(data, pos)
    if (!tagInfo) {
      pos++
      continue
    }

    const { tagEnd, contents } = tagInfo
    if (contents) {
      processTagContents(data, contents, meta)
    }

    pos = tagEnd
  }

  return meta
}

const handleFileLoad = (
  event: ProgressEvent<FileReader>,
  resolve: (value: ComfyMetadata) => void
) => {
  if (!event.target?.result) {
    resolve({})
    return
  }

  try {
    const data = new Uint8Array(event.target.result as ArrayBuffer)
    if (data.length < 4 || !hasWebmSignature(data)) {
      resolve({})
      return
    }

    resolve(parseMetadata(data))
  } catch {
    resolve({})
  }
}

/**
 * Extracts ComfyUI Workflow metadata from a WebM file
 * @param file - The WebM file to extract metadata from
 */
export function getFromWebmFile(file: File): Promise<ComfyMetadata> {
  return new Promise<ComfyMetadata>((resolve) => {
    const reader = new FileReader()
    reader.onload = (event) => handleFileLoad(event, resolve)
    reader.onerror = () => resolve({})
    reader.readAsArrayBuffer(file.slice(0, MAX_READ_BYTES))
  })
}
