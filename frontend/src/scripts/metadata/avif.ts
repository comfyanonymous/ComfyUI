import {
  type AvifIinfBox,
  type AvifIlocBox,
  type AvifInfeBox,
  type ComfyMetadata,
  ComfyMetadataTags,
  type IsobmffBoxContentRange
} from '@/types/metadataTypes'

const readNullTerminatedString = (
  dataView: DataView,
  start: number,
  end: number
): { str: string; length: number } => {
  let length = 0
  while (start + length < end && dataView.getUint8(start + length) !== 0) {
    length++
  }
  const str = new TextDecoder('utf-8').decode(
    new Uint8Array(dataView.buffer, dataView.byteOffset + start, length)
  )
  return { str, length: length + 1 } // Include null terminator
}

const parseInfeBox = (dataView: DataView, start: number): AvifInfeBox => {
  const version = dataView.getUint8(start)
  const flags = dataView.getUint32(start) & 0xffffff
  let offset = start + 4

  let item_ID: number, item_protection_index: number, item_type: string

  if (version >= 2) {
    if (version === 2) {
      item_ID = dataView.getUint16(offset)
      offset += 2
    } else {
      item_ID = dataView.getUint32(offset)
      offset += 4
    }

    item_protection_index = dataView.getUint16(offset)
    offset += 2
    item_type = String.fromCharCode(
      ...new Uint8Array(dataView.buffer, dataView.byteOffset + offset, 4)
    )
    offset += 4

    const { str: item_name, length: name_len } = readNullTerminatedString(
      dataView,
      offset,
      dataView.byteLength
    )
    offset += name_len

    const content_type = readNullTerminatedString(
      dataView,
      offset,
      dataView.byteLength
    ).str

    return {
      box_header: { size: 0, type: 'infe' }, // Size is dynamic
      version,
      flags,
      item_ID,
      item_protection_index,
      item_type,
      item_name,
      content_type
    }
  }
  throw new Error(`Unsupported infe box version: ${version}`)
}

const parseIinfBox = (
  dataView: DataView,
  range: IsobmffBoxContentRange
): AvifIinfBox => {
  if (!range) throw new Error('iinf box not found')

  const version = dataView.getUint8(range.start)
  const flags = dataView.getUint32(range.start) & 0xffffff
  let offset = range.start + 4

  const entry_count =
    version === 0 ? dataView.getUint16(offset) : dataView.getUint32(offset)
  offset += version === 0 ? 2 : 4

  const entries: AvifInfeBox[] = []
  for (let i = 0; i < entry_count; i++) {
    const boxSize = dataView.getUint32(offset)
    const boxType = String.fromCharCode(
      ...new Uint8Array(dataView.buffer, dataView.byteOffset + offset + 4, 4)
    )

    if (boxType === 'infe') {
      const infe = parseInfeBox(dataView, offset + 8)
      infe.box_header.size = boxSize
      entries.push(infe)
    }
    offset += boxSize
  }

  return {
    box_header: { size: range.end - range.start + 8, type: 'iinf' },
    version,
    flags,
    entry_count,
    entries
  }
}

const parseIlocBox = (
  dataView: DataView,
  range: IsobmffBoxContentRange
): AvifIlocBox => {
  if (!range) throw new Error('iloc box not found')

  const version = dataView.getUint8(range.start)
  const flags = dataView.getUint32(range.start) & 0xffffff
  let offset = range.start + 4

  const sizes = dataView.getUint8(offset++)
  const offset_size = (sizes >> 4) & 0x0f
  const length_size = sizes & 0x0f

  const base_offset_size = (dataView.getUint8(offset) >> 4) & 0x0f
  const index_size =
    version === 1 || version === 2 ? dataView.getUint8(offset) & 0x0f : 0
  offset++

  const item_count =
    version < 2 ? dataView.getUint16(offset) : dataView.getUint32(offset)
  offset += version < 2 ? 2 : 4

  const items = []
  for (let i = 0; i < item_count; i++) {
    const item_ID =
      version < 2 ? dataView.getUint16(offset) : dataView.getUint32(offset)
    offset += version < 2 ? 2 : 4

    if (version === 1 || version === 2) {
      offset += 2 // construction_method
    }

    const data_reference_index = dataView.getUint16(offset)
    offset += 2

    const base_offset = base_offset_size > 0 ? dataView.getUint32(offset) : 0 // Simplified
    offset += base_offset_size

    const extent_count = dataView.getUint16(offset)
    offset += 2

    const extents = []
    for (let j = 0; j < extent_count; j++) {
      if ((version === 1 || version === 2) && index_size > 0) {
        offset += index_size
      }
      const extent_offset = dataView.getUint32(offset) // Simplified
      offset += offset_size
      const extent_length = dataView.getUint32(offset) // Simplified
      offset += length_size
      extents.push({ extent_offset, extent_length })
    }
    items.push({
      item_ID,
      data_reference_index,
      base_offset,
      extent_count,
      extents
    })
  }

  return {
    box_header: { size: range.end - range.start + 8, type: 'iloc' },
    version,
    flags,
    offset_size,
    length_size,
    base_offset_size,
    index_size,
    item_count,
    items
  }
}

function findBox(
  dataView: DataView,
  start: number,
  end: number,
  type: string
): IsobmffBoxContentRange {
  let offset = start
  while (offset < end) {
    if (offset + 8 > end) break
    const boxLength = dataView.getUint32(offset)
    const boxType = String.fromCharCode(
      ...new Uint8Array(dataView.buffer, dataView.byteOffset + offset + 4, 4)
    )

    if (boxLength === 0) break

    if (boxType === type) {
      return { start: offset + 8, end: offset + boxLength }
    }
    if (offset + boxLength > end) break
    offset += boxLength
  }
  return null
}

function parseAvifMetadata(buffer: ArrayBuffer): ComfyMetadata {
  const metadata: ComfyMetadata = {}
  const dataView = new DataView(buffer)

  if (
    dataView.getUint32(4) !== 0x66747970 ||
    dataView.getUint32(8) !== 0x61766966
  ) {
    console.error('Not a valid AVIF file')
    return {}
  }

  const metaBox = findBox(dataView, 0, dataView.byteLength, 'meta')
  if (!metaBox) return {}

  const metaBoxContentStart = metaBox.start + 4 // Skip version and flags

  const iinfBoxRange = findBox(
    dataView,
    metaBoxContentStart,
    metaBox.end,
    'iinf'
  )
  const iinf = parseIinfBox(dataView, iinfBoxRange)

  const exifInfe = iinf.entries.find((e) => e.item_type === 'Exif')
  if (!exifInfe) return {}

  const ilocBoxRange = findBox(
    dataView,
    metaBoxContentStart,
    metaBox.end,
    'iloc'
  )
  const iloc = parseIlocBox(dataView, ilocBoxRange)

  const exifIloc = iloc.items.find((i) => i.item_ID === exifInfe.item_ID)
  if (!exifIloc || exifIloc.extents.length === 0) return {}

  const exifExtent = exifIloc.extents[0]
  const itemData = new Uint8Array(
    buffer,
    exifExtent.extent_offset,
    exifExtent.extent_length
  )

  let tiffHeaderOffset = -1
  for (let i = 0; i < itemData.length - 4; i++) {
    if (
      (itemData[i] === 0x4d &&
        itemData[i + 1] === 0x4d &&
        itemData[i + 2] === 0x00 &&
        itemData[i + 3] === 0x2a) || // MM*
      (itemData[i] === 0x49 &&
        itemData[i + 1] === 0x49 &&
        itemData[i + 2] === 0x2a &&
        itemData[i + 3] === 0x00) // II*
    ) {
      tiffHeaderOffset = i
      break
    }
  }

  if (tiffHeaderOffset !== -1) {
    const exifData = itemData.subarray(tiffHeaderOffset)
    const data: Record<string, any> = parseExifData(exifData)
    for (const key in data) {
      const value = data[key]
      if (typeof value === 'string') {
        if (key === 'usercomment') {
          try {
            const metadataJson = JSON.parse(value)
            if (metadataJson.prompt) {
              metadata[ComfyMetadataTags.PROMPT] = metadataJson.prompt
            }
            if (metadataJson.workflow) {
              metadata[ComfyMetadataTags.WORKFLOW] = metadataJson.workflow
            }
          } catch (e) {
            console.error('Failed to parse usercomment JSON', e)
          }
        } else {
          const [metadataKey, ...metadataValueParts] = value.split(':')
          const metadataValue = metadataValueParts.join(':').trim()
          if (
            metadataKey.toLowerCase() ===
              ComfyMetadataTags.PROMPT.toLowerCase() ||
            metadataKey.toLowerCase() ===
              ComfyMetadataTags.WORKFLOW.toLowerCase()
          ) {
            try {
              const jsonValue = JSON.parse(metadataValue)
              metadata[metadataKey.toLowerCase() as keyof ComfyMetadata] =
                jsonValue
            } catch (e) {
              console.error(`Failed to parse JSON for ${metadataKey}`, e)
            }
          }
        }
      }
    }
  } else {
    console.log('Warning: TIFF header not found in EXIF data.')
  }

  return metadata
}

// @ts-expect-error fixme ts strict error
function parseExifData(exifData) {
  // Check for the correct TIFF header (0x4949 for little-endian or 0x4D4D for big-endian)
  const isLittleEndian = String.fromCharCode(...exifData.slice(0, 2)) === 'II'

  // Function to read 16-bit and 32-bit integers from binary data
  // @ts-expect-error fixme ts strict error
  function readInt(offset, isLittleEndian, length) {
    let arr = exifData.slice(offset, offset + length)
    if (length === 2) {
      return new DataView(arr.buffer, arr.byteOffset, arr.byteLength).getUint16(
        0,
        isLittleEndian
      )
    } else if (length === 4) {
      return new DataView(arr.buffer, arr.byteOffset, arr.byteLength).getUint32(
        0,
        isLittleEndian
      )
    }
  }

  // Read the offset to the first IFD (Image File Directory)
  const ifdOffset = readInt(4, isLittleEndian, 4)

  // @ts-expect-error fixme ts strict error
  function parseIFD(offset) {
    const numEntries = readInt(offset, isLittleEndian, 2)
    const result = {}

    // @ts-expect-error fixme ts strict error
    for (let i = 0; i < numEntries; i++) {
      const entryOffset = offset + 2 + i * 12
      const tag = readInt(entryOffset, isLittleEndian, 2)
      const type = readInt(entryOffset + 2, isLittleEndian, 2)
      const numValues = readInt(entryOffset + 4, isLittleEndian, 4)
      const valueOffset = readInt(entryOffset + 8, isLittleEndian, 4)

      // Read the value(s) based on the data type
      let value
      if (type === 2) {
        // ASCII string
        value = new TextDecoder('utf-8').decode(
          // @ts-expect-error fixme ts strict error
          exifData.subarray(valueOffset, valueOffset + numValues - 1)
        )
      }

      // @ts-expect-error fixme ts strict error
      result[tag] = value
    }

    return result
  }

  // Parse the first IFD
  const ifdData = parseIFD(ifdOffset)
  return ifdData
}

export function getFromAvifFile(file: File): Promise<Record<string, string>> {
  return new Promise<Record<string, string>>((resolve) => {
    const reader = new FileReader()
    reader.onload = (event) => {
      const buffer = event.target?.result as ArrayBuffer
      if (!buffer) {
        resolve({})
        return
      }

      try {
        const comfyMetadata = parseAvifMetadata(buffer)
        const result: Record<string, string> = {}
        if (comfyMetadata.prompt) {
          result.prompt = JSON.stringify(comfyMetadata.prompt)
        }
        if (comfyMetadata.workflow) {
          result.workflow = JSON.stringify(comfyMetadata.workflow)
        }
        resolve(result)
      } catch (e) {
        console.error('Parser: Error parsing AVIF metadata:', e)
        resolve({})
      }
    }
    reader.onerror = (err) => {
      console.error('FileReader: Error reading AVIF file:', err)
      resolve({})
    }
    reader.readAsArrayBuffer(file)
  })
}
