import { describe, expect, it } from 'vitest'

import { getFromPngBuffer } from './png'

function createPngWithChunk(
  chunkType: string,
  keyword: string,
  content: string,
  options: {
    compressionFlag?: number
    compressionMethod?: number
    languageTag?: string
    translatedKeyword?: string
  } = {}
): ArrayBuffer {
  const {
    compressionFlag = 0,
    compressionMethod = 0,
    languageTag = '',
    translatedKeyword = ''
  } = options

  const signature = new Uint8Array([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a
  ])
  const typeBytes = new TextEncoder().encode(chunkType)
  const keywordBytes = new TextEncoder().encode(keyword)
  const contentBytes = new TextEncoder().encode(content)

  let chunkData: Uint8Array
  if (chunkType === 'iTXt') {
    const langBytes = new TextEncoder().encode(languageTag)
    const transBytes = new TextEncoder().encode(translatedKeyword)
    const totalLength =
      keywordBytes.length +
      1 +
      2 +
      langBytes.length +
      1 +
      transBytes.length +
      1 +
      contentBytes.length

    chunkData = new Uint8Array(totalLength)
    let pos = 0
    chunkData.set(keywordBytes, pos)
    pos += keywordBytes.length
    chunkData[pos++] = 0
    chunkData[pos++] = compressionFlag
    chunkData[pos++] = compressionMethod
    chunkData.set(langBytes, pos)
    pos += langBytes.length
    chunkData[pos++] = 0
    chunkData.set(transBytes, pos)
    pos += transBytes.length
    chunkData[pos++] = 0
    chunkData.set(contentBytes, pos)
  } else {
    chunkData = new Uint8Array(keywordBytes.length + 1 + contentBytes.length)
    chunkData.set(keywordBytes, 0)
    chunkData[keywordBytes.length] = 0
    chunkData.set(contentBytes, keywordBytes.length + 1)
  }

  const lengthBytes = new Uint8Array(4)
  new DataView(lengthBytes.buffer).setUint32(0, chunkData.length, false)

  const crc = new Uint8Array(4)

  const iendType = new TextEncoder().encode('IEND')
  const iendLength = new Uint8Array(4)
  const iendCrc = new Uint8Array(4)

  const total = signature.length + 4 + 4 + chunkData.length + 4 + 4 + 4 + 0 + 4
  const result = new Uint8Array(total)

  let offset = 0
  result.set(signature, offset)
  offset += signature.length

  result.set(lengthBytes, offset)
  offset += 4
  result.set(typeBytes, offset)
  offset += 4
  result.set(chunkData, offset)
  offset += chunkData.length
  result.set(crc, offset)
  offset += 4

  result.set(iendLength, offset)
  offset += 4
  result.set(iendType, offset)
  offset += 4
  result.set(iendCrc, offset)

  return result.buffer
}

describe('getFromPngBuffer', () => {
  it('returns empty object for invalid PNG', async () => {
    const invalidData = new ArrayBuffer(8)
    const result = await getFromPngBuffer(invalidData)
    expect(result).toEqual({})
  })

  it('parses tEXt chunk', async () => {
    const workflow = '{"nodes":[]}'
    const buffer = createPngWithChunk('tEXt', 'workflow', workflow)
    const result = await getFromPngBuffer(buffer)
    expect(result['workflow']).toBe(workflow)
  })

  it('parses comf chunk', async () => {
    const prompt = '{"1":{"class_type":"Test"}}'
    const buffer = createPngWithChunk('comf', 'prompt', prompt)
    const result = await getFromPngBuffer(buffer)
    expect(result['prompt']).toBe(prompt)
  })

  it('parses uncompressed iTXt chunk', async () => {
    const workflow = '{"nodes":[{"id":1}]}'
    const buffer = createPngWithChunk('iTXt', 'workflow', workflow, {
      compressionFlag: 0,
      compressionMethod: 0
    })
    const result = await getFromPngBuffer(buffer)
    expect(result['workflow']).toBe(workflow)
  })

  it('parses iTXt chunk with language tag and translated keyword', async () => {
    const workflow = '{"test":"value"}'
    const buffer = createPngWithChunk('iTXt', 'workflow', workflow, {
      compressionFlag: 0,
      languageTag: 'en',
      translatedKeyword: 'Workflow'
    })
    const result = await getFromPngBuffer(buffer)
    expect(result['workflow']).toBe(workflow)
  })

  it('parses compressed iTXt chunk', async () => {
    const workflow = '{"nodes":[{"id":1,"type":"KSampler"}]}'
    const contentBytes = new TextEncoder().encode(workflow)

    const stream = new CompressionStream('deflate')
    const writer = stream.writable.getWriter()
    await writer.write(contentBytes)
    await writer.close()

    const reader = stream.readable.getReader()
    const chunks: Uint8Array[] = []
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
    }
    const compressedBytes = new Uint8Array(
      chunks.reduce((acc, c) => acc + c.length, 0)
    )
    let pos = 0
    for (const chunk of chunks) {
      compressedBytes.set(chunk, pos)
      pos += chunk.length
    }

    const buffer = createPngWithCompressedITXt(
      'workflow',
      compressedBytes,
      '',
      ''
    )
    const result = await getFromPngBuffer(buffer)
    expect(result['workflow']).toBe(workflow)
  })
})

function createPngWithCompressedITXt(
  keyword: string,
  compressedContent: Uint8Array,
  languageTag: string,
  translatedKeyword: string
): ArrayBuffer {
  const signature = new Uint8Array([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a
  ])
  const typeBytes = new TextEncoder().encode('iTXt')
  const keywordBytes = new TextEncoder().encode(keyword)
  const langBytes = new TextEncoder().encode(languageTag)
  const transBytes = new TextEncoder().encode(translatedKeyword)

  const totalLength =
    keywordBytes.length +
    1 +
    2 +
    langBytes.length +
    1 +
    transBytes.length +
    1 +
    compressedContent.length

  const chunkData = new Uint8Array(totalLength)
  let pos = 0
  chunkData.set(keywordBytes, pos)
  pos += keywordBytes.length
  chunkData[pos++] = 0
  chunkData[pos++] = 1
  chunkData[pos++] = 0
  chunkData.set(langBytes, pos)
  pos += langBytes.length
  chunkData[pos++] = 0
  chunkData.set(transBytes, pos)
  pos += transBytes.length
  chunkData[pos++] = 0
  chunkData.set(compressedContent, pos)

  const lengthBytes = new Uint8Array(4)
  new DataView(lengthBytes.buffer).setUint32(0, chunkData.length, false)

  const crc = new Uint8Array(4)
  const iendType = new TextEncoder().encode('IEND')
  const iendLength = new Uint8Array(4)
  const iendCrc = new Uint8Array(4)

  const total = signature.length + 4 + 4 + chunkData.length + 4 + 4 + 4 + 0 + 4
  const result = new Uint8Array(total)

  let offset = 0
  result.set(signature, offset)
  offset += signature.length
  result.set(lengthBytes, offset)
  offset += 4
  result.set(typeBytes, offset)
  offset += 4
  result.set(chunkData, offset)
  offset += chunkData.length
  result.set(crc, offset)
  offset += 4
  result.set(iendLength, offset)
  offset += 4
  result.set(iendType, offset)
  offset += 4
  result.set(iendCrc, offset)

  return result.buffer
}
