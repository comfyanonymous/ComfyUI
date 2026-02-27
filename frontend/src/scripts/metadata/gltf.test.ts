import { describe, expect, it } from 'vitest'

import { ASCII, GltfSizeBytes } from '@/types/metadataTypes'

import { getGltfBinaryMetadata } from './gltf'

describe('GLTF binary metadata parser', () => {
  const createGLTFFileStructure = () => {
    const header = new ArrayBuffer(GltfSizeBytes.HEADER)
    const headerView = new DataView(header)
    return { header, headerView }
  }

  const jsonToBinary = (json: object) => {
    const jsonString = JSON.stringify(json)
    const jsonData = new TextEncoder().encode(jsonString)
    return jsonData
  }

  const createJSONChunk = (jsonData: ArrayBuffer) => {
    const chunkHeader = new ArrayBuffer(GltfSizeBytes.CHUNK_HEADER)
    const chunkView = new DataView(chunkHeader)
    chunkView.setUint32(0, jsonData.byteLength, true)
    chunkView.setUint32(4, ASCII.JSON, true)
    return chunkHeader
  }

  const setVersionHeader = (headerView: DataView, version: number) => {
    headerView.setUint32(4, version, true)
  }

  const setTypeHeader = (headerView: DataView, type: number) => {
    headerView.setUint32(0, type, true)
  }

  const setTotalLengthHeader = (headerView: DataView, length: number) => {
    headerView.setUint32(8, length, true)
  }

  const setHeaders = (headerView: DataView, jsonData: ArrayBuffer) => {
    setTypeHeader(headerView, ASCII.GLTF)
    setVersionHeader(headerView, 2)
    setTotalLengthHeader(
      headerView,
      GltfSizeBytes.HEADER + GltfSizeBytes.CHUNK_HEADER + jsonData.byteLength
    )
  }

  function createMockGltfFile(jsonContent: object): File {
    const jsonData = jsonToBinary(jsonContent)
    const { header, headerView } = createGLTFFileStructure()

    setHeaders(headerView, jsonData.buffer)

    const chunkHeader = createJSONChunk(jsonData.buffer)

    const fileContent = new Uint8Array(
      header.byteLength + chunkHeader.byteLength + jsonData.byteLength
    )
    fileContent.set(new Uint8Array(header), 0)
    fileContent.set(new Uint8Array(chunkHeader), header.byteLength)
    fileContent.set(jsonData, header.byteLength + chunkHeader.byteLength)

    return new File([fileContent], 'test.glb', { type: 'model/gltf-binary' })
  }

  it('should extract workflow metadata from GLTF binary file', async () => {
    const testWorkflow = {
      nodes: [
        {
          id: 1,
          type: 'TestNode',
          pos: [100, 100]
        }
      ],
      links: []
    }

    const mockFile = createMockGltfFile({
      asset: {
        version: '2.0',
        generator: 'ComfyUI GLTF Test',
        extras: {
          workflow: testWorkflow
        }
      },
      scenes: []
    })

    const metadata = await getGltfBinaryMetadata(mockFile)

    expect(metadata).toBeDefined()
    expect(metadata.workflow).toBeDefined()

    const workflow = metadata.workflow as {
      nodes: Array<{ id: number; type: string }>
    }
    expect(workflow.nodes[0].id).toBe(1)
    expect(workflow.nodes[0].type).toBe('TestNode')
  })

  it('should extract prompt metadata from GLTF binary file', async () => {
    const testPrompt = {
      node1: {
        class_type: 'TestNode',
        inputs: {
          seed: 123456
        }
      }
    }

    const mockFile = createMockGltfFile({
      asset: {
        version: '2.0',
        generator: 'ComfyUI GLTF Test',
        extras: {
          prompt: testPrompt
        }
      },
      scenes: []
    })

    const metadata = await getGltfBinaryMetadata(mockFile)
    expect(metadata).toBeDefined()
    expect(metadata.prompt).toBeDefined()

    const prompt = metadata.prompt as Record<string, any>
    expect(prompt.node1.class_type).toBe('TestNode')
    expect(prompt.node1.inputs.seed).toBe(123456)
  })

  it('should handle string JSON content', async () => {
    const workflowStr = JSON.stringify({
      nodes: [{ id: 1, type: 'StringifiedNode' }],
      links: []
    })

    const mockFile = createMockGltfFile({
      asset: {
        version: '2.0',
        extras: {
          workflow: workflowStr // As string instead of object
        }
      }
    })

    const metadata = await getGltfBinaryMetadata(mockFile)

    expect(metadata).toBeDefined()
    expect(metadata.workflow).toBeDefined()

    const workflow = metadata.workflow as {
      nodes: Array<{ id: number; type: string }>
    }
    expect(workflow.nodes[0].type).toBe('StringifiedNode')
  })

  it('should handle invalid GLTF binary files gracefully', async () => {
    const invalidEmptyFile = new File([], 'invalid.glb')
    const metadata = await getGltfBinaryMetadata(invalidEmptyFile)
    expect(metadata).toEqual({})
  })
})
