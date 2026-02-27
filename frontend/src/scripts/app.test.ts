import { beforeEach, describe, expect, it, vi } from 'vitest'

import type {
  LGraph,
  LGraphCanvas,
  LGraphNode
} from '@/lib/litegraph/src/litegraph'
import { ComfyApp } from './app'
import { createNode } from '@/utils/litegraphUtil'
import {
  pasteAudioNode,
  pasteAudioNodes,
  pasteImageNode,
  pasteImageNodes,
  pasteVideoNode,
  pasteVideoNodes
} from '@/composables/usePaste'
import { getWorkflowDataFromFile } from '@/scripts/metadata/parser'

vi.mock('@/utils/litegraphUtil', () => ({
  createNode: vi.fn(),
  isImageNode: vi.fn(),
  isVideoNode: vi.fn(),
  isAudioNode: vi.fn(),
  executeWidgetsCallback: vi.fn(),
  fixLinkInputSlots: vi.fn()
}))

vi.mock('@/composables/usePaste', () => ({
  pasteAudioNode: vi.fn(),
  pasteAudioNodes: vi.fn(),
  pasteImageNode: vi.fn(),
  pasteImageNodes: vi.fn(),
  pasteVideoNode: vi.fn(),
  pasteVideoNodes: vi.fn()
}))

vi.mock('@/scripts/metadata/parser', () => ({
  getWorkflowDataFromFile: vi.fn()
}))

vi.mock('@/platform/updates/common/toastStore', () => ({
  useToastStore: vi.fn(() => ({
    addAlert: vi.fn(),
    add: vi.fn(),
    remove: vi.fn()
  }))
}))

function createMockNode(options: { [K in keyof LGraphNode]?: any } = {}) {
  return {
    id: 1,
    pos: [0, 0],
    size: [200, 100],
    type: 'LoadImage',
    connect: vi.fn(),
    getBounding: vi.fn(() => new Float64Array([0, 0, 200, 100])),
    ...options
  } as LGraphNode
}

function createMockCanvas(): Partial<LGraphCanvas> {
  const mockGraph: Partial<LGraph> = {
    change: vi.fn()
  }

  return {
    graph: mockGraph as LGraph,
    selectItems: vi.fn()
  }
}

function createTestFile(name: string, type: string): File {
  return new File([''], name, { type })
}

describe('ComfyApp', () => {
  let app: ComfyApp
  let mockCanvas: LGraphCanvas

  beforeEach(() => {
    vi.clearAllMocks()
    app = new ComfyApp()
    mockCanvas = createMockCanvas() as LGraphCanvas
    app.canvas = mockCanvas as LGraphCanvas
  })

  describe('handleFileList', () => {
    it('should create image nodes for each file in the list', async () => {
      const mockNode1 = createMockNode({ id: 1 })
      const mockNode2 = createMockNode({ id: 2 })
      const mockBatchNode = createMockNode({ id: 3, type: 'BatchImagesNode' })

      vi.mocked(pasteImageNodes).mockResolvedValue([mockNode1, mockNode2])
      vi.mocked(createNode).mockResolvedValue(mockBatchNode)

      const file1 = createTestFile('test1.png', 'image/png')
      const file2 = createTestFile('test2.jpg', 'image/jpeg')
      const files = [file1, file2]

      await app.handleFileList(files)

      expect(pasteImageNodes).toHaveBeenCalledWith(mockCanvas, files)
      expect(createNode).toHaveBeenCalledWith(mockCanvas, 'BatchImagesNode')
      expect(mockCanvas.selectItems).toHaveBeenCalledWith([
        mockNode1,
        mockNode2,
        mockBatchNode
      ])
      expect(mockNode1.connect).toHaveBeenCalledWith(0, mockBatchNode, 0)
      expect(mockNode2.connect).toHaveBeenCalledWith(0, mockBatchNode, 1)
    })

    it('should select single image node without batch node', async () => {
      const mockNode1 = createMockNode({ id: 1 })
      vi.mocked(pasteImageNodes).mockResolvedValue([mockNode1])

      const file = createTestFile('test.png', 'image/png')

      await app.handleFileList([file])

      expect(createNode).not.toHaveBeenCalled()
      expect(mockCanvas.selectItems).toHaveBeenCalledWith([mockNode1])
      expect(mockNode1.connect).not.toHaveBeenCalled()
    })

    it('should handle empty file list', async () => {
      await app.handleFileList([])

      expect(pasteImageNodes).not.toHaveBeenCalled()
      expect(createNode).not.toHaveBeenCalled()
    })

    it('should not process unsupported file types', async () => {
      const invalidFile = createTestFile('test.pdf', 'application/pdf')

      await app.handleFileList([invalidFile])

      expect(pasteImageNodes).not.toHaveBeenCalled()
      expect(createNode).not.toHaveBeenCalled()
    })
  })

  describe('handleAudioFileList', () => {
    it('should create audio nodes and select them', async () => {
      const mockNode1 = createMockNode({ id: 1, type: 'LoadAudio' })
      const mockNode2 = createMockNode({ id: 2, type: 'LoadAudio' })
      vi.mocked(pasteAudioNodes).mockResolvedValue([mockNode1, mockNode2])

      const file1 = createTestFile('test1.mp3', 'audio/mpeg')
      const file2 = createTestFile('test2.wav', 'audio/wav')

      await app.handleAudioFileList([file1, file2])

      expect(pasteAudioNodes).toHaveBeenCalledWith(mockCanvas, [file1, file2])
      expect(mockCanvas.selectItems).toHaveBeenCalledWith([
        mockNode1,
        mockNode2
      ])
    })

    it('should not select when no nodes created', async () => {
      vi.mocked(pasteAudioNodes).mockResolvedValue([])

      await app.handleAudioFileList([createTestFile('test.mp3', 'audio/mpeg')])

      expect(mockCanvas.selectItems).not.toHaveBeenCalled()
    })
  })

  describe('handleVideoFileList', () => {
    it('should create video nodes and select them', async () => {
      const mockNode1 = createMockNode({ id: 1, type: 'LoadVideo' })
      const mockNode2 = createMockNode({ id: 2, type: 'LoadVideo' })
      vi.mocked(pasteVideoNodes).mockResolvedValue([mockNode1, mockNode2])

      const file1 = createTestFile('test1.mp4', 'video/mp4')
      const file2 = createTestFile('test2.webm', 'video/webm')

      await app.handleVideoFileList([file1, file2])

      expect(pasteVideoNodes).toHaveBeenCalledWith(mockCanvas, [file1, file2])
      expect(mockCanvas.selectItems).toHaveBeenCalledWith([
        mockNode1,
        mockNode2
      ])
    })

    it('should not select when no nodes created', async () => {
      vi.mocked(pasteVideoNodes).mockResolvedValue([])

      await app.handleVideoFileList([createTestFile('test.mp4', 'video/mp4')])

      expect(mockCanvas.selectItems).not.toHaveBeenCalled()
    })
  })

  describe('positionBatchNodes', () => {
    it('should position batch node to the right of first node', () => {
      const mockNode1 = createMockNode({
        pos: [100, 200],
        getBounding: vi.fn(() => new Float64Array([100, 200, 300, 400]))
      })
      const mockBatchNode = createMockNode({ pos: [0, 0] })

      app.positionBatchNodes([mockNode1], mockBatchNode)

      expect(mockBatchNode.pos).toEqual([500, 230])
    })

    it('should stack multiple image nodes vertically', () => {
      const mockNode1 = createMockNode({
        pos: [100, 200],
        type: 'LoadImage',
        getBounding: vi.fn(() => new Float64Array([100, 200, 300, 400]))
      })
      const mockNode2 = createMockNode({ pos: [0, 0], type: 'LoadImage' })
      const mockNode3 = createMockNode({ pos: [0, 0], type: 'LoadImage' })
      const mockBatchNode = createMockNode({ pos: [0, 0] })

      app.positionBatchNodes([mockNode1, mockNode2, mockNode3], mockBatchNode)

      expect(mockNode1.pos).toEqual([100, 200])
      expect(mockNode2.pos).toEqual([100, 594])
      expect(mockNode3.pos).toEqual([100, 963])
    })

    it('should call graph change once for all nodes', () => {
      const mockNode1 = createMockNode({
        getBounding: vi.fn(() => new Float64Array([100, 200, 300, 400]))
      })
      const mockBatchNode = createMockNode()

      app.positionBatchNodes([mockNode1], mockBatchNode)

      expect(mockCanvas.graph?.change).toHaveBeenCalledTimes(1)
    })
  })

  describe('handleFile', () => {
    it('should handle image files by creating LoadImage node', async () => {
      vi.mocked(getWorkflowDataFromFile).mockResolvedValue({})

      const mockNode = createMockNode()
      vi.mocked(createNode).mockResolvedValue(mockNode)

      const imageFile = createTestFile('test.png', 'image/png')

      await app.handleFile(imageFile)

      expect(createNode).toHaveBeenCalledWith(mockCanvas, 'LoadImage')
      expect(pasteImageNode).toHaveBeenCalledWith(
        mockCanvas,
        expect.any(DataTransferItemList),
        mockNode
      )
    })

    it('should handle audio files by creating LoadAudio node', async () => {
      vi.mocked(getWorkflowDataFromFile).mockResolvedValue({})

      const mockNode = createMockNode({ type: 'LoadAudio' })
      vi.mocked(createNode).mockResolvedValue(mockNode)

      const audioFile = createTestFile('test.mp3', 'audio/mpeg')

      await app.handleFile(audioFile)

      expect(createNode).toHaveBeenCalledWith(mockCanvas, 'LoadAudio')
      expect(pasteAudioNode).toHaveBeenCalledWith(
        mockCanvas,
        expect.any(DataTransferItemList),
        mockNode
      )
    })

    it('should handle video files by creating LoadVideo node', async () => {
      vi.mocked(getWorkflowDataFromFile).mockResolvedValue({})

      const mockNode = createMockNode({ type: 'LoadVideo' })
      vi.mocked(createNode).mockResolvedValue(mockNode)

      const videoFile = createTestFile('test.mp4', 'video/mp4')

      await app.handleFile(videoFile)

      expect(createNode).toHaveBeenCalledWith(mockCanvas, 'LoadVideo')
      expect(pasteVideoNode).toHaveBeenCalledWith(
        mockCanvas,
        expect.any(DataTransferItemList),
        mockNode
      )
    })

    it('should handle image files with non-workflow metadata by creating LoadImage node', async () => {
      vi.mocked(getWorkflowDataFromFile).mockResolvedValue({
        Software: 'gnome-screenshot'
      })

      const mockNode = createMockNode()
      vi.mocked(createNode).mockResolvedValue(mockNode)

      const imageFile = createTestFile('screenshot.png', 'image/png')

      await app.handleFile(imageFile)

      expect(createNode).toHaveBeenCalledWith(mockCanvas, 'LoadImage')
      expect(pasteImageNode).toHaveBeenCalledWith(
        mockCanvas,
        expect.any(DataTransferItemList),
        mockNode
      )
    })
  })
})
