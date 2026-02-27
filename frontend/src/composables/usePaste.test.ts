import { beforeEach, describe, expect, it, vi } from 'vitest'
import type {
  LGraphCanvas,
  LGraph,
  LGraphGroup,
  LGraphNode
} from '@/lib/litegraph/src/litegraph'
import { app } from '@/scripts/app'
import { createMockLGraphNode } from '@/utils/__tests__/litegraphTestUtils'
import {
  createNode,
  isAudioNode,
  isImageNode,
  isVideoNode
} from '@/utils/litegraphUtil'
import {
  cloneDataTransfer,
  pasteAudioNode,
  pasteAudioNodes,
  pasteImageNode,
  pasteImageNodes,
  pasteVideoNode,
  pasteVideoNodes,
  usePaste
} from './usePaste'

function createMockNode(): LGraphNode {
  return createMockLGraphNode({
    pos: [0, 0],
    pasteFile: vi.fn(),
    pasteFiles: vi.fn()
  })
}

function createImageFile(
  name: string = 'test.png',
  type: string = 'image/png'
): File {
  return new File([''], name, { type })
}

function createAudioFile(
  name: string = 'test.mp3',
  type: string = 'audio/mpeg'
): File {
  return new File([''], name, { type })
}

function createVideoFile(
  name: string = 'test.mp4',
  type: string = 'video/mp4'
): File {
  return new File([''], name, { type })
}

function createDataTransfer(files: File[] = []): DataTransfer {
  const dataTransfer = new DataTransfer()
  files.forEach((file) => dataTransfer.items.add(file))
  return dataTransfer
}

const mockCanvas = {
  current_node: null as LGraphNode | null,
  graph: {
    add: vi.fn(),
    change: vi.fn()
  } as Partial<LGraph> as LGraph,
  graph_mouse: [100, 200],
  pasteFromClipboard: vi.fn(),
  _deserializeItems: vi.fn()
} as Partial<LGraphCanvas> as LGraphCanvas

const mockCanvasStore = {
  canvas: mockCanvas,
  getCanvas: vi.fn(() => mockCanvas)
}

const mockWorkspaceStore = {
  shiftDown: false
}

vi.mock('@vueuse/core', () => ({
  useEventListener: vi.fn((target, event, handler) => {
    target.addEventListener(event, handler)
    return () => target.removeEventListener(event, handler)
  })
}))

vi.mock('@/renderer/core/canvas/canvasStore', () => ({
  useCanvasStore: () => mockCanvasStore
}))

vi.mock('@/stores/workspaceStore', () => ({
  useWorkspaceStore: () => mockWorkspaceStore
}))

vi.mock('@/scripts/app', () => ({
  app: {
    loadGraphData: vi.fn()
  }
}))

vi.mock('@/lib/litegraph/src/litegraph', async (importOriginal) => ({
  ...(await importOriginal()),
  LiteGraph: {
    createNode: vi.fn()
  }
}))

vi.mock('@/utils/litegraphUtil', () => ({
  createNode: vi.fn(),
  isAudioNode: vi.fn(),
  isImageNode: vi.fn(),
  isVideoNode: vi.fn()
}))

vi.mock('@/workbench/eventHelpers', () => ({
  shouldIgnoreCopyPaste: vi.fn()
}))

describe('pasteImageNode', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(mockCanvas.graph!.add).mockImplementation(
      (node: LGraphNode | LGraphGroup | null) => node as LGraphNode
    )
  })

  it('should create new LoadImage node when no image node provided', async () => {
    const mockNode = createMockNode()
    vi.mocked(createNode).mockResolvedValue(mockNode)

    const file = createImageFile()
    const dataTransfer = createDataTransfer([file])

    await pasteImageNode(mockCanvas, dataTransfer.items)

    expect(createNode).toHaveBeenCalledWith(mockCanvas, 'LoadImage')
    expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
  })

  it('should use existing image node when provided', async () => {
    const mockNode = createMockNode()
    const file = createImageFile()
    const dataTransfer = createDataTransfer([file])

    await pasteImageNode(mockCanvas, dataTransfer.items, mockNode)

    expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
    expect(mockNode.pasteFiles).toHaveBeenCalledWith([file])
  })

  it('should handle multiple image files', async () => {
    const mockNode = createMockNode()
    const file1 = createImageFile('test1.png')
    const file2 = createImageFile('test2.jpg', 'image/jpeg')
    const dataTransfer = createDataTransfer([file1, file2])

    await pasteImageNode(mockCanvas, dataTransfer.items, mockNode)

    expect(mockNode.pasteFile).toHaveBeenCalledWith(file1)
    expect(mockNode.pasteFiles).toHaveBeenCalledWith([file1, file2])
  })

  it('should do nothing when no image files present', async () => {
    const mockNode = createMockNode()
    const dataTransfer = createDataTransfer()

    await pasteImageNode(mockCanvas, dataTransfer.items, mockNode)

    expect(mockNode.pasteFile).not.toHaveBeenCalled()
    expect(mockNode.pasteFiles).not.toHaveBeenCalled()
  })

  it('should filter non-image items', async () => {
    const mockNode = createMockNode()
    const imageFile = createImageFile()
    const textFile = new File([''], 'test.txt', { type: 'text/plain' })
    const dataTransfer = createDataTransfer([textFile, imageFile])

    await pasteImageNode(mockCanvas, dataTransfer.items, mockNode)

    expect(mockNode.pasteFile).toHaveBeenCalledWith(imageFile)
    expect(mockNode.pasteFiles).toHaveBeenCalledWith([imageFile])
  })
})

describe('pasteImageNodes', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should create multiple nodes for multiple files', async () => {
    const mockNode1 = createMockNode()
    const mockNode2 = createMockNode()
    vi.mocked(createNode)
      .mockResolvedValueOnce(mockNode1)
      .mockResolvedValueOnce(mockNode2)

    const file1 = createImageFile('test1.png')
    const file2 = createImageFile('test2.jpg', 'image/jpeg')

    const result = await pasteImageNodes(mockCanvas, [file1, file2])

    expect(createNode).toHaveBeenCalledTimes(2)
    expect(createNode).toHaveBeenNthCalledWith(1, mockCanvas, 'LoadImage')
    expect(createNode).toHaveBeenNthCalledWith(2, mockCanvas, 'LoadImage')
    expect(mockNode1.pasteFile).toHaveBeenCalledWith(file1)
    expect(mockNode2.pasteFile).toHaveBeenCalledWith(file2)
    expect(result).toEqual([mockNode1, mockNode2])
  })

  it('should handle empty file list', async () => {
    const result = await pasteImageNodes(mockCanvas, [])

    expect(createNode).not.toHaveBeenCalled()
    expect(result).toEqual([])
  })
})

describe('pasteAudioNode', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should create new LoadAudio node when no audio node provided', async () => {
    const mockNode = createMockNode()
    vi.mocked(createNode).mockResolvedValue(mockNode)

    const file = createAudioFile()
    const dataTransfer = createDataTransfer([file])

    await pasteAudioNode(mockCanvas, dataTransfer.items)

    expect(createNode).toHaveBeenCalledWith(mockCanvas, 'LoadAudio')
    expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
  })

  it('should use existing audio node when provided', async () => {
    const mockNode = createMockNode()
    const file = createAudioFile()
    const dataTransfer = createDataTransfer([file])

    await pasteAudioNode(mockCanvas, dataTransfer.items, mockNode)

    expect(createNode).not.toHaveBeenCalled()
    expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
  })

  it('should filter non-audio items', async () => {
    const mockNode = createMockNode()
    const audioFile = createAudioFile()
    const textFile = new File([''], 'test.txt', { type: 'text/plain' })
    const dataTransfer = createDataTransfer([textFile, audioFile])

    await pasteAudioNode(mockCanvas, dataTransfer.items, mockNode)

    expect(mockNode.pasteFile).toHaveBeenCalledWith(audioFile)
    expect(mockNode.pasteFiles).toHaveBeenCalledWith([audioFile])
  })

  it('should do nothing when no audio files present', async () => {
    const mockNode = createMockNode()
    const dataTransfer = createDataTransfer()

    await pasteAudioNode(mockCanvas, dataTransfer.items, mockNode)

    expect(mockNode.pasteFile).not.toHaveBeenCalled()
    expect(mockNode.pasteFiles).not.toHaveBeenCalled()
  })
})

describe('pasteAudioNodes', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should create multiple nodes for multiple audio files', async () => {
    const mockNode1 = createMockNode()
    const mockNode2 = createMockNode()
    vi.mocked(createNode)
      .mockResolvedValueOnce(mockNode1)
      .mockResolvedValueOnce(mockNode2)

    const file1 = createAudioFile('file1.mp3')
    const file2 = createAudioFile('file2.wav', 'audio/wav')

    const result = await pasteAudioNodes(mockCanvas, [file1, file2])

    expect(createNode).toHaveBeenCalledTimes(2)
    expect(createNode).toHaveBeenNthCalledWith(1, mockCanvas, 'LoadAudio')
    expect(createNode).toHaveBeenNthCalledWith(2, mockCanvas, 'LoadAudio')
    expect(mockNode1.pasteFile).toHaveBeenCalledWith(file1)
    expect(mockNode2.pasteFile).toHaveBeenCalledWith(file2)
    expect(result).toEqual([mockNode1, mockNode2])
  })

  it('should handle empty file list', async () => {
    const result = await pasteAudioNodes(mockCanvas, [])

    expect(createNode).not.toHaveBeenCalled()
    expect(result).toEqual([])
  })

  it('should handle single audio file', async () => {
    const mockNode = createMockNode()
    vi.mocked(createNode).mockResolvedValue(mockNode)

    const file = createAudioFile()
    const result = await pasteAudioNodes(mockCanvas, [file])

    expect(createNode).toHaveBeenCalledTimes(1)
    expect(result).toEqual([mockNode])
  })
})

describe('pasteVideoNode', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should create new LoadVideo node when no video node provided', async () => {
    const mockNode = createMockNode()
    vi.mocked(createNode).mockResolvedValue(mockNode)

    const file = createVideoFile()
    const dataTransfer = createDataTransfer([file])

    await pasteVideoNode(mockCanvas, dataTransfer.items)

    expect(createNode).toHaveBeenCalledWith(mockCanvas, 'LoadVideo')
    expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
  })

  it('should use existing video node when provided', async () => {
    const mockNode = createMockNode()
    const file = createVideoFile()
    const dataTransfer = createDataTransfer([file])

    await pasteVideoNode(mockCanvas, dataTransfer.items, mockNode)

    expect(createNode).not.toHaveBeenCalled()
    expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
  })

  it('should filter non-video items', async () => {
    const mockNode = createMockNode()
    const videoFile = createVideoFile()
    const imageFile = createImageFile()
    const dataTransfer = createDataTransfer([imageFile, videoFile])

    await pasteVideoNode(mockCanvas, dataTransfer.items, mockNode)

    expect(mockNode.pasteFile).toHaveBeenCalledWith(videoFile)
    expect(mockNode.pasteFiles).toHaveBeenCalledWith([videoFile])
  })

  it('should do nothing when no video files present', async () => {
    const mockNode = createMockNode()
    const dataTransfer = createDataTransfer()

    await pasteVideoNode(mockCanvas, dataTransfer.items, mockNode)

    expect(mockNode.pasteFile).not.toHaveBeenCalled()
    expect(mockNode.pasteFiles).not.toHaveBeenCalled()
  })
})

describe('pasteVideoNodes', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should create multiple nodes for multiple video files', async () => {
    const mockNode1 = createMockNode()
    const mockNode2 = createMockNode()
    vi.mocked(createNode)
      .mockResolvedValueOnce(mockNode1)
      .mockResolvedValueOnce(mockNode2)

    const file1 = createVideoFile('file1.mp4')
    const file2 = createVideoFile('file2.webm', 'video/webm')

    const result = await pasteVideoNodes(mockCanvas, [file1, file2])

    expect(createNode).toHaveBeenCalledTimes(2)
    expect(createNode).toHaveBeenNthCalledWith(1, mockCanvas, 'LoadVideo')
    expect(createNode).toHaveBeenNthCalledWith(2, mockCanvas, 'LoadVideo')
    expect(mockNode1.pasteFile).toHaveBeenCalledWith(file1)
    expect(mockNode2.pasteFile).toHaveBeenCalledWith(file2)
    expect(result).toEqual([mockNode1, mockNode2])
  })

  it('should handle empty file list', async () => {
    const result = await pasteVideoNodes(mockCanvas, [])

    expect(createNode).not.toHaveBeenCalled()
    expect(result).toEqual([])
  })

  it('should handle single video file', async () => {
    const mockNode = createMockNode()
    vi.mocked(createNode).mockResolvedValue(mockNode)

    const file = createVideoFile()
    const result = await pasteVideoNodes(mockCanvas, [file])

    expect(createNode).toHaveBeenCalledTimes(1)
    expect(result).toEqual([mockNode])
  })
})

describe('usePaste', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockCanvas.current_node = null
    mockWorkspaceStore.shiftDown = false
    vi.mocked(mockCanvas.graph!.add).mockImplementation(
      (node: LGraphNode | LGraphGroup | null) => node as LGraphNode
    )
  })

  it('should handle image paste', async () => {
    const mockNode = createMockNode()
    vi.mocked(createNode).mockResolvedValue(mockNode)

    usePaste()

    const file = createImageFile()
    const dataTransfer = createDataTransfer([file])
    const event = new ClipboardEvent('paste', { clipboardData: dataTransfer })
    document.dispatchEvent(event)

    await vi.waitFor(() => {
      expect(createNode).toHaveBeenCalledWith(mockCanvas, 'LoadImage')
      expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
    })
  })

  it('should handle audio paste using createNode helper', async () => {
    const mockNode = createMockNode()
    vi.mocked(createNode).mockResolvedValue(mockNode)

    usePaste()

    const file = createAudioFile()
    const dataTransfer = createDataTransfer([file])
    const event = new ClipboardEvent('paste', { clipboardData: dataTransfer })
    document.dispatchEvent(event)

    await vi.waitFor(() => {
      expect(createNode).toHaveBeenCalledWith(mockCanvas, 'LoadAudio')
      expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
    })
  })

  it('should paste audio onto selected LoadAudio node', async () => {
    const mockNode = createMockLGraphNode({
      is_selected: true,
      pasteFile: vi.fn(),
      pasteFiles: vi.fn()
    })
    mockCanvas.current_node = mockNode
    vi.mocked(isAudioNode).mockReturnValue(true)

    usePaste()

    const file = createAudioFile()
    const dataTransfer = createDataTransfer([file])
    const event = new ClipboardEvent('paste', { clipboardData: dataTransfer })
    document.dispatchEvent(event)

    await vi.waitFor(() => {
      expect(createNode).not.toHaveBeenCalled()
      expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
    })
  })

  it('should handle video paste', async () => {
    const mockNode = createMockNode()
    vi.mocked(createNode).mockResolvedValue(mockNode)

    usePaste()

    const file = createVideoFile()
    const dataTransfer = createDataTransfer([file])
    const event = new ClipboardEvent('paste', { clipboardData: dataTransfer })
    document.dispatchEvent(event)

    await vi.waitFor(() => {
      expect(createNode).toHaveBeenCalledWith(mockCanvas, 'LoadVideo')
      expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
    })
  })

  it('should paste video onto selected LoadVideo node', async () => {
    const mockNode = createMockLGraphNode({
      is_selected: true,
      pasteFile: vi.fn(),
      pasteFiles: vi.fn()
    })
    mockCanvas.current_node = mockNode
    vi.mocked(isVideoNode).mockReturnValue(true)

    usePaste()

    const file = createVideoFile()
    const dataTransfer = createDataTransfer([file])
    const event = new ClipboardEvent('paste', { clipboardData: dataTransfer })
    document.dispatchEvent(event)

    await vi.waitFor(() => {
      expect(createNode).not.toHaveBeenCalled()
      expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
    })
  })

  it('should handle workflow JSON paste', async () => {
    const workflow = { version: '1.0', nodes: [], extra: {} }

    usePaste()

    const dataTransfer = new DataTransfer()
    dataTransfer.setData('text/plain', JSON.stringify(workflow))

    const event = new ClipboardEvent('paste', { clipboardData: dataTransfer })
    document.dispatchEvent(event)

    await vi.waitFor(() => {
      expect(app.loadGraphData).toHaveBeenCalledWith(workflow)
    })
  })

  it('should ignore paste when shift is down', () => {
    mockWorkspaceStore.shiftDown = true

    usePaste()

    const file = createImageFile()
    const dataTransfer = createDataTransfer([file])
    const event = new ClipboardEvent('paste', { clipboardData: dataTransfer })
    document.dispatchEvent(event)

    expect(createNode).not.toHaveBeenCalled()
  })

  it('should use existing image node when selected', () => {
    const mockNode = createMockLGraphNode({
      is_selected: true,
      pasteFile: vi.fn(),
      pasteFiles: vi.fn()
    })
    mockCanvas.current_node = mockNode
    vi.mocked(isImageNode).mockReturnValue(true)

    usePaste()

    const file = createImageFile()
    const dataTransfer = createDataTransfer([file])
    const event = new ClipboardEvent('paste', { clipboardData: dataTransfer })
    document.dispatchEvent(event)

    expect(mockNode.pasteFile).toHaveBeenCalledWith(file)
  })

  it('should call canvas pasteFromClipboard for non-workflow text', () => {
    usePaste()

    const dataTransfer = new DataTransfer()
    dataTransfer.setData('text/plain', 'just some text')

    const event = new ClipboardEvent('paste', { clipboardData: dataTransfer })
    document.dispatchEvent(event)

    expect(mockCanvas.pasteFromClipboard).toHaveBeenCalled()
  })

  it('should handle clipboard items with metadata', async () => {
    const data = { test: 'data' }
    const encoded = btoa(JSON.stringify(data))
    const html = `<div data-metadata="${encoded}"></div>`

    usePaste()

    const dataTransfer = new DataTransfer()
    dataTransfer.setData('text/html', html)

    const event = new ClipboardEvent('paste', { clipboardData: dataTransfer })
    document.dispatchEvent(event)

    await vi.waitFor(() => {
      expect(mockCanvas._deserializeItems).toHaveBeenCalledWith(
        data,
        expect.any(Object)
      )
    })
  })
})

describe('cloneDataTransfer', () => {
  it('should clone string data', () => {
    const original = new DataTransfer()
    original.setData('text/plain', 'test text')
    original.setData('text/html', '<p>test html</p>')

    const cloned = cloneDataTransfer(original)

    expect(cloned.getData('text/plain')).toBe('test text')
    expect(cloned.getData('text/html')).toBe('<p>test html</p>')
  })

  it('should clone files', () => {
    const file1 = createImageFile('test1.png')
    const file2 = createImageFile('test2.jpg', 'image/jpeg')
    const original = createDataTransfer([file1, file2])

    const cloned = cloneDataTransfer(original)

    // Files are added from both .files and .items, causing duplicates
    expect(cloned.files.length).toBeGreaterThanOrEqual(2)
    expect(Array.from(cloned.files)).toContain(file1)
    expect(Array.from(cloned.files)).toContain(file2)
  })

  it('should preserve dropEffect and effectAllowed', () => {
    const original = new DataTransfer()
    original.dropEffect = 'copy'
    original.effectAllowed = 'copyMove'

    const cloned = cloneDataTransfer(original)

    expect(cloned.dropEffect).toBe('copy')
    expect(cloned.effectAllowed).toBe('copyMove')
  })

  it('should handle empty DataTransfer', () => {
    const original = new DataTransfer()

    const cloned = cloneDataTransfer(original)

    expect(cloned.types.length).toBe(0)
    expect(cloned.files.length).toBe(0)
  })

  it('should clone both string data and files', () => {
    const file = createImageFile()
    const original = createDataTransfer([file])
    original.setData('text/plain', 'test')

    const cloned = cloneDataTransfer(original)

    expect(cloned.getData('text/plain')).toBe('test')
    // Files are added from both .files and .items
    expect(cloned.files.length).toBeGreaterThanOrEqual(1)
    expect(Array.from(cloned.files)).toContain(file)
  })
})
