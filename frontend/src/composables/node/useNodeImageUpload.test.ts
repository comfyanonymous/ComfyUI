import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { LGraphNode } from '@/lib/litegraph/src/litegraph'

const { mockFetchApi, mockAddAlert, mockUpdateInputs } = vi.hoisted(() => ({
  mockFetchApi: vi.fn(),
  mockAddAlert: vi.fn(),
  mockUpdateInputs: vi.fn()
}))

let capturedDragOnDrop: (files: File[]) => Promise<string[]>

vi.mock('@/composables/node/useNodeDragAndDrop', () => ({
  useNodeDragAndDrop: (
    _node: LGraphNode,
    opts: { onDrop: typeof capturedDragOnDrop }
  ) => {
    capturedDragOnDrop = opts.onDrop
  }
}))

vi.mock('@/composables/node/useNodeFileInput', () => ({
  useNodeFileInput: () => ({ openFileSelection: vi.fn() })
}))

vi.mock('@/composables/node/useNodePaste', () => ({
  useNodePaste: vi.fn()
}))

vi.mock('@/i18n', () => ({
  t: (key: string) => key
}))

vi.mock('@/platform/updates/common/toastStore', () => ({
  useToastStore: () => ({ addAlert: mockAddAlert })
}))

vi.mock('@/scripts/api', () => ({
  api: { fetchApi: mockFetchApi }
}))

vi.mock('@/stores/assetsStore', () => ({
  useAssetsStore: () => ({ updateInputs: mockUpdateInputs })
}))

function createMockNode(): LGraphNode {
  return {
    isUploading: false,
    imgs: [new Image()],
    graph: { setDirtyCanvas: vi.fn() },
    size: [300, 400]
  } as unknown as LGraphNode
}

function createFile(name = 'test.png'): File {
  return new File(['data'], name, { type: 'image/png' })
}

function successResponse(name: string, subfolder?: string) {
  return {
    status: 200,
    json: () => Promise.resolve({ name, subfolder })
  }
}

function failResponse(status = 500) {
  return {
    status,
    statusText: 'Server Error'
  }
}

describe('useNodeImageUpload', () => {
  let node: LGraphNode
  let onUploadComplete: (paths: string[]) => void
  let onUploadStart: (files: File[]) => void
  let onUploadError: () => void

  beforeEach(async () => {
    vi.resetModules()
    vi.clearAllMocks()
    node = createMockNode()
    onUploadComplete = vi.fn()
    onUploadStart = vi.fn()
    onUploadError = vi.fn()

    const { useNodeImageUpload } = await import('./useNodeImageUpload')
    useNodeImageUpload(node, {
      onUploadComplete,
      onUploadStart,
      onUploadError,
      folder: 'input'
    })
  })

  it('sets isUploading true during upload and false after', async () => {
    mockFetchApi.mockResolvedValueOnce(successResponse('test.png'))

    const promise = capturedDragOnDrop([createFile()])
    expect(node.isUploading).toBe(true)

    await promise
    expect(node.isUploading).toBe(false)
  })

  it('clears node.imgs on upload start', async () => {
    mockFetchApi.mockResolvedValueOnce(successResponse('test.png'))

    const promise = capturedDragOnDrop([createFile()])
    expect(node.imgs).toBeUndefined()

    await promise
  })

  it('calls onUploadStart with files', async () => {
    mockFetchApi.mockResolvedValueOnce(successResponse('test.png'))
    const files = [createFile()]

    await capturedDragOnDrop(files)
    expect(onUploadStart).toHaveBeenCalledWith(files)
  })

  it('calls onUploadComplete with valid paths on success', async () => {
    mockFetchApi.mockResolvedValueOnce(successResponse('test.png'))

    await capturedDragOnDrop([createFile()])
    expect(onUploadComplete).toHaveBeenCalledWith(['test.png'])
  })

  it('includes subfolder in returned path', async () => {
    mockFetchApi.mockResolvedValueOnce(successResponse('test.png', 'pasted'))

    await capturedDragOnDrop([createFile()])
    expect(onUploadComplete).toHaveBeenCalledWith(['pasted/test.png'])
  })

  it('calls onUploadError when all uploads fail', async () => {
    mockFetchApi.mockResolvedValueOnce(failResponse())

    await capturedDragOnDrop([createFile()])
    expect(onUploadError).toHaveBeenCalled()
    expect(onUploadComplete).not.toHaveBeenCalled()
  })

  it('resets isUploading even when upload fails', async () => {
    mockFetchApi.mockRejectedValueOnce(new Error('Network error'))

    await capturedDragOnDrop([createFile()])
    expect(node.isUploading).toBe(false)
  })

  it('rejects concurrent uploads with a toast', async () => {
    mockFetchApi.mockImplementation(
      () =>
        new Promise((resolve) =>
          setTimeout(() => resolve(successResponse('a.png')), 50)
        )
    )

    const first = capturedDragOnDrop([createFile('a.png')])
    const second = await capturedDragOnDrop([createFile('b.png')])

    expect(second).toEqual([])
    expect(mockAddAlert).toHaveBeenCalledWith('g.uploadAlreadyInProgress')

    await first
  })

  it('calls setDirtyCanvas on start and finish', async () => {
    mockFetchApi.mockResolvedValueOnce(successResponse('test.png'))

    await capturedDragOnDrop([createFile()])
    expect(node.graph?.setDirtyCanvas).toHaveBeenCalledTimes(2)
  })
})
