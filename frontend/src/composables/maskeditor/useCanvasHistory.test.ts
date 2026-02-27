import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { useCanvasHistory } from '@/composables/maskeditor/useCanvasHistory'

// Define the store shape to avoid 'any' and cast to the expected type
interface MaskEditorStoreState {
  maskCanvas: HTMLCanvasElement | null
  rgbCanvas: HTMLCanvasElement | null
  imgCanvas: HTMLCanvasElement | null
  maskCtx: CanvasRenderingContext2D | null
  rgbCtx: CanvasRenderingContext2D | null
  imgCtx: CanvasRenderingContext2D | null
}

// Use vi.hoisted to create isolated mock state container
const mockRefs = vi.hoisted(() => ({
  maskCanvas: null as HTMLCanvasElement | null,
  rgbCanvas: null as HTMLCanvasElement | null,
  imgCanvas: null as HTMLCanvasElement | null,
  maskCtx: null as CanvasRenderingContext2D | null,
  rgbCtx: null as CanvasRenderingContext2D | null,
  imgCtx: null as CanvasRenderingContext2D | null
}))

const mockStore: MaskEditorStoreState = {
  get maskCanvas() {
    return mockRefs.maskCanvas
  },
  set maskCanvas(val) {
    mockRefs.maskCanvas = val
  },
  get rgbCanvas() {
    return mockRefs.rgbCanvas
  },
  set rgbCanvas(val) {
    mockRefs.rgbCanvas = val
  },
  get imgCanvas() {
    return mockRefs.imgCanvas
  },
  set imgCanvas(val) {
    mockRefs.imgCanvas = val
  },
  get maskCtx() {
    return mockRefs.maskCtx
  },
  set maskCtx(val) {
    mockRefs.maskCtx = val
  },
  get rgbCtx() {
    return mockRefs.rgbCtx
  },
  set rgbCtx(val) {
    mockRefs.rgbCtx = val
  },
  get imgCtx() {
    return mockRefs.imgCtx
  },
  set imgCtx(val) {
    mockRefs.imgCtx = val
  }
}

vi.mock('@/stores/maskEditorStore', () => ({
  useMaskEditorStore: vi.fn(() => mockStore)
}))

// Mock ImageBitmap using safe global augmentation pattern
if (typeof globalThis.ImageBitmap === 'undefined') {
  globalThis.ImageBitmap = class ImageBitmap {
    width: number
    height: number
    constructor(width = 100, height = 100) {
      this.width = width
      this.height = height
    }
    close() {}
  } as typeof ImageBitmap
}

describe('useCanvasHistory', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    let rafCallCount = 0
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(
      (cb: FrameRequestCallback) => {
        if (rafCallCount++ < 100) {
          setTimeout(() => cb(0), 0)
        }
        return rafCallCount
      }
    )

    const createMockImageData = (): ImageData => {
      return {
        data: new Uint8ClampedArray(100 * 100 * 4),
        width: 100,
        height: 100
      } as ImageData
    }

    // Mock contexts using explicit partial-cast pattern
    mockRefs.maskCtx = {
      getImageData: vi.fn(() => createMockImageData()),
      putImageData: vi.fn(),
      clearRect: vi.fn(),
      drawImage: vi.fn()
    } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D

    mockRefs.rgbCtx = {
      getImageData: vi.fn(() => createMockImageData()),
      putImageData: vi.fn(),
      clearRect: vi.fn(),
      drawImage: vi.fn()
    } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D

    mockRefs.imgCtx = {
      getImageData: vi.fn(() => createMockImageData()),
      putImageData: vi.fn(),
      clearRect: vi.fn(),
      drawImage: vi.fn()
    } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D

    // Mock canvases using explicit partial-cast pattern
    mockRefs.maskCanvas = {
      width: 100,
      height: 100
    } as Partial<HTMLCanvasElement> as HTMLCanvasElement

    mockRefs.rgbCanvas = {
      width: 100,
      height: 100
    } as Partial<HTMLCanvasElement> as HTMLCanvasElement

    mockRefs.imgCanvas = {
      width: 100,
      height: 100
    } as Partial<HTMLCanvasElement> as HTMLCanvasElement
  })

  describe('initialization', () => {
    it('should initialize with default values', () => {
      const history = useCanvasHistory()

      expect(history.canUndo.value).toBe(false)
      expect(history.canRedo.value).toBe(false)
    })

    it('should save initial state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()

      expect(mockRefs.maskCtx!.getImageData).toHaveBeenCalledWith(
        0,
        0,
        100,
        100
      )
      expect(mockRefs.rgbCtx!.getImageData).toHaveBeenCalledWith(0, 0, 100, 100)
      expect(mockRefs.imgCtx!.getImageData).toHaveBeenCalledWith(0, 0, 100, 100)
      expect(history.canUndo.value).toBe(false)
      expect(history.canRedo.value).toBe(false)
    })

    it('should wait for canvas to be ready', () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame')

      mockRefs.maskCanvas = {
        // oxlint-disable-next-line no-misused-spread
        ...mockRefs.maskCanvas,
        width: 0,
        height: 0
      } as Partial<HTMLCanvasElement> as HTMLCanvasElement

      const history = useCanvasHistory()
      history.saveInitialState()

      expect(rafSpy).toHaveBeenCalled()

      mockRefs.maskCanvas = {
        width: 100,
        height: 100
      } as Partial<HTMLCanvasElement> as HTMLCanvasElement
    })

    it('should wait for context to be ready', () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame')

      mockRefs.maskCtx = null

      const history = useCanvasHistory()
      history.saveInitialState()

      expect(rafSpy).toHaveBeenCalled()

      const createMockImageData = (): ImageData => {
        return {
          data: new Uint8ClampedArray(100 * 100 * 4),
          width: 100,
          height: 100
        } as ImageData
      }

      mockRefs.maskCtx = {
        getImageData: vi.fn(() => createMockImageData()),
        putImageData: vi.fn(),
        clearRect: vi.fn(),
        drawImage: vi.fn()
      } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D
    })
  })

  describe('saveState', () => {
    it('should save a new state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      vi.mocked(mockRefs.maskCtx!.getImageData).mockClear()
      vi.mocked(mockRefs.rgbCtx!.getImageData).mockClear()
      vi.mocked(mockRefs.imgCtx!.getImageData).mockClear()

      history.saveState()

      expect(mockRefs.maskCtx!.getImageData).toHaveBeenCalledWith(
        0,
        0,
        100,
        100
      )
      expect(mockRefs.rgbCtx!.getImageData).toHaveBeenCalledWith(0, 0, 100, 100)
      expect(mockRefs.imgCtx!.getImageData).toHaveBeenCalledWith(0, 0, 100, 100)
      expect(history.canUndo.value).toBe(true)
    })

    it('should clear redo states when saving new state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()
      history.saveState()
      history.undo()

      expect(history.canRedo.value).toBe(true)

      history.saveState()

      expect(history.canRedo.value).toBe(false)
    })

    it('should respect maxStates limit', () => {
      const history = useCanvasHistory(3)

      history.saveInitialState()
      history.saveState()
      history.saveState()
      history.saveState()
      history.saveState()

      expect(history.canUndo.value).toBe(true)

      let undoCount = 0
      while (history.canUndo.value && undoCount < 10) {
        history.undo()
        undoCount++
      }

      expect(undoCount).toBe(2)
    })

    it('should call saveInitialState if not initialized', () => {
      const history = useCanvasHistory()

      history.saveState()

      expect(mockRefs.maskCtx!.getImageData).toHaveBeenCalled()
      expect(mockRefs.rgbCtx!.getImageData).toHaveBeenCalled()
      expect(mockRefs.imgCtx!.getImageData).toHaveBeenCalled()
    })

    it('should not save state if context is missing', () => {
      const history = useCanvasHistory()

      history.saveInitialState()

      const savedMaskCtx = mockRefs.maskCtx
      mockRefs.maskCtx = null
      vi.mocked(savedMaskCtx!.getImageData).mockClear()
      vi.mocked(mockRefs.rgbCtx!.getImageData).mockClear()
      vi.mocked(mockRefs.imgCtx!.getImageData).mockClear()

      history.saveState()

      expect(savedMaskCtx!.getImageData).not.toHaveBeenCalled()

      mockRefs.maskCtx = savedMaskCtx
    })
  })

  describe('undo', () => {
    it('should undo to previous state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()

      history.undo()

      expect(mockRefs.maskCtx!.putImageData).toHaveBeenCalled()
      expect(mockRefs.rgbCtx!.putImageData).toHaveBeenCalled()
      expect(mockRefs.imgCtx!.putImageData).toHaveBeenCalled()
      expect(history.canUndo.value).toBe(false)
      expect(history.canRedo.value).toBe(true)
    })

    it('should not undo when no undo states available', () => {
      const history = useCanvasHistory()

      history.saveInitialState()

      vi.mocked(mockRefs.maskCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.rgbCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.imgCtx!.putImageData).mockClear()

      history.undo()

      expect(mockRefs.maskCtx!.putImageData).not.toHaveBeenCalled()
      expect(mockRefs.rgbCtx!.putImageData).not.toHaveBeenCalled()
      expect(mockRefs.imgCtx!.putImageData).not.toHaveBeenCalled()
    })

    it('should undo multiple times', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()
      history.saveState()
      history.saveState()

      history.undo()
      expect(history.canUndo.value).toBe(true)

      history.undo()
      expect(history.canUndo.value).toBe(true)

      history.undo()
      expect(history.canUndo.value).toBe(false)
    })

    it('should not undo beyond first state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()

      history.undo()

      vi.mocked(mockRefs.maskCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.rgbCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.imgCtx!.putImageData).mockClear()

      history.undo()

      expect(mockRefs.maskCtx!.putImageData).not.toHaveBeenCalled()
      expect(mockRefs.rgbCtx!.putImageData).not.toHaveBeenCalled()
      expect(mockRefs.imgCtx!.putImageData).not.toHaveBeenCalled()
    })
  })

  describe('redo', () => {
    it('should redo to next state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()
      history.undo()

      vi.mocked(mockRefs.maskCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.rgbCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.imgCtx!.putImageData).mockClear()

      history.redo()

      expect(mockRefs.maskCtx!.putImageData).toHaveBeenCalled()
      expect(mockRefs.rgbCtx!.putImageData).toHaveBeenCalled()
      expect(mockRefs.imgCtx!.putImageData).toHaveBeenCalled()
      expect(history.canRedo.value).toBe(false)
      expect(history.canUndo.value).toBe(true)
    })

    it('should not redo when no redo states available', () => {
      const history = useCanvasHistory()

      history.saveInitialState()

      vi.mocked(mockRefs.maskCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.rgbCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.imgCtx!.putImageData).mockClear()

      history.redo()

      expect(mockRefs.maskCtx!.putImageData).not.toHaveBeenCalled()
      expect(mockRefs.rgbCtx!.putImageData).not.toHaveBeenCalled()
      expect(mockRefs.imgCtx!.putImageData).not.toHaveBeenCalled()
    })

    it('should redo multiple times', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()
      history.saveState()
      history.saveState()

      history.undo()
      history.undo()
      history.undo()

      history.redo()
      expect(history.canRedo.value).toBe(true)

      history.redo()
      expect(history.canRedo.value).toBe(true)

      history.redo()
      expect(history.canRedo.value).toBe(false)
    })

    it('should not redo beyond last state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()
      history.undo()

      history.redo()

      vi.mocked(mockRefs.maskCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.rgbCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.imgCtx!.putImageData).mockClear()

      history.redo()

      expect(mockRefs.maskCtx!.putImageData).not.toHaveBeenCalled()
      expect(mockRefs.rgbCtx!.putImageData).not.toHaveBeenCalled()
      expect(mockRefs.imgCtx!.putImageData).not.toHaveBeenCalled()
    })
  })

  describe('clearStates', () => {
    it('should clear all states', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()
      history.saveState()

      history.clearStates()

      expect(history.canUndo.value).toBe(false)
      expect(history.canRedo.value).toBe(false)
    })

    it('should allow saving initial state after clear', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.clearStates()

      vi.mocked(mockRefs.maskCtx!.getImageData).mockClear()
      vi.mocked(mockRefs.rgbCtx!.getImageData).mockClear()
      vi.mocked(mockRefs.imgCtx!.getImageData).mockClear()

      history.saveInitialState()

      expect(mockRefs.maskCtx!.getImageData).toHaveBeenCalled()
      expect(mockRefs.rgbCtx!.getImageData).toHaveBeenCalled()
      expect(mockRefs.imgCtx!.getImageData).toHaveBeenCalled()
    })
  })

  describe('canUndo computed', () => {
    it('should be false with no states', () => {
      const history = useCanvasHistory()

      expect(history.canUndo.value).toBe(false)
    })

    it('should be false with only initial state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()

      expect(history.canUndo.value).toBe(false)
    })

    it('should be true after saving a state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()

      expect(history.canUndo.value).toBe(true)
    })

    it('should be false after undoing to first state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()
      history.undo()

      expect(history.canUndo.value).toBe(false)
    })
  })

  describe('canRedo computed', () => {
    it('should be false with no undo', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()

      expect(history.canRedo.value).toBe(false)
    })

    it('should be true after undo', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()
      history.undo()

      expect(history.canRedo.value).toBe(true)
    })

    it('should be false after redo to last state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()
      history.undo()
      history.redo()

      expect(history.canRedo.value).toBe(false)
    })

    it('should be false after saving new state', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()
      history.undo()

      expect(history.canRedo.value).toBe(true)

      history.saveState()

      expect(history.canRedo.value).toBe(false)
    })
  })

  describe('restoreState', () => {
    it('should not restore if context is missing', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()

      const savedMaskCtx = mockRefs.maskCtx
      mockRefs.maskCtx = null
      vi.mocked(savedMaskCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.rgbCtx!.putImageData).mockClear()
      vi.mocked(mockRefs.imgCtx!.putImageData).mockClear()

      history.undo()

      expect(savedMaskCtx!.putImageData).not.toHaveBeenCalled()

      mockRefs.maskCtx = savedMaskCtx
    })
  })

  describe('edge cases', () => {
    it('should handle rapid state saves', async () => {
      const history = useCanvasHistory()

      history.saveInitialState()

      for (let i = 0; i < 10; i++) {
        history.saveState()
        await nextTick()
      }

      expect(history.canUndo.value).toBe(true)
    })

    it('should handle maxStates of 1', () => {
      const history = useCanvasHistory(1)

      history.saveInitialState()
      history.saveState()

      expect(history.canUndo.value).toBe(false)
    })

    it('should handle undo/redo cycling', () => {
      const history = useCanvasHistory()

      history.saveInitialState()
      history.saveState()
      history.saveState()

      history.undo()
      history.redo()
      history.undo()
      history.redo()
      history.undo()

      expect(history.canRedo.value).toBe(true)
      expect(history.canUndo.value).toBe(true)
    })

    it('should handle zero-sized canvas', () => {
      if (mockRefs.maskCanvas) {
        mockRefs.maskCanvas = {
          width: 0,
          height: 0
        } as Partial<HTMLCanvasElement> as HTMLCanvasElement
      }

      const history = useCanvasHistory()

      history.saveInitialState()

      expect(window.requestAnimationFrame).toHaveBeenCalled()
    })
  })
})
