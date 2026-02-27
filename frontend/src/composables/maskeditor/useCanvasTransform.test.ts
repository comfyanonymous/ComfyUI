import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useCanvasTransform } from '@/composables/maskeditor/useCanvasTransform'

interface IMockCanvas {
  width: number
  height: number
}

interface IMockContext {
  getImageData: ReturnType<typeof vi.fn>
  putImageData: ReturnType<typeof vi.fn>
  clearRect: ReturnType<typeof vi.fn>
  drawImage: ReturnType<typeof vi.fn>
}

interface IMockCanvasHistory {
  saveState: ReturnType<typeof vi.fn>
}

interface IMockStore {
  maskCanvas: IMockCanvas | null
  rgbCanvas: IMockCanvas | null
  imgCanvas: IMockCanvas | null
  maskCtx: IMockContext | null
  rgbCtx: IMockContext | null
  imgCtx: IMockContext | null
  tgpuRoot: unknown
  canvasHistory: IMockCanvasHistory
  gpuTexturesNeedRecreation: boolean
  gpuTextureWidth: number
  gpuTextureHeight: number
  pendingGPUMaskData: Uint8ClampedArray | null
  pendingGPURgbData: Uint8ClampedArray | null
}

const { mockStore, mockCanvasHistory } = vi.hoisted(() => {
  const mockCanvasHistory: IMockCanvasHistory = {
    saveState: vi.fn()
  }

  const mockStore: IMockStore = {
    maskCanvas: null,
    rgbCanvas: null,
    imgCanvas: null,
    maskCtx: null,
    rgbCtx: null,
    imgCtx: null,
    tgpuRoot: null,
    canvasHistory: mockCanvasHistory,
    gpuTexturesNeedRecreation: false,
    gpuTextureWidth: 0,
    gpuTextureHeight: 0,
    pendingGPUMaskData: null,
    pendingGPURgbData: null
  }

  return { mockStore, mockCanvasHistory }
})

vi.mock('@/stores/maskEditorStore', () => ({
  useMaskEditorStore: vi.fn(() => mockStore)
}))

// Mock ImageData with improved type safety
if (typeof globalThis.ImageData === 'undefined') {
  globalThis.ImageData = class ImageData {
    data: Uint8ClampedArray
    width: number
    height: number

    constructor(
      dataOrWidth: Uint8ClampedArray | number,
      widthOrHeight?: number,
      height?: number
    ) {
      if (dataOrWidth instanceof Uint8ClampedArray) {
        // Constructor overload: new ImageData(data, width, height)
        if (widthOrHeight === undefined || height === undefined) {
          throw new Error(
            'ImageData constructor requires width and height when data is provided'
          )
        }
        this.data = dataOrWidth
        this.width = widthOrHeight
        this.height = height
      } else {
        // Constructor overload: new ImageData(width, height)
        if (widthOrHeight === undefined) {
          throw new Error(
            'ImageData constructor requires height when width is provided'
          )
        }
        this.width = dataOrWidth
        this.height = widthOrHeight
        this.data = new Uint8ClampedArray(dataOrWidth * widthOrHeight * 4)
      }
    }
  } as typeof ImageData
}

// Mock ImageBitmap for test environment using safe type casting
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

describe('useCanvasTransform', () => {
  let mockMaskCanvas: IMockCanvas
  let mockRgbCanvas: IMockCanvas
  let mockImgCanvas: IMockCanvas
  let mockMaskCtx: IMockContext
  let mockRgbCtx: IMockContext
  let mockImgCtx: IMockContext

  beforeEach(() => {
    vi.clearAllMocks()

    const createMockImageData = (width: number, height: number) => {
      const data = new Uint8ClampedArray(width * height * 4)
      for (let i = 0; i < data.length; i += 4) {
        data[i] = 255 // R
        data[i + 1] = 0 // G
        data[i + 2] = 0 // B
        data[i + 3] = 255 // A
      }
      return {
        data,
        width,
        height
      } as ImageData
    }

    mockMaskCtx = {
      getImageData: vi.fn((_x, _y, w, h) => createMockImageData(w, h)),
      putImageData: vi.fn(),
      clearRect: vi.fn(),
      drawImage: vi.fn()
    }

    mockRgbCtx = {
      getImageData: vi.fn((_x, _y, w, h) => createMockImageData(w, h)),
      putImageData: vi.fn(),
      clearRect: vi.fn(),
      drawImage: vi.fn()
    }

    mockImgCtx = {
      getImageData: vi.fn((_x, _y, w, h) => createMockImageData(w, h)),
      putImageData: vi.fn(),
      clearRect: vi.fn(),
      drawImage: vi.fn()
    }

    mockMaskCanvas = {
      width: 100,
      height: 50
    }

    mockRgbCanvas = {
      width: 100,
      height: 50
    }

    mockImgCanvas = {
      width: 100,
      height: 50
    }

    mockStore.maskCanvas = mockMaskCanvas
    mockStore.rgbCanvas = mockRgbCanvas
    mockStore.imgCanvas = mockImgCanvas
    mockStore.maskCtx = mockMaskCtx
    mockStore.rgbCtx = mockRgbCtx
    mockStore.imgCtx = mockImgCtx
    mockStore.tgpuRoot = null
    mockStore.gpuTexturesNeedRecreation = false
    mockStore.gpuTextureWidth = 0
    mockStore.gpuTextureHeight = 0
    mockStore.pendingGPUMaskData = null
    mockStore.pendingGPURgbData = null
  })

  describe('rotateClockwise', () => {
    it('should rotate canvas 90 degrees clockwise', async () => {
      const transform = useCanvasTransform()
      await transform.rotateClockwise()

      expect(mockMaskCanvas.width).toBe(50)
      expect(mockMaskCanvas.height).toBe(100)
      expect(mockRgbCanvas.width).toBe(50)
      expect(mockRgbCanvas.height).toBe(100)
      expect(mockImgCanvas.width).toBe(50)
      expect(mockImgCanvas.height).toBe(100)
    })

    it('should call getImageData with original dimensions', async () => {
      const transform = useCanvasTransform()
      await transform.rotateClockwise()

      expect(mockMaskCtx.getImageData).toHaveBeenCalledWith(0, 0, 100, 50)
      expect(mockRgbCtx.getImageData).toHaveBeenCalledWith(0, 0, 100, 50)
      expect(mockImgCtx.getImageData).toHaveBeenCalledWith(0, 0, 100, 50)
    })

    it('should call putImageData with rotated data', async () => {
      const transform = useCanvasTransform()
      await transform.rotateClockwise()

      expect(mockMaskCtx.putImageData).toHaveBeenCalled()
      expect(mockRgbCtx.putImageData).toHaveBeenCalled()
      expect(mockImgCtx.putImageData).toHaveBeenCalled()

      const maskCall = mockMaskCtx.putImageData.mock.calls[0][0]
      expect(maskCall.width).toBe(50)
      expect(maskCall.height).toBe(100)
    })

    it('should save transformed state to history', async () => {
      const transform = useCanvasTransform()
      await transform.rotateClockwise()

      expect(mockCanvasHistory.saveState).toHaveBeenCalled()

      const savedArgs = mockCanvasHistory.saveState.mock.calls[0]
      expect(savedArgs).toHaveLength(3)

      expect(savedArgs[0].width).toBe(50)
      expect(savedArgs[0].height).toBe(100)
      expect(savedArgs[1].width).toBe(50)
      expect(savedArgs[1].height).toBe(100)
      expect(savedArgs[2].width).toBe(50)
      expect(savedArgs[2].height).toBe(100)
    })

    it('should log error when canvas contexts not ready', async () => {
      const consoleErrorSpy = vi
        .spyOn(console, 'error')
        .mockImplementation(() => {})
      mockStore.maskCanvas = null

      const transform = useCanvasTransform()
      await transform.rotateClockwise()

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        '[useCanvasTransform] Canvas contexts not ready'
      )

      consoleErrorSpy.mockRestore()
    })

    it('should handle GPU texture recreation when GPU is active', async () => {
      mockStore.tgpuRoot = {}

      const transform = useCanvasTransform()
      await transform.rotateClockwise()

      expect(mockStore.gpuTexturesNeedRecreation).toBe(true)
      expect(mockStore.gpuTextureWidth).toBe(50)
      expect(mockStore.gpuTextureHeight).toBe(100)
    })

    it('should not recreate GPU textures when GPU is inactive', async () => {
      mockStore.tgpuRoot = null

      const transform = useCanvasTransform()
      await transform.rotateClockwise()

      expect(mockStore.gpuTexturesNeedRecreation).toBe(false)
    })

    it('should correctly rotate pixels clockwise at pixel level', async () => {
      mockMaskCanvas.width = 2
      mockMaskCanvas.height = 2

      const createTestPattern = () => {
        const data = new Uint8ClampedArray(2 * 2 * 4)
        // TL (0,0): Red
        data[0] = 255
        data[1] = 0
        data[2] = 0
        data[3] = 255
        // TR (1,0): Green
        data[4] = 0
        data[5] = 255
        data[6] = 0
        data[7] = 255
        // BL (0,1): Blue
        data[8] = 0
        data[9] = 0
        data[10] = 255
        data[11] = 255
        // BR (1,1): Yellow
        data[12] = 255
        data[13] = 255
        data[14] = 0
        data[15] = 255
        return { data, width: 2, height: 2 } as ImageData
      }

      mockMaskCtx.getImageData = vi.fn(() => createTestPattern())
      const transform = useCanvasTransform()
      await transform.rotateClockwise()

      const result = mockMaskCtx.putImageData.mock.calls[0][0] as ImageData

      // After clockwise rotation:
      // New TL should be old BL (Blue)
      expect(result.data[0]).toBe(0) // R
      expect(result.data[1]).toBe(0) // G
      expect(result.data[2]).toBe(255) // B
      expect(result.data[3]).toBe(255) // A

      // New TR should be old TL (Red)
      expect(result.data[4]).toBe(255) // R
      expect(result.data[5]).toBe(0) // G
      expect(result.data[6]).toBe(0) // B
      expect(result.data[7]).toBe(255) // A

      // New BL should be old BR (Yellow)
      expect(result.data[8]).toBe(255) // R
      expect(result.data[9]).toBe(255) // G
      expect(result.data[10]).toBe(0) // B
      expect(result.data[11]).toBe(255) // A

      // New BR should be old TR (Green)
      expect(result.data[12]).toBe(0) // R
      expect(result.data[13]).toBe(255) // G
      expect(result.data[14]).toBe(0) // B
      expect(result.data[15]).toBe(255) // A
    })
  })

  describe('rotateCounterclockwise', () => {
    it('should rotate canvas 90 degrees counterclockwise', async () => {
      const transform = useCanvasTransform()
      await transform.rotateCounterclockwise()

      expect(mockMaskCanvas.width).toBe(50)
      expect(mockMaskCanvas.height).toBe(100)
    })

    it('should call getImageData with original dimensions', async () => {
      const transform = useCanvasTransform()
      await transform.rotateCounterclockwise()

      expect(mockMaskCtx.getImageData).toHaveBeenCalledWith(0, 0, 100, 50)
    })

    it('should correctly rotate pixels counterclockwise at pixel level', async () => {
      mockMaskCanvas.width = 2
      mockMaskCanvas.height = 2

      const createTestPattern = () => {
        const data = new Uint8ClampedArray(2 * 2 * 4)
        // TL (0,0): Red
        data[0] = 255
        data[1] = 0
        data[2] = 0
        data[3] = 255
        // TR (1,0): Green
        data[4] = 0
        data[5] = 255
        data[6] = 0
        data[7] = 255
        // BL (0,1): Blue
        data[8] = 0
        data[9] = 0
        data[10] = 255
        data[11] = 255
        // BR (1,1): Yellow
        data[12] = 255
        data[13] = 255
        data[14] = 0
        data[15] = 255
        return { data, width: 2, height: 2 } as ImageData
      }

      mockMaskCtx.getImageData = vi.fn(() => createTestPattern())
      const transform = useCanvasTransform()
      await transform.rotateCounterclockwise()

      const result = mockMaskCtx.putImageData.mock.calls[0][0] as ImageData

      // After counterclockwise rotation:
      // New TL should be old TR (Green)
      expect(result.data[0]).toBe(0) // R
      expect(result.data[1]).toBe(255) // G
      expect(result.data[2]).toBe(0) // B
      expect(result.data[3]).toBe(255) // A

      // New TR should be old BR (Yellow)
      expect(result.data[4]).toBe(255) // R
      expect(result.data[5]).toBe(255) // G
      expect(result.data[6]).toBe(0) // B
      expect(result.data[7]).toBe(255) // A

      // New BL should be old TL (Red)
      expect(result.data[8]).toBe(255) // R
      expect(result.data[9]).toBe(0) // G
      expect(result.data[10]).toBe(0) // B
      expect(result.data[11]).toBe(255) // A

      // New BR should be old BL (Blue)
      expect(result.data[12]).toBe(0) // R
      expect(result.data[13]).toBe(0) // G
      expect(result.data[14]).toBe(255) // B
      expect(result.data[15]).toBe(255) // A
    })

    it('should produce different result than clockwise rotation', async () => {
      const transform = useCanvasTransform()

      const createAsymmetricImageData = (width: number, height: number) => {
        const data = new Uint8ClampedArray(width * height * 4)
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4
            if (x < width / 2 && y < height / 2) {
              data[i] = 255
              data[i + 3] = 255
            } else {
              data[i + 3] = 255
            }
          }
        }
        return { data, width, height } as ImageData
      }

      mockMaskCtx.getImageData = vi.fn(() => createAsymmetricImageData(100, 50))
      await transform.rotateCounterclockwise()
      const ccwResult = mockMaskCtx.putImageData.mock.calls[0][0] as ImageData

      mockMaskCanvas.width = 100
      mockMaskCanvas.height = 50
      mockMaskCtx.putImageData.mockClear()

      mockMaskCtx.getImageData = vi.fn(() => createAsymmetricImageData(100, 50))
      await transform.rotateClockwise()
      const cwResult = mockMaskCtx.putImageData.mock.calls[0][0] as ImageData

      let pixelDifferences = 0
      for (let i = 0; i < ccwResult.data.length; i++) {
        if (ccwResult.data[i] !== cwResult.data[i]) {
          pixelDifferences++
        }
      }

      expect(pixelDifferences).toBeGreaterThan(0)
    })
  })

  describe('mirrorHorizontal', () => {
    it('should mirror canvas horizontally', async () => {
      const transform = useCanvasTransform()
      await transform.mirrorHorizontal()

      expect(mockMaskCanvas.width).toBe(100)
      expect(mockMaskCanvas.height).toBe(50)
    })

    it('should handle GPU texture recreation when GPU is active', async () => {
      mockStore.tgpuRoot = {}

      const transform = useCanvasTransform()
      await transform.mirrorHorizontal()

      expect(mockStore.gpuTexturesNeedRecreation).toBe(true)
      expect(mockStore.gpuTextureWidth).toBe(100)
      expect(mockStore.gpuTextureHeight).toBe(50)
    })

    it('should correctly flip pixels horizontally at pixel level', async () => {
      mockMaskCanvas.width = 2
      mockMaskCanvas.height = 2

      const createTestPattern = () => {
        const data = new Uint8ClampedArray(2 * 2 * 4)
        // TL (0,0): Red
        data[0] = 255
        data[1] = 0
        data[2] = 0
        data[3] = 255
        // TR (1,0): Green
        data[4] = 0
        data[5] = 255
        data[6] = 0
        data[7] = 255
        // BL (0,1): Blue
        data[8] = 0
        data[9] = 0
        data[10] = 255
        data[11] = 255
        // BR (1,1): Yellow
        data[12] = 255
        data[13] = 255
        data[14] = 0
        data[15] = 255
        return { data, width: 2, height: 2 } as ImageData
      }

      mockMaskCtx.getImageData = vi.fn(() => createTestPattern())
      const transform = useCanvasTransform()
      await transform.mirrorHorizontal()

      const result = mockMaskCtx.putImageData.mock.calls[0][0] as ImageData

      // After horizontal flip:
      // New TL should be old TR (Green)
      expect(result.data[0]).toBe(0)
      expect(result.data[1]).toBe(255)
      // New TR should be old TL (Red)
      expect(result.data[4]).toBe(255)
      expect(result.data[5]).toBe(0)
    })
  })

  describe('mirrorVertical', () => {
    it('should mirror canvas vertically', async () => {
      const transform = useCanvasTransform()
      await transform.mirrorVertical()

      expect(mockMaskCanvas.width).toBe(100)
      expect(mockMaskCanvas.height).toBe(50)
    })

    it('should handle GPU texture recreation when GPU is active', async () => {
      mockStore.tgpuRoot = {}

      const transform = useCanvasTransform()
      await transform.mirrorVertical()

      expect(mockStore.gpuTexturesNeedRecreation).toBe(true)
      expect(mockStore.gpuTextureWidth).toBe(100)
      expect(mockStore.gpuTextureHeight).toBe(50)
    })

    it('should correctly flip pixels vertically at pixel level', async () => {
      mockMaskCanvas.width = 2
      mockMaskCanvas.height = 2

      const createTestPattern = () => {
        const data = new Uint8ClampedArray(2 * 2 * 4)
        // TL (0,0): Red
        data[0] = 255
        data[1] = 0
        data[2] = 0
        data[3] = 255
        // TR (1,0): Green
        data[4] = 0
        data[5] = 255
        data[6] = 0
        data[7] = 255
        // BL (0,1): Blue
        data[8] = 0
        data[9] = 0
        data[10] = 255
        data[11] = 255
        // BR (1,1): Yellow
        data[12] = 255
        data[13] = 255
        data[14] = 0
        data[15] = 255
        return { data, width: 2, height: 2 } as ImageData
      }

      mockMaskCtx.getImageData = vi.fn(() => createTestPattern())
      const transform = useCanvasTransform()
      await transform.mirrorVertical()

      const result = mockMaskCtx.putImageData.mock.calls[0][0] as ImageData

      // After vertical flip:
      // New TL should be old BL (Blue)
      expect(result.data[0]).toBe(0) // R
      expect(result.data[1]).toBe(0) // G
      expect(result.data[2]).toBe(255) // B
      expect(result.data[3]).toBe(255) // A

      // New TR should be old BR (Yellow)
      expect(result.data[4]).toBe(255) // R
      expect(result.data[5]).toBe(255) // G
      expect(result.data[6]).toBe(0) // B
      expect(result.data[7]).toBe(255) // A

      // New BL should be old TL (Red)
      expect(result.data[8]).toBe(255) // R
      expect(result.data[9]).toBe(0) // G
      expect(result.data[10]).toBe(0) // B
      expect(result.data[11]).toBe(255) // A

      // New BR should be old TR (Green)
      expect(result.data[12]).toBe(0) // R
      expect(result.data[13]).toBe(255) // G
      expect(result.data[14]).toBe(0) // B
      expect(result.data[15]).toBe(255) // A
    })

    it('should log error when canvas contexts not ready', async () => {
      const consoleErrorSpy = vi
        .spyOn(console, 'error')
        .mockImplementation(() => {})
      mockStore.maskCanvas = null

      const transform = useCanvasTransform()
      await transform.mirrorVertical()

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        '[useCanvasTransform] Canvas contexts not ready'
      )

      consoleErrorSpy.mockRestore()
    })
  })

  describe('GPU integration', () => {
    it('should set GPU recreation flags for rotation', async () => {
      mockStore.tgpuRoot = {}
      mockMaskCanvas.width = 100
      mockMaskCanvas.height = 50

      const transform = useCanvasTransform()
      await transform.rotateClockwise()

      expect(mockStore.gpuTexturesNeedRecreation).toBe(true)
      expect(mockStore.gpuTextureWidth).toBe(50)
      expect(mockStore.gpuTextureHeight).toBe(100)
      expect(mockStore.pendingGPUMaskData!.length).toBe(50 * 100 * 4)
      expect(mockStore.pendingGPURgbData!.length).toBe(50 * 100 * 4)
    })

    it('should premultiply alpha when preparing GPU data', async () => {
      mockStore.tgpuRoot = {}
      mockMaskCanvas.width = 1
      mockMaskCanvas.height = 1

      // Create 1x1 ImageData with semi-transparent pixel
      const createSemiTransparentImageData = () => {
        const data = new Uint8ClampedArray(1 * 1 * 4)
        data[0] = 200 // R
        data[1] = 100 // G
        data[2] = 50 // B
        data[3] = 128 // A (50% opacity)
        return { data, width: 1, height: 1 } as ImageData
      }

      mockMaskCtx.getImageData = vi.fn(() => createSemiTransparentImageData())
      mockRgbCtx.getImageData = vi.fn(() => createSemiTransparentImageData())
      mockImgCtx.getImageData = vi.fn(() => createSemiTransparentImageData())

      const transform = useCanvasTransform()
      await transform.rotateClockwise()

      // Verify pendingGPUMaskData contains premultiplied values
      expect(mockStore.pendingGPUMaskData).not.toBeNull()
      const maskData = mockStore.pendingGPUMaskData!

      // Expected premultiplied values: RGB * alpha / 255
      // R: 200 * 128 / 255 ≈ 100
      // G: 100 * 128 / 255 ≈ 50
      // B: 50 * 128 / 255 ≈ 25
      // A: 128 (preserved)
      expect(maskData[0]).toBeCloseTo(100, 0) // R premultiplied
      expect(maskData[1]).toBeCloseTo(50, 0) // G premultiplied
      expect(maskData[2]).toBeCloseTo(25, 0) // B premultiplied
      expect(maskData[3]).toBe(128) // A preserved

      // Also verify RGB canvas data
      expect(mockStore.pendingGPURgbData).not.toBeNull()
      const rgbData = mockStore.pendingGPURgbData!
      expect(rgbData[0]).toBeCloseTo(100, 0)
      expect(rgbData[1]).toBeCloseTo(50, 0)
      expect(rgbData[2]).toBeCloseTo(25, 0)
      expect(rgbData[3]).toBe(128)
    })
  })
})
