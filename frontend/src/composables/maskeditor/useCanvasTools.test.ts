import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ColorComparisonMethod } from '@/extensions/core/maskeditor/types'

import { useCanvasTools } from '@/composables/maskeditor/useCanvasTools'

// Mock store interface matching the real store's nullable fields
interface MockMaskEditorStore {
  maskCtx: CanvasRenderingContext2D | null
  imgCtx: CanvasRenderingContext2D | null
  maskCanvas: HTMLCanvasElement | null
  imgCanvas: HTMLCanvasElement | null
  rgbCtx: CanvasRenderingContext2D | null
  rgbCanvas: HTMLCanvasElement | null
  maskColor: { r: number; g: number; b: number }
  paintBucketTolerance: number
  fillOpacity: number
  colorSelectTolerance: number
  colorComparisonMethod: ColorComparisonMethod
  selectionOpacity: number
  applyWholeImage: boolean
  maskBoundary: boolean
  maskTolerance: number
  canvasHistory: { saveState: ReturnType<typeof vi.fn> }
}

const mockCanvasHistory = {
  saveState: vi.fn()
}

const mockStore: MockMaskEditorStore = {
  maskCtx: null,
  imgCtx: null,
  maskCanvas: null,
  imgCanvas: null,
  rgbCtx: null,
  rgbCanvas: null,
  maskColor: { r: 255, g: 255, b: 255 },
  paintBucketTolerance: 10,
  fillOpacity: 100,
  colorSelectTolerance: 30,
  colorComparisonMethod: ColorComparisonMethod.Simple,
  selectionOpacity: 100,
  applyWholeImage: false,
  maskBoundary: false,
  maskTolerance: 10,
  canvasHistory: mockCanvasHistory
}

vi.mock('@/stores/maskEditorStore', () => ({
  useMaskEditorStore: vi.fn(() => mockStore)
}))

describe('useCanvasTools', () => {
  let mockMaskImageData: ImageData
  let mockImgImageData: ImageData

  beforeEach(() => {
    vi.clearAllMocks()

    mockMaskImageData = {
      data: new Uint8ClampedArray(100 * 100 * 4),
      width: 100,
      height: 100
    } as ImageData

    mockImgImageData = {
      data: new Uint8ClampedArray(100 * 100 * 4),
      width: 100,
      height: 100
    } as ImageData

    for (let i = 0; i < mockImgImageData.data.length; i += 4) {
      mockImgImageData.data[i] = 255
      mockImgImageData.data[i + 1] = 0
      mockImgImageData.data[i + 2] = 0
      mockImgImageData.data[i + 3] = 255
    }

    const partialMaskCtx: Partial<CanvasRenderingContext2D> = {
      getImageData: vi.fn(() => mockMaskImageData),
      putImageData: vi.fn(),
      clearRect: vi.fn()
    }
    mockStore.maskCtx = partialMaskCtx as CanvasRenderingContext2D

    const partialImgCtx: Partial<CanvasRenderingContext2D> = {
      getImageData: vi.fn(() => mockImgImageData)
    }
    mockStore.imgCtx = partialImgCtx as CanvasRenderingContext2D

    const partialRgbCtx: Partial<CanvasRenderingContext2D> = {
      clearRect: vi.fn()
    }
    mockStore.rgbCtx = partialRgbCtx as CanvasRenderingContext2D

    const partialMaskCanvas: Partial<HTMLCanvasElement> = {
      width: 100,
      height: 100
    }
    mockStore.maskCanvas = partialMaskCanvas as HTMLCanvasElement

    const partialImgCanvas: Partial<HTMLCanvasElement> = {
      width: 100,
      height: 100
    }
    mockStore.imgCanvas = partialImgCanvas as HTMLCanvasElement

    const partialRgbCanvas: Partial<HTMLCanvasElement> = {
      width: 100,
      height: 100
    }
    mockStore.rgbCanvas = partialRgbCanvas as HTMLCanvasElement

    mockStore.maskColor = { r: 255, g: 255, b: 255 }
    mockStore.paintBucketTolerance = 10
    mockStore.fillOpacity = 100
    mockStore.colorSelectTolerance = 30
    mockStore.colorComparisonMethod = ColorComparisonMethod.Simple
    mockStore.selectionOpacity = 100
    mockStore.applyWholeImage = false
    mockStore.maskBoundary = false
    mockStore.maskTolerance = 10
  })

  describe('paintBucketFill', () => {
    it('should fill empty area with mask color', () => {
      const tools = useCanvasTools()

      tools.paintBucketFill({ x: 50, y: 50 })

      expect(mockStore.maskCtx!.getImageData).toHaveBeenCalledWith(
        0,
        0,
        100,
        100
      )
      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalledWith(
        mockMaskImageData,
        0,
        0
      )
      expect(mockCanvasHistory.saveState).toHaveBeenCalled()

      const index = (50 * 100 + 50) * 4
      expect(mockMaskImageData.data[index + 3]).toBe(255)
    })

    it('should erase filled area', () => {
      for (let i = 0; i < mockMaskImageData.data.length; i += 4) {
        mockMaskImageData.data[i + 3] = 255
      }

      const tools = useCanvasTools()

      tools.paintBucketFill({ x: 50, y: 50 })

      const index = (50 * 100 + 50) * 4
      expect(mockMaskImageData.data[index + 3]).toBe(0)
    })

    it('should respect tolerance', () => {
      mockStore.paintBucketTolerance = 0

      for (let i = 0; i < mockMaskImageData.data.length; i += 4) {
        mockMaskImageData.data[i + 3] = 0
      }
      const centerIndex = (50 * 100 + 50) * 4
      mockMaskImageData.data[centerIndex + 3] = 10

      const tools = useCanvasTools()

      tools.paintBucketFill({ x: 51, y: 50 })

      expect(mockMaskImageData.data[centerIndex + 3]).toBe(10)
    })

    it('should return early when point out of bounds', () => {
      const tools = useCanvasTools()

      tools.paintBucketFill({ x: -1, y: 50 })

      expect(mockStore.maskCtx!.putImageData).not.toHaveBeenCalled()
    })

    it('should return early when canvas missing', () => {
      mockStore.maskCanvas = null

      const tools = useCanvasTools()

      tools.paintBucketFill({ x: 50, y: 50 })

      expect(mockStore.maskCtx?.getImageData).not.toHaveBeenCalled()
    })

    it('should apply fill opacity', () => {
      mockStore.fillOpacity = 50

      const tools = useCanvasTools()

      tools.paintBucketFill({ x: 50, y: 50 })

      const index = (50 * 100 + 50) * 4
      expect(mockMaskImageData.data[index + 3]).toBe(127)
    })

    it('should apply mask color', () => {
      mockStore.maskColor = { r: 128, g: 64, b: 32 }

      const tools = useCanvasTools()

      tools.paintBucketFill({ x: 50, y: 50 })

      const index = (50 * 100 + 50) * 4
      expect(mockMaskImageData.data[index]).toBe(128)
      expect(mockMaskImageData.data[index + 1]).toBe(64)
      expect(mockMaskImageData.data[index + 2]).toBe(32)
    })
  })

  describe('colorSelectFill', () => {
    it('should select pixels by color with flood fill', async () => {
      const tools = useCanvasTools()

      await tools.colorSelectFill({ x: 50, y: 50 })

      expect(mockStore.maskCtx!.getImageData).toHaveBeenCalledWith(
        0,
        0,
        100,
        100
      )
      expect(mockStore.imgCtx!.getImageData).toHaveBeenCalledWith(
        0,
        0,
        100,
        100
      )
      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalled()
      expect(mockCanvasHistory.saveState).toHaveBeenCalled()
    })

    it('should select pixels in whole image when applyWholeImage is true', async () => {
      mockStore.applyWholeImage = true

      const tools = useCanvasTools()

      await tools.colorSelectFill({ x: 50, y: 50 })

      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalled()
    })

    it('should respect color tolerance', async () => {
      mockStore.colorSelectTolerance = 0

      for (let i = 0; i < mockImgImageData.data.length; i += 4) {
        mockImgImageData.data[i] = 200
      }

      const tools = useCanvasTools()

      await tools.colorSelectFill({ x: 50, y: 50 })

      const index = (50 * 100 + 50) * 4
      expect(mockMaskImageData.data[index + 3]).toBe(255)
    })

    it('should return early when point out of bounds', async () => {
      const tools = useCanvasTools()

      await tools.colorSelectFill({ x: -1, y: 50 })

      expect(mockStore.maskCtx!.putImageData).not.toHaveBeenCalled()
    })

    it('should return early when canvas missing', async () => {
      mockStore.imgCanvas = null

      const tools = useCanvasTools()

      await tools.colorSelectFill({ x: 50, y: 50 })

      expect(mockStore.maskCtx?.getImageData).not.toHaveBeenCalled()
    })

    it('should apply selection opacity', async () => {
      mockStore.selectionOpacity = 50

      const tools = useCanvasTools()

      await tools.colorSelectFill({ x: 50, y: 50 })

      const index = (50 * 100 + 50) * 4
      expect(mockMaskImageData.data[index + 3]).toBe(127)
    })

    it('should use HSL color comparison method', async () => {
      mockStore.colorComparisonMethod = ColorComparisonMethod.HSL

      const tools = useCanvasTools()

      await tools.colorSelectFill({ x: 50, y: 50 })

      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalled()
    })

    it('should use LAB color comparison method', async () => {
      mockStore.colorComparisonMethod = ColorComparisonMethod.LAB

      const tools = useCanvasTools()

      await tools.colorSelectFill({ x: 50, y: 50 })

      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalled()
    })

    it('should respect mask boundary', async () => {
      mockStore.maskBoundary = true
      mockStore.maskTolerance = 0

      for (let i = 0; i < mockMaskImageData.data.length; i += 4) {
        mockMaskImageData.data[i + 3] = 255
      }

      const tools = useCanvasTools()

      await tools.colorSelectFill({ x: 50, y: 50 })

      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalled()
    })

    it('should update last color select point', async () => {
      const tools = useCanvasTools()

      await tools.colorSelectFill({ x: 30, y: 40 })

      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalled()
    })
  })

  describe('invertMask', () => {
    it('should invert mask alpha values', () => {
      for (let i = 0; i < mockMaskImageData.data.length; i += 4) {
        mockMaskImageData.data[i] = 255
        mockMaskImageData.data[i + 1] = 255
        mockMaskImageData.data[i + 2] = 255
        mockMaskImageData.data[i + 3] = 128
      }

      const tools = useCanvasTools()

      tools.invertMask()

      expect(mockStore.maskCtx!.getImageData).toHaveBeenCalledWith(
        0,
        0,
        100,
        100
      )
      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalledWith(
        mockMaskImageData,
        0,
        0
      )
      expect(mockCanvasHistory.saveState).toHaveBeenCalled()

      for (let i = 0; i < mockMaskImageData.data.length; i += 4) {
        expect(mockMaskImageData.data[i + 3]).toBe(127)
      }
    })

    it('should preserve mask color for empty pixels', () => {
      for (let i = 0; i < mockMaskImageData.data.length; i += 4) {
        mockMaskImageData.data[i + 3] = 0
      }

      const firstPixelIndex = 100
      mockMaskImageData.data[firstPixelIndex * 4] = 128
      mockMaskImageData.data[firstPixelIndex * 4 + 1] = 64
      mockMaskImageData.data[firstPixelIndex * 4 + 2] = 32
      mockMaskImageData.data[firstPixelIndex * 4 + 3] = 255

      const tools = useCanvasTools()

      tools.invertMask()

      for (let i = 0; i < mockMaskImageData.data.length; i += 4) {
        if (i !== firstPixelIndex * 4) {
          expect(mockMaskImageData.data[i]).toBe(128)
          expect(mockMaskImageData.data[i + 1]).toBe(64)
          expect(mockMaskImageData.data[i + 2]).toBe(32)
        }
      }
    })

    it('should return early when canvas missing', () => {
      mockStore.maskCanvas = null

      const tools = useCanvasTools()

      tools.invertMask()

      expect(mockStore.maskCtx?.getImageData).not.toHaveBeenCalled()
    })

    it('should return early when context missing', () => {
      mockStore.maskCtx = null

      const tools = useCanvasTools()

      tools.invertMask()

      expect(mockCanvasHistory.saveState).not.toHaveBeenCalled()
    })
  })

  describe('clearMask', () => {
    it('should clear mask canvas', () => {
      const tools = useCanvasTools()

      tools.clearMask()

      expect(mockStore.maskCtx!.clearRect).toHaveBeenCalledWith(0, 0, 100, 100)
      expect(mockStore.rgbCtx!.clearRect).toHaveBeenCalledWith(0, 0, 100, 100)
      expect(mockCanvasHistory.saveState).toHaveBeenCalled()
    })

    it('should handle missing mask canvas', () => {
      mockStore.maskCanvas = null

      const tools = useCanvasTools()

      tools.clearMask()

      expect(mockStore.maskCtx?.clearRect).not.toHaveBeenCalled()
      expect(mockCanvasHistory.saveState).toHaveBeenCalled()
    })

    it('should handle missing rgb canvas', () => {
      mockStore.rgbCanvas = null

      const tools = useCanvasTools()

      tools.clearMask()

      expect(mockStore.maskCtx?.clearRect).toHaveBeenCalledWith(0, 0, 100, 100)
      expect(mockStore.rgbCtx?.clearRect).not.toHaveBeenCalled()
      expect(mockCanvasHistory.saveState).toHaveBeenCalled()
    })
  })

  describe('clearLastColorSelectPoint', () => {
    it('should clear last color select point', async () => {
      const tools = useCanvasTools()

      await tools.colorSelectFill({ x: 50, y: 50 })

      tools.clearLastColorSelectPoint()

      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalled()
    })
  })

  describe('edge cases', () => {
    it('should handle small canvas', () => {
      mockStore.maskCanvas!.width = 1
      mockStore.maskCanvas!.height = 1
      mockMaskImageData = {
        data: new Uint8ClampedArray(1 * 1 * 4),
        width: 1,
        height: 1
      } as ImageData
      mockStore.maskCtx!.getImageData = vi.fn(() => mockMaskImageData)

      const tools = useCanvasTools()

      tools.paintBucketFill({ x: 0, y: 0 })

      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalled()
    })

    it('should handle fractional coordinates', () => {
      const tools = useCanvasTools()

      tools.paintBucketFill({ x: 50.7, y: 50.3 })

      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalled()
    })

    it('should handle maximum tolerance', () => {
      mockStore.paintBucketTolerance = 255

      const tools = useCanvasTools()

      tools.paintBucketFill({ x: 50, y: 50 })

      expect(mockStore.maskCtx!.putImageData).toHaveBeenCalled()
    })

    it('should handle zero opacity', () => {
      mockStore.fillOpacity = 0

      const tools = useCanvasTools()

      tools.paintBucketFill({ x: 50, y: 50 })

      const index = (50 * 100 + 50) * 4
      expect(mockMaskImageData.data[index + 3]).toBe(0)
    })
  })
})
