import { beforeEach, describe, expect, it, vi } from 'vitest'

import { MaskBlendMode } from '@/extensions/core/maskeditor/types'
import { useCanvasManager } from '@/composables/maskeditor/useCanvasManager'
const mockStore = {
  imgCanvas: null! as HTMLCanvasElement,
  maskCanvas: null! as HTMLCanvasElement,
  rgbCanvas: null! as HTMLCanvasElement,
  imgCtx: null! as CanvasRenderingContext2D,
  maskCtx: null! as CanvasRenderingContext2D,
  rgbCtx: null! as CanvasRenderingContext2D,
  canvasBackground: null! as HTMLElement,
  maskColor: { r: 0, g: 0, b: 0 },
  maskBlendMode: MaskBlendMode.Black,
  maskOpacity: 0.8
}

vi.mock('@/stores/maskEditorStore', () => ({
  useMaskEditorStore: vi.fn(() => mockStore)
}))

function createMockImage(width: number, height: number): HTMLImageElement {
  return {
    width,
    height
  } as HTMLImageElement
}

describe('useCanvasManager', () => {
  let mockImageData: ImageData

  beforeEach(() => {
    vi.clearAllMocks()

    mockImageData = {
      data: new Uint8ClampedArray(100 * 100 * 4),
      width: 100,
      height: 100
    } as ImageData

    const partialImgCtx: Partial<CanvasRenderingContext2D> = {
      drawImage: vi.fn()
    }
    mockStore.imgCtx = partialImgCtx as CanvasRenderingContext2D

    const partialMaskCtx: Partial<CanvasRenderingContext2D> = {
      drawImage: vi.fn(),
      getImageData: vi.fn(() => mockImageData),
      putImageData: vi.fn(),
      globalCompositeOperation: 'source-over',
      fillStyle: ''
    }
    mockStore.maskCtx = partialMaskCtx as CanvasRenderingContext2D

    const partialRgbCtx: Partial<CanvasRenderingContext2D> = {
      drawImage: vi.fn()
    }
    mockStore.rgbCtx = partialRgbCtx as CanvasRenderingContext2D

    const partialImgCanvas: Partial<HTMLCanvasElement> = {
      width: 0,
      height: 0
    }
    mockStore.imgCanvas = partialImgCanvas as HTMLCanvasElement

    mockStore.maskCanvas = {
      width: 0,
      height: 0,
      style: {
        mixBlendMode: '',
        opacity: ''
      } as Pick<CSSStyleDeclaration, 'mixBlendMode' | 'opacity'>
    } as HTMLCanvasElement

    mockStore.rgbCanvas = {
      width: 0,
      height: 0
    } as HTMLCanvasElement

    mockStore.canvasBackground = {
      style: {
        backgroundColor: ''
      } as Pick<CSSStyleDeclaration, 'backgroundColor'>
    } as HTMLElement

    mockStore.maskColor = { r: 0, g: 0, b: 0 }
    mockStore.maskBlendMode = MaskBlendMode.Black
    mockStore.maskOpacity = 0.8
  })

  describe('invalidateCanvas', () => {
    it('should set canvas dimensions', async () => {
      const manager = useCanvasManager()

      const origImage = createMockImage(512, 512)
      const maskImage = createMockImage(512, 512)

      await manager.invalidateCanvas(origImage, maskImage, null)

      expect(mockStore.imgCanvas.width).toBe(512)
      expect(mockStore.imgCanvas.height).toBe(512)
      expect(mockStore.maskCanvas.width).toBe(512)
      expect(mockStore.maskCanvas.height).toBe(512)
      expect(mockStore.rgbCanvas.width).toBe(512)
      expect(mockStore.rgbCanvas.height).toBe(512)
    })

    it('should draw original image', async () => {
      const manager = useCanvasManager()

      const origImage = createMockImage(512, 512)
      const maskImage = createMockImage(512, 512)

      await manager.invalidateCanvas(origImage, maskImage, null)

      expect(mockStore.imgCtx.drawImage).toHaveBeenCalledWith(
        origImage,
        0,
        0,
        512,
        512
      )
    })

    it('should draw paint image when provided', async () => {
      const manager = useCanvasManager()

      const origImage = createMockImage(512, 512)
      const maskImage = createMockImage(512, 512)
      const paintImage = createMockImage(512, 512)

      await manager.invalidateCanvas(origImage, maskImage, paintImage)

      expect(mockStore.rgbCtx.drawImage).toHaveBeenCalledWith(
        paintImage,
        0,
        0,
        512,
        512
      )
    })

    it('should not draw paint image when null', async () => {
      const manager = useCanvasManager()

      const origImage = createMockImage(512, 512)
      const maskImage = createMockImage(512, 512)

      await manager.invalidateCanvas(origImage, maskImage, null)

      expect(mockStore.rgbCtx.drawImage).not.toHaveBeenCalled()
    })

    it('should prepare mask', async () => {
      const manager = useCanvasManager()

      const origImage = createMockImage(512, 512)
      const maskImage = createMockImage(512, 512)

      await manager.invalidateCanvas(origImage, maskImage, null)

      expect(mockStore.maskCtx.drawImage).toHaveBeenCalled()
      expect(mockStore.maskCtx.getImageData).toHaveBeenCalled()
      expect(mockStore.maskCtx.putImageData).toHaveBeenCalled()
    })

    it('should throw error when canvas missing', async () => {
      const manager = useCanvasManager()

      mockStore.imgCanvas = null! as HTMLCanvasElement

      const origImage = createMockImage(512, 512)
      const maskImage = createMockImage(512, 512)

      await expect(
        manager.invalidateCanvas(origImage, maskImage, null)
      ).rejects.toThrow('Canvas elements or contexts not available')
    })

    it('should throw error when context missing', async () => {
      const manager = useCanvasManager()

      mockStore.imgCtx = null! as CanvasRenderingContext2D

      const origImage = createMockImage(512, 512)
      const maskImage = createMockImage(512, 512)

      await expect(
        manager.invalidateCanvas(origImage, maskImage, null)
      ).rejects.toThrow('Canvas elements or contexts not available')
    })
  })

  describe('updateMaskColor', () => {
    it('should update mask color for black blend mode', async () => {
      const manager = useCanvasManager()

      mockStore.maskBlendMode = MaskBlendMode.Black
      mockStore.maskColor = { r: 0, g: 0, b: 0 }

      await manager.updateMaskColor()

      expect(mockStore.maskCtx.fillStyle).toBe('rgb(0, 0, 0)')
      expect(mockStore.maskCanvas.style.mixBlendMode).toBe('initial')
      expect(mockStore.maskCanvas.style.opacity).toBe('0.8')
      expect(mockStore.canvasBackground.style.backgroundColor).toBe(
        'rgba(0,0,0,1)'
      )
    })

    it('should update mask color for white blend mode', async () => {
      const manager = useCanvasManager()

      mockStore.maskBlendMode = MaskBlendMode.White
      mockStore.maskColor = { r: 255, g: 255, b: 255 }

      await manager.updateMaskColor()

      expect(mockStore.maskCtx.fillStyle).toBe('rgb(255, 255, 255)')
      expect(mockStore.maskCanvas.style.mixBlendMode).toBe('initial')
      expect(mockStore.canvasBackground.style.backgroundColor).toBe(
        'rgba(255,255,255,1)'
      )
    })

    it('should update mask color for negative blend mode', async () => {
      const manager = useCanvasManager()

      mockStore.maskBlendMode = MaskBlendMode.Negative
      mockStore.maskColor = { r: 255, g: 255, b: 255 }

      await manager.updateMaskColor()

      expect(mockStore.maskCanvas.style.mixBlendMode).toBe('difference')
      expect(mockStore.maskCanvas.style.opacity).toBe('1')
      expect(mockStore.canvasBackground.style.backgroundColor).toBe(
        'rgba(255,255,255,1)'
      )
    })

    it('should update all pixels with mask color', async () => {
      const manager = useCanvasManager()

      mockStore.maskColor = { r: 128, g: 64, b: 32 }
      mockStore.maskCanvas.width = 100
      mockStore.maskCanvas.height = 100

      await manager.updateMaskColor()

      for (let i = 0; i < mockImageData.data.length; i += 4) {
        expect(mockImageData.data[i]).toBe(128)
        expect(mockImageData.data[i + 1]).toBe(64)
        expect(mockImageData.data[i + 2]).toBe(32)
      }

      expect(mockStore.maskCtx.putImageData).toHaveBeenCalledWith(
        mockImageData,
        0,
        0
      )
    })

    it('should return early when canvas missing', async () => {
      const manager = useCanvasManager()

      mockStore.maskCanvas = null! as HTMLCanvasElement

      await manager.updateMaskColor()

      expect(mockStore.maskCtx.getImageData).not.toHaveBeenCalled()
    })

    it('should return early when context missing', async () => {
      const manager = useCanvasManager()

      mockStore.maskCtx = null! as CanvasRenderingContext2D

      await manager.updateMaskColor()

      expect(mockStore.canvasBackground.style.backgroundColor).toBe('')
    })

    it('should handle different opacity values', async () => {
      const manager = useCanvasManager()

      mockStore.maskOpacity = 0.5

      await manager.updateMaskColor()

      expect(mockStore.maskCanvas.style.opacity).toBe('0.5')
    })
  })

  describe('prepareMask', () => {
    it('should invert mask alpha', async () => {
      const manager = useCanvasManager()

      for (let i = 0; i < mockImageData.data.length; i += 4) {
        mockImageData.data[i + 3] = 128
      }

      const origImage = createMockImage(100, 100)
      const maskImage = createMockImage(100, 100)

      await manager.invalidateCanvas(origImage, maskImage, null)

      for (let i = 0; i < mockImageData.data.length; i += 4) {
        expect(mockImageData.data[i + 3]).toBe(127)
      }
    })

    it('should apply mask color to all pixels', async () => {
      const manager = useCanvasManager()

      mockStore.maskColor = { r: 100, g: 150, b: 200 }

      const origImage = createMockImage(100, 100)
      const maskImage = createMockImage(100, 100)

      await manager.invalidateCanvas(origImage, maskImage, null)

      for (let i = 0; i < mockImageData.data.length; i += 4) {
        expect(mockImageData.data[i]).toBe(100)
        expect(mockImageData.data[i + 1]).toBe(150)
        expect(mockImageData.data[i + 2]).toBe(200)
      }
    })

    it('should set composite operation', async () => {
      const manager = useCanvasManager()

      const origImage = createMockImage(100, 100)
      const maskImage = createMockImage(100, 100)

      await manager.invalidateCanvas(origImage, maskImage, null)

      expect(mockStore.maskCtx.globalCompositeOperation).toBe('source-over')
    })
  })
})
