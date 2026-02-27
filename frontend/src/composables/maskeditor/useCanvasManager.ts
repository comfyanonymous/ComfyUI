import { useMaskEditorStore } from '@/stores/maskEditorStore'

import { MaskBlendMode } from '@/extensions/core/maskeditor/types'

export function useCanvasManager() {
  const store = useMaskEditorStore()

  const prepareMask = async (
    image: HTMLImageElement,
    maskCanvasEl: HTMLCanvasElement,
    maskContext: CanvasRenderingContext2D
  ): Promise<void> => {
    const maskColor = store.maskColor

    maskContext.drawImage(image, 0, 0, maskCanvasEl.width, maskCanvasEl.height)

    const maskData = maskContext.getImageData(
      0,
      0,
      maskCanvasEl.width,
      maskCanvasEl.height
    )

    for (let i = 0; i < maskData.data.length; i += 4) {
      const alpha = maskData.data[i + 3]
      maskData.data[i] = maskColor.r
      maskData.data[i + 1] = maskColor.g
      maskData.data[i + 2] = maskColor.b
      maskData.data[i + 3] = 255 - alpha
    }

    maskContext.globalCompositeOperation = 'source-over'
    maskContext.putImageData(maskData, 0, 0)
  }

  const invalidateCanvas = async (
    origImage: HTMLImageElement,
    maskImage: HTMLImageElement,
    paintImage: HTMLImageElement | null
  ): Promise<void> => {
    const { imgCanvas, maskCanvas, rgbCanvas, imgCtx, maskCtx, rgbCtx } = store

    if (
      !imgCanvas ||
      !maskCanvas ||
      !rgbCanvas ||
      !imgCtx ||
      !maskCtx ||
      !rgbCtx
    ) {
      throw new Error('Canvas elements or contexts not available')
    }

    imgCanvas.width = origImage.width
    imgCanvas.height = origImage.height
    maskCanvas.width = origImage.width
    maskCanvas.height = origImage.height
    rgbCanvas.width = origImage.width
    rgbCanvas.height = origImage.height

    imgCtx.drawImage(origImage, 0, 0, origImage.width, origImage.height)

    if (paintImage) {
      rgbCtx.drawImage(paintImage, 0, 0, paintImage.width, paintImage.height)
    }

    await prepareMask(maskImage, maskCanvas, maskCtx)
  }

  const setCanvasBackground = (): void => {
    const canvasBackground = store.canvasBackground

    if (!canvasBackground) return

    if (store.maskBlendMode === MaskBlendMode.Black) {
      canvasBackground.style.backgroundColor = 'rgba(0,0,0,1)'
    } else if (store.maskBlendMode === MaskBlendMode.White) {
      canvasBackground.style.backgroundColor = 'rgba(255,255,255,1)'
    } else if (store.maskBlendMode === MaskBlendMode.Negative) {
      canvasBackground.style.backgroundColor = 'rgba(255,255,255,1)'
    }
  }

  const updateMaskColor = async (): Promise<void> => {
    const { maskCanvas, maskCtx, maskColor, maskBlendMode, maskOpacity } = store

    if (!maskCanvas || !maskCtx) return

    if (maskBlendMode === MaskBlendMode.Negative) {
      maskCanvas.style.mixBlendMode = 'difference'
      maskCanvas.style.opacity = '1'
    } else {
      maskCanvas.style.mixBlendMode = 'initial'
      maskCanvas.style.opacity = String(maskOpacity)
    }

    maskCtx.fillStyle = `rgb(${maskColor.r}, ${maskColor.g}, ${maskColor.b})`

    setCanvasBackground()

    const maskData = maskCtx.getImageData(
      0,
      0,
      maskCanvas.width,
      maskCanvas.height
    )

    for (let i = 0; i < maskData.data.length; i += 4) {
      maskData.data[i] = maskColor.r
      maskData.data[i + 1] = maskColor.g
      maskData.data[i + 2] = maskColor.b
    }

    maskCtx.putImageData(maskData, 0, 0)
  }

  return {
    invalidateCanvas,
    updateMaskColor
  }
}
