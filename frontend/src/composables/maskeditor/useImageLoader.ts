import { useMaskEditorStore } from '@/stores/maskEditorStore'
import { useMaskEditorDataStore } from '@/stores/maskEditorDataStore'
import { createSharedComposable } from '@vueuse/core'
import { useCanvasManager } from '@/composables/maskeditor/useCanvasManager'

function useImageLoaderInternal() {
  const store = useMaskEditorStore()
  const dataStore = useMaskEditorDataStore()
  const canvasManager = useCanvasManager()

  const loadImages = async (): Promise<HTMLImageElement> => {
    const inputData = dataStore.inputData

    if (!inputData) {
      throw new Error('No input data available in dataStore')
    }

    const { imgCanvas, maskCanvas, rgbCanvas, imgCtx, maskCtx } = store

    if (!imgCanvas || !maskCanvas || !rgbCanvas || !imgCtx || !maskCtx) {
      throw new Error('Canvas elements or contexts not available')
    }

    imgCtx.clearRect(0, 0, imgCanvas.width, imgCanvas.height)
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)

    const baseImage = inputData.baseLayer.image
    const maskImage = inputData.maskLayer.image
    const paintImage = inputData.paintLayer?.image

    maskCanvas.width = baseImage.width
    maskCanvas.height = baseImage.height
    rgbCanvas.width = baseImage.width
    rgbCanvas.height = baseImage.height

    store.image = baseImage

    await canvasManager.invalidateCanvas(
      baseImage,
      maskImage,
      paintImage || null
    )

    await canvasManager.updateMaskColor()

    return baseImage
  }

  return {
    loadImages
  }
}

export const useImageLoader = createSharedComposable(useImageLoaderInternal)
