import { ref, computed } from 'vue'
import { useMaskEditorStore } from '@/stores/maskEditorStore'

// Define the state interface for better readability
interface CanvasState {
  mask: ImageData | ImageBitmap
  rgb: ImageData | ImageBitmap
  img: ImageData | ImageBitmap
}

export function useCanvasHistory(maxStates = 20) {
  const store = useMaskEditorStore()

  const states = ref<CanvasState[]>([])
  const currentStateIndex = ref(-1)
  const initialized = ref(false)

  const canUndo = computed(
    () => states.value.length > 1 && currentStateIndex.value > 0
  )

  const canRedo = computed(() => {
    return (
      states.value.length > 1 &&
      currentStateIndex.value < states.value.length - 1
    )
  })

  const saveInitialState = () => {
    const { maskCtx, rgbCtx, imgCtx, maskCanvas, rgbCanvas, imgCanvas } = store

    // Ensure all 3 contexts and canvases are ready
    if (
      !maskCtx ||
      !rgbCtx ||
      !imgCtx ||
      !maskCanvas ||
      !rgbCanvas ||
      !imgCanvas
    ) {
      requestAnimationFrame(saveInitialState)
      return
    }

    if (!maskCanvas.width || !rgbCanvas.width || !imgCanvas.width) {
      requestAnimationFrame(saveInitialState)
      return
    }

    states.value = []

    // Capture all three layers
    const maskState = maskCtx.getImageData(
      0,
      0,
      maskCanvas.width,
      maskCanvas.height
    )
    const rgbState = rgbCtx.getImageData(
      0,
      0,
      rgbCanvas.width,
      rgbCanvas.height
    )
    const imgState = imgCtx.getImageData(
      0,
      0,
      imgCanvas.width,
      imgCanvas.height
    )

    states.value.push({ mask: maskState, rgb: rgbState, img: imgState })
    currentStateIndex.value = 0
    initialized.value = true
  }

  const saveState = (
    providedMaskData?: ImageData | ImageBitmap,
    providedRgbData?: ImageData | ImageBitmap,
    providedImgData?: ImageData | ImageBitmap
  ) => {
    const { maskCtx, rgbCtx, imgCtx, maskCanvas, rgbCanvas, imgCanvas } = store

    if (
      !maskCtx ||
      !rgbCtx ||
      !imgCtx ||
      !maskCanvas ||
      !rgbCanvas ||
      !imgCanvas
    )
      return

    if (!initialized.value || currentStateIndex.value === -1) {
      saveInitialState()
      return
    }

    // Clear redo history
    states.value = states.value.slice(0, currentStateIndex.value + 1)

    let maskState: ImageData | ImageBitmap
    let rgbState: ImageData | ImageBitmap
    let imgState: ImageData | ImageBitmap

    if (providedMaskData && providedRgbData && providedImgData) {
      maskState = providedMaskData
      rgbState = providedRgbData
      imgState = providedImgData
    } else {
      maskState = maskCtx.getImageData(
        0,
        0,
        maskCanvas.width,
        maskCanvas.height
      )
      rgbState = rgbCtx.getImageData(0, 0, rgbCanvas.width, rgbCanvas.height)
      imgState = imgCtx.getImageData(0, 0, imgCanvas.width, imgCanvas.height)
    }

    states.value.push({ mask: maskState, rgb: rgbState, img: imgState })
    currentStateIndex.value++

    // Maintain max history size and clean up memory
    if (states.value.length > maxStates) {
      const removed = states.value.shift()
      if (removed) {
        cleanupState(removed)
      }
      currentStateIndex.value--
    }
  }

  const undo = () => {
    if (!canUndo.value) return
    currentStateIndex.value--
    restoreState(states.value[currentStateIndex.value])
  }

  const redo = () => {
    if (!canRedo.value) return
    currentStateIndex.value++
    restoreState(states.value[currentStateIndex.value])
  }

  const restoreState = (state: CanvasState) => {
    const { maskCtx, rgbCtx, imgCtx, maskCanvas, rgbCanvas, imgCanvas } = store
    if (
      !maskCtx ||
      !rgbCtx ||
      !imgCtx ||
      !maskCanvas ||
      !rgbCanvas ||
      !imgCanvas
    )
      return

    // Update canvas dimensions to match state (handles rotation undo/redo)
    const refData = state.mask
    const newWidth = refData.width
    const newHeight = refData.height

    if (maskCanvas.width !== newWidth || maskCanvas.height !== newHeight) {
      maskCanvas.width = newWidth
      maskCanvas.height = newHeight
      rgbCanvas.width = newWidth
      rgbCanvas.height = newHeight
      imgCanvas.width = newWidth
      imgCanvas.height = newHeight
    }

    const layers = [
      { ctx: maskCtx, data: state.mask },
      { ctx: rgbCtx, data: state.rgb },
      { ctx: imgCtx, data: state.img }
    ]

    layers.forEach(({ ctx, data }) => {
      if (data instanceof ImageBitmap) {
        ctx.clearRect(0, 0, data.width, data.height)
        ctx.drawImage(data, 0, 0)
      } else {
        ctx.putImageData(data, 0, 0)
      }
    })
  }

  const cleanupState = (state: CanvasState) => {
    if (state.mask instanceof ImageBitmap) state.mask.close()
    if (state.rgb instanceof ImageBitmap) state.rgb.close()
    if (state.img instanceof ImageBitmap) state.img.close()
  }

  const clearStates = () => {
    states.value.forEach(cleanupState)
    states.value = []
    currentStateIndex.value = -1
    initialized.value = false
  }

  return {
    canUndo,
    canRedo,
    currentStateIndex,
    saveInitialState,
    saveState,
    undo,
    redo,
    clearStates
  }
}
