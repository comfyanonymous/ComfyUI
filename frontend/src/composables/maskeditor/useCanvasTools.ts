import { ref, watch } from 'vue'
import { useMaskEditorStore } from '@/stores/maskEditorStore'
import { ColorComparisonMethod } from '@/extensions/core/maskeditor/types'
import type { Point } from '@/extensions/core/maskeditor/types'

const getPixelAlpha = (
  data: Uint8ClampedArray,
  x: number,
  y: number,
  width: number
): number => {
  return data[(y * width + x) * 4 + 3]
}

const getPixelColor = (
  data: Uint8ClampedArray,
  x: number,
  y: number,
  width: number
): { r: number; g: number; b: number } => {
  const index = (y * width + x) * 4
  return {
    r: data[index],
    g: data[index + 1],
    b: data[index + 2]
  }
}

const setPixel = (
  data: Uint8ClampedArray,
  x: number,
  y: number,
  width: number,
  alpha: number,
  color: { r: number; g: number; b: number }
): void => {
  const index = (y * width + x) * 4
  data[index] = color.r
  data[index + 1] = color.g
  data[index + 2] = color.b
  data[index + 3] = alpha
}

// Color comparison utilities
const rgbToHSL = (
  r: number,
  g: number,
  b: number
): { h: number; s: number; l: number } => {
  r /= 255
  g /= 255
  b /= 255

  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  let h = 0
  let s = 0
  const l = (max + min) / 2

  if (max !== min) {
    const d = max - min
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)

    switch (max) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0)
        break
      case g:
        h = (b - r) / d + 2
        break
      case b:
        h = (r - g) / d + 4
        break
    }
    h /= 6
  }

  return {
    h: h * 360,
    s: s * 100,
    l: l * 100
  }
}

const rgbToLab = (rgb: {
  r: number
  g: number
  b: number
}): {
  l: number
  a: number
  b: number
} => {
  let r = rgb.r / 255
  let g = rgb.g / 255
  let b = rgb.b / 255

  r = r > 0.04045 ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92
  g = g > 0.04045 ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92
  b = b > 0.04045 ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92

  r *= 100
  g *= 100
  b *= 100

  const x = r * 0.4124 + g * 0.3576 + b * 0.1805
  const y = r * 0.2126 + g * 0.7152 + b * 0.0722
  const z = r * 0.0193 + g * 0.1192 + b * 0.9505

  const xn = 95.047
  const yn = 100.0
  const zn = 108.883

  const xyz = [x / xn, y / yn, z / zn]
  for (let i = 0; i < xyz.length; i++) {
    xyz[i] =
      xyz[i] > 0.008856 ? Math.pow(xyz[i], 1 / 3) : 7.787 * xyz[i] + 16 / 116
  }

  return {
    l: 116 * xyz[1] - 16,
    a: 500 * (xyz[0] - xyz[1]),
    b: 200 * (xyz[1] - xyz[2])
  }
}

const isPixelInRangeSimple = (
  pixel: { r: number; g: number; b: number },
  target: { r: number; g: number; b: number },
  tolerance: number
): boolean => {
  const distance = Math.sqrt(
    Math.pow(pixel.r - target.r, 2) +
      Math.pow(pixel.g - target.g, 2) +
      Math.pow(pixel.b - target.b, 2)
  )
  return distance <= tolerance
}

const isPixelInRangeHSL = (
  pixel: { r: number; g: number; b: number },
  target: { r: number; g: number; b: number },
  tolerance: number
): boolean => {
  const pixelHSL = rgbToHSL(pixel.r, pixel.g, pixel.b)
  const targetHSL = rgbToHSL(target.r, target.g, target.b)

  const hueDiff = Math.abs(pixelHSL.h - targetHSL.h)
  const satDiff = Math.abs(pixelHSL.s - targetHSL.s)
  const lightDiff = Math.abs(pixelHSL.l - targetHSL.l)

  const distance = Math.sqrt(
    Math.pow((hueDiff / 360) * 255, 2) +
      Math.pow((satDiff / 100) * 255, 2) +
      Math.pow((lightDiff / 100) * 255, 2)
  )
  return distance <= tolerance
}

const isPixelInRangeLab = (
  pixel: { r: number; g: number; b: number },
  target: { r: number; g: number; b: number },
  tolerance: number
): boolean => {
  const pixelLab = rgbToLab(pixel)
  const targetLab = rgbToLab(target)

  const deltaE = Math.sqrt(
    Math.pow(pixelLab.l - targetLab.l, 2) +
      Math.pow(pixelLab.a - targetLab.a, 2) +
      Math.pow(pixelLab.b - targetLab.b, 2)
  )

  const normalizedDeltaE = (deltaE / 100) * 255
  return normalizedDeltaE <= tolerance
}

const isPixelInRange = (
  pixel: { r: number; g: number; b: number },
  target: { r: number; g: number; b: number },
  tolerance: number,
  method: ColorComparisonMethod
): boolean => {
  switch (method) {
    case ColorComparisonMethod.Simple:
      return isPixelInRangeSimple(pixel, target, tolerance)
    case ColorComparisonMethod.HSL:
      return isPixelInRangeHSL(pixel, target, tolerance)
    case ColorComparisonMethod.LAB:
      return isPixelInRangeLab(pixel, target, tolerance)
    default:
      return isPixelInRangeSimple(pixel, target, tolerance)
  }
}

export function useCanvasTools() {
  const store = useMaskEditorStore()
  const lastColorSelectPoint = ref<Point | null>(null)

  const paintBucketFill = (point: Point): void => {
    const ctx = store.maskCtx
    const canvas = store.maskCanvas
    if (!ctx || !canvas) return

    const startX = Math.floor(point.x)
    const startY = Math.floor(point.y)
    const width = canvas.width
    const height = canvas.height

    if (startX < 0 || startX >= width || startY < 0 || startY >= height) {
      return
    }

    const imageData = ctx.getImageData(0, 0, width, height)
    const data = imageData.data

    const targetAlpha = getPixelAlpha(data, startX, startY, width)
    const isFillMode = targetAlpha !== 255

    if (targetAlpha === -1) return

    const maskColor = store.maskColor
    const tolerance = store.paintBucketTolerance
    const fillOpacity = Math.floor((store.fillOpacity / 100) * 255)

    const stack: Array<[number, number]> = []
    const visited = new Uint8Array(width * height)

    const shouldProcessPixel = (
      currentAlpha: number,
      targetAlpha: number,
      tolerance: number,
      isFillMode: boolean
    ): boolean => {
      if (currentAlpha === -1) return false

      if (isFillMode) {
        return (
          currentAlpha !== 255 &&
          Math.abs(currentAlpha - targetAlpha) <= tolerance
        )
      } else {
        return (
          currentAlpha === 255 ||
          Math.abs(currentAlpha - targetAlpha) <= tolerance
        )
      }
    }

    if (shouldProcessPixel(targetAlpha, targetAlpha, tolerance, isFillMode)) {
      stack.push([startX, startY])
    }

    while (stack.length > 0) {
      const [x, y] = stack.pop()!
      const visitedIndex = y * width + x

      if (visited[visitedIndex]) continue

      const currentAlpha = getPixelAlpha(data, x, y, width)
      if (
        !shouldProcessPixel(currentAlpha, targetAlpha, tolerance, isFillMode)
      ) {
        continue
      }

      visited[visitedIndex] = 1
      setPixel(data, x, y, width, isFillMode ? fillOpacity : 0, maskColor)

      const checkNeighbor = (nx: number, ny: number) => {
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) return
        if (!visited[ny * width + nx]) {
          const alpha = getPixelAlpha(data, nx, ny, width)
          if (shouldProcessPixel(alpha, targetAlpha, tolerance, isFillMode)) {
            stack.push([nx, ny])
          }
        }
      }

      checkNeighbor(x - 1, y)
      checkNeighbor(x + 1, y)
      checkNeighbor(x, y - 1)
      checkNeighbor(x, y + 1)
    }

    ctx.putImageData(imageData, 0, 0)
    store.canvasHistory.saveState()
  }

  const colorSelectFill = async (point: Point): Promise<void> => {
    const maskCtx = store.maskCtx
    const imgCtx = store.imgCtx
    const imgCanvas = store.imgCanvas

    if (!maskCtx || !imgCtx || !imgCanvas) return

    const width = imgCanvas.width
    const height = imgCanvas.height
    lastColorSelectPoint.value = point

    const maskData = maskCtx.getImageData(0, 0, width, height)
    const maskDataArray = maskData.data
    const imageDataArray = imgCtx.getImageData(0, 0, width, height).data

    const tolerance = store.colorSelectTolerance
    const method = store.colorComparisonMethod
    const maskColor = store.maskColor
    const selectOpacity = Math.floor((store.selectionOpacity / 100) * 255)
    const applyWholeImage = store.applyWholeImage
    const maskBoundary = store.maskBoundary
    const maskTolerance = store.maskTolerance

    if (applyWholeImage) {
      const targetPixel = getPixelColor(
        imageDataArray,
        Math.floor(point.x),
        Math.floor(point.y),
        width
      )

      const CHUNK_SIZE = 10000
      for (let i = 0; i < width * height; i += CHUNK_SIZE) {
        const endIndex = Math.min(i + CHUNK_SIZE, width * height)
        for (let pixelIndex = i; pixelIndex < endIndex; pixelIndex++) {
          const x = pixelIndex % width
          const y = Math.floor(pixelIndex / width)
          if (
            isPixelInRange(
              getPixelColor(imageDataArray, x, y, width),
              targetPixel,
              tolerance,
              method
            )
          ) {
            setPixel(maskDataArray, x, y, width, selectOpacity, maskColor)
          }
        }
        await new Promise((resolve) => setTimeout(resolve, 0))
      }
    } else {
      const startX = Math.floor(point.x)
      const startY = Math.floor(point.y)

      if (startX < 0 || startX >= width || startY < 0 || startY >= height) {
        return
      }

      const targetPixel = getPixelColor(imageDataArray, startX, startY, width)
      const stack: Array<[number, number]> = []
      const visited = new Uint8Array(width * height)

      stack.push([startX, startY])

      while (stack.length > 0) {
        const [x, y] = stack.pop()!
        const visitedIndex = y * width + x

        if (
          visited[visitedIndex] ||
          !isPixelInRange(
            getPixelColor(imageDataArray, x, y, width),
            targetPixel,
            tolerance,
            method
          )
        ) {
          continue
        }

        visited[visitedIndex] = 1
        setPixel(maskDataArray, x, y, width, selectOpacity, maskColor)

        const checkNeighbor = (nx: number, ny: number) => {
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) return
          if (visited[ny * width + nx]) return
          if (
            !isPixelInRange(
              getPixelColor(imageDataArray, nx, ny, width),
              targetPixel,
              tolerance,
              method
            )
          )
            return
          if (
            maskBoundary &&
            255 - getPixelAlpha(maskDataArray, nx, ny, width) <= maskTolerance
          )
            return

          stack.push([nx, ny])
        }

        checkNeighbor(x - 1, y)
        checkNeighbor(x + 1, y)
        checkNeighbor(x, y - 1)
        checkNeighbor(x, y + 1)
      }
    }

    maskCtx.putImageData(maskData, 0, 0)
    store.canvasHistory.saveState()
  }

  const invertMask = (): void => {
    const ctx = store.maskCtx
    const canvas = store.maskCanvas
    if (!ctx || !canvas) return

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const data = imageData.data

    let maskR = 0,
      maskG = 0,
      maskB = 0
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] > 0) {
        maskR = data[i]
        maskG = data[i + 1]
        maskB = data[i + 2]
        break
      }
    }

    for (let i = 0; i < data.length; i += 4) {
      const alpha = data[i + 3]
      data[i + 3] = 255 - alpha

      if (alpha === 0) {
        data[i] = maskR
        data[i + 1] = maskG
        data[i + 2] = maskB
      }
    }

    ctx.putImageData(imageData, 0, 0)
    store.canvasHistory.saveState()
  }

  const clearMask = (): void => {
    const maskCtx = store.maskCtx
    const maskCanvas = store.maskCanvas
    const rgbCtx = store.rgbCtx
    const rgbCanvas = store.rgbCanvas

    if (maskCtx && maskCanvas) {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
    }
    if (rgbCtx && rgbCanvas) {
      rgbCtx.clearRect(0, 0, rgbCanvas.width, rgbCanvas.height)
    }
    store.canvasHistory.saveState()
  }

  const clearLastColorSelectPoint = () => {
    lastColorSelectPoint.value = null
  }

  watch(
    [
      () => store.colorSelectTolerance,
      () => store.colorComparisonMethod,
      () => store.selectionOpacity
    ],
    async () => {
      if (
        lastColorSelectPoint.value &&
        store.colorSelectLivePreview &&
        store.canUndo
      ) {
        store.canvasHistory.undo()
        await colorSelectFill(lastColorSelectPoint.value)
      }
    }
  )

  return {
    paintBucketFill,

    colorSelectFill,
    clearLastColorSelectPoint,

    invertMask,
    clearMask
  }
}
