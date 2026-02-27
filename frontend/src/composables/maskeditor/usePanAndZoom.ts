import { ref, watch } from 'vue'
import type { Offset, Point } from '@/extensions/core/maskeditor/types'
import { useMaskEditorStore } from '@/stores/maskEditorStore'

export function usePanAndZoom() {
  const store = useMaskEditorStore()

  const DOUBLE_TAP_DELAY = 300

  const lastTwoFingerTap = ref(0)
  const isTouchZooming = ref(false)
  const lastTouchZoomDistance = ref(0)
  const lastTouchMidPoint = ref<Point>({ x: 0, y: 0 })
  const lastTouchPoint = ref<Point>({ x: 0, y: 0 })

  const zoom_ratio = ref(1)
  const interpolatedZoomRatio = ref(1)
  const pan_offset = ref<Offset>({ x: 0, y: 0 })

  const mouseDownPoint = ref<Point | null>(null)
  const initialPan = ref<Offset>({ x: 0, y: 0 })

  const canvasContainer = ref<HTMLElement | null>(null)
  const maskCanvas = ref<HTMLCanvasElement | null>(null)
  const rgbCanvas = ref<HTMLCanvasElement | null>(null)
  const rootElement = ref<HTMLElement | null>(null)

  const toolPanelElement = ref<HTMLElement | null>(null)
  const sidePanelElement = ref<HTMLElement | null>(null)

  const image = ref<HTMLImageElement | null>(null)
  const imageRootWidth = ref(0)
  const imageRootHeight = ref(0)

  const cursorPoint = ref<Point>({ x: 0, y: 0 })
  const penPointerIdList = ref<number[]>([])

  const getTouchDistance = (touches: TouchList): number => {
    const dx = touches[0].clientX - touches[1].clientX
    const dy = touches[0].clientY - touches[1].clientY
    return Math.sqrt(dx * dx + dy * dy)
  }

  const getTouchMidpoint = (touches: TouchList): Point => {
    return {
      x: (touches[0].clientX + touches[1].clientX) / 2,
      y: (touches[0].clientY + touches[1].clientY) / 2
    }
  }

  const updateCursorPosition = (clientPoint: Point): void => {
    const cursorX = clientPoint.x - pan_offset.value.x
    const cursorY = clientPoint.y - pan_offset.value.y
    cursorPoint.value = { x: cursorX, y: cursorY }
    store.setCursorPoint({ x: cursorX, y: cursorY })
  }

  const handleDoubleTap = (): void => {
    store.canvasHistory.undo()
  }

  const invalidatePanZoom = async (): Promise<void> => {
    // Single validation check upfront
    if (
      !image.value?.width ||
      !image.value?.height ||
      !pan_offset.value ||
      !zoom_ratio.value
    ) {
      console.warn('Missing required properties for pan/zoom')
      return
    }

    const raw_width = image.value.width * zoom_ratio.value
    const raw_height = image.value.height * zoom_ratio.value

    if (!canvasContainer.value) {
      canvasContainer.value = store.canvasContainer
    }
    if (!canvasContainer.value) return

    Object.assign(canvasContainer.value.style, {
      width: `${raw_width}px`,
      height: `${raw_height}px`,
      left: `${pan_offset.value.x}px`,
      top: `${pan_offset.value.y}px`
    })

    if (!rgbCanvas.value) {
      rgbCanvas.value = store.rgbCanvas
    }
    if (rgbCanvas.value) {
      if (
        rgbCanvas.value.width !== image.value.width ||
        rgbCanvas.value.height !== image.value.height
      ) {
        rgbCanvas.value.width = image.value.width
        rgbCanvas.value.height = image.value.height
      }

      rgbCanvas.value.style.width = `${raw_width}px`
      rgbCanvas.value.style.height = `${raw_height}px`
    }

    store.setPanOffset(pan_offset.value)
    store.setZoomRatio(zoom_ratio.value)
  }

  const handlePanStart = (event: PointerEvent): void => {
    mouseDownPoint.value = { x: event.clientX, y: event.clientY }

    store.isPanning = true
    initialPan.value = { ...pan_offset.value }
  }

  const handlePanMove = async (event: PointerEvent): Promise<void> => {
    if (mouseDownPoint.value === null) {
      throw new Error('mouseDownPoint is null')
    }

    const deltaX = mouseDownPoint.value.x - event.clientX
    const deltaY = mouseDownPoint.value.y - event.clientY

    const pan_x = initialPan.value.x - deltaX
    const pan_y = initialPan.value.y - deltaY

    pan_offset.value = { x: pan_x, y: pan_y }

    await invalidatePanZoom()
  }

  const handleSingleTouchPan = async (touch: Touch): Promise<void> => {
    if (lastTouchPoint.value === null) {
      lastTouchPoint.value = { x: touch.clientX, y: touch.clientY }
      return
    }

    const deltaX = touch.clientX - lastTouchPoint.value.x
    const deltaY = touch.clientY - lastTouchPoint.value.y

    pan_offset.value.x += deltaX
    pan_offset.value.y += deltaY

    await invalidatePanZoom()

    lastTouchPoint.value = { x: touch.clientX, y: touch.clientY }
  }

  const handleTouchStart = (event: TouchEvent): void => {
    event.preventDefault()

    if (penPointerIdList.value.length > 0) return

    store.brushVisible = false

    if (event.touches.length === 2) {
      const currentTime = new Date().getTime()
      const tapTimeDiff = currentTime - lastTwoFingerTap.value

      if (tapTimeDiff < DOUBLE_TAP_DELAY) {
        handleDoubleTap()
        lastTwoFingerTap.value = 0
      } else {
        lastTwoFingerTap.value = currentTime

        isTouchZooming.value = true
        lastTouchZoomDistance.value = getTouchDistance(event.touches)
        lastTouchMidPoint.value = getTouchMidpoint(event.touches)
      }
    } else if (event.touches.length === 1) {
      lastTouchPoint.value = {
        x: event.touches[0].clientX,
        y: event.touches[0].clientY
      }
    }
  }

  const handleTouchMove = async (event: TouchEvent): Promise<void> => {
    event.preventDefault()

    if (penPointerIdList.value.length > 0) return

    lastTwoFingerTap.value = 0

    if (isTouchZooming.value && event.touches.length === 2) {
      const newDistance = getTouchDistance(event.touches)
      const zoomFactor = newDistance / lastTouchZoomDistance.value
      const oldZoom = zoom_ratio.value
      zoom_ratio.value = Math.max(
        0.2,
        Math.min(10.0, zoom_ratio.value * zoomFactor)
      )
      const newZoom = zoom_ratio.value

      const midpoint = getTouchMidpoint(event.touches)

      if (lastTouchMidPoint.value) {
        const deltaX = midpoint.x - lastTouchMidPoint.value.x
        const deltaY = midpoint.y - lastTouchMidPoint.value.y

        pan_offset.value.x += deltaX
        pan_offset.value.y += deltaY
      }

      if (maskCanvas.value === null) {
        maskCanvas.value = store.maskCanvas
      }
      if (!maskCanvas.value) return

      const rect = maskCanvas.value.getBoundingClientRect()
      const touchX = midpoint.x - rect.left
      const touchY = midpoint.y - rect.top

      const scaleFactor = newZoom / oldZoom
      pan_offset.value.x += touchX - touchX * scaleFactor
      pan_offset.value.y += touchY - touchY * scaleFactor

      await invalidatePanZoom()
      lastTouchZoomDistance.value = newDistance
      lastTouchMidPoint.value = midpoint
    } else if (event.touches.length === 1) {
      await handleSingleTouchPan(event.touches[0])
    }
  }

  const handleTouchEnd = (event: TouchEvent): void => {
    event.preventDefault()

    const lastTouch = event.touches[0]

    if (lastTouch) {
      lastTouchPoint.value = {
        x: lastTouch.clientX,
        y: lastTouch.clientY
      }
    } else {
      isTouchZooming.value = false
      lastTouchMidPoint.value = { x: 0, y: 0 }
    }
  }

  const zoom = async (event: WheelEvent): Promise<void> => {
    const cursorPosition = { x: event.clientX, y: event.clientY }

    const oldZoom = zoom_ratio.value
    const zoomFactor = event.deltaY < 0 ? 1.1 : 0.9
    zoom_ratio.value = Math.max(
      0.2,
      Math.min(10.0, zoom_ratio.value * zoomFactor)
    )
    const newZoom = zoom_ratio.value

    if (!maskCanvas.value) {
      maskCanvas.value = store.maskCanvas
    }
    if (!maskCanvas.value) return

    const rect = maskCanvas.value.getBoundingClientRect()
    const mouseX = cursorPosition.x - rect.left
    const mouseY = cursorPosition.y - rect.top

    const scaleFactor = newZoom / oldZoom
    pan_offset.value.x += mouseX - mouseX * scaleFactor
    pan_offset.value.y += mouseY - mouseY * scaleFactor

    await invalidatePanZoom()

    const newImageWidth = maskCanvas.value.clientWidth

    const zoomRatio = newImageWidth / imageRootWidth.value

    interpolatedZoomRatio.value = zoomRatio
    store.displayZoomRatio = zoomRatio

    updateCursorPosition(cursorPosition)
  }

  const getPanelDimensions = (): {
    sidePanelWidth: number
    toolPanelWidth: number
  } => {
    const toolPanelWidth =
      toolPanelElement.value?.getBoundingClientRect().width || 64
    const sidePanelWidth =
      sidePanelElement.value?.getBoundingClientRect().width || 220

    return { sidePanelWidth, toolPanelWidth }
  }

  const smoothResetView = async (duration: number = 500): Promise<void> => {
    if (!image.value || !rootElement.value) return

    const startZoom = zoom_ratio.value
    const startPan = { ...pan_offset.value }

    const { sidePanelWidth, toolPanelWidth } = getPanelDimensions()

    const availableWidth =
      rootElement.value.clientWidth - sidePanelWidth - toolPanelWidth
    const availableHeight = rootElement.value.clientHeight

    // Calculate target zoom
    const zoomRatioWidth = availableWidth / image.value.width
    const zoomRatioHeight = availableHeight / image.value.height
    const targetZoom = Math.min(zoomRatioWidth, zoomRatioHeight)

    const aspectRatio = image.value.width / image.value.height
    let finalWidth: number
    let finalHeight: number

    const targetPan = { x: toolPanelWidth, y: 0 }

    if (zoomRatioHeight > zoomRatioWidth) {
      finalWidth = availableWidth
      finalHeight = finalWidth / aspectRatio
      targetPan.y = (availableHeight - finalHeight) / 2
    } else {
      finalHeight = availableHeight
      finalWidth = finalHeight * aspectRatio
      targetPan.x = (availableWidth - finalWidth) / 2 + toolPanelWidth
    }

    const startTime = performance.now()
    const animate = async (currentTime: number) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)

      zoom_ratio.value = startZoom + (targetZoom - startZoom) * eased
      pan_offset.value.x = startPan.x + (targetPan.x - startPan.x) * eased
      pan_offset.value.y = startPan.y + (targetPan.y - startPan.y) * eased

      await invalidatePanZoom()

      const interpolatedRatio = startZoom + (1.0 - startZoom) * eased
      store.displayZoomRatio = interpolatedRatio

      if (progress < 1) {
        requestAnimationFrame(animate)
      }
    }

    requestAnimationFrame(animate)
    interpolatedZoomRatio.value = 1.0
  }

  const initializeCanvasPanZoom = async (
    img: HTMLImageElement,
    root: HTMLElement,
    toolPanel?: HTMLElement | null,
    sidePanel?: HTMLElement | null
  ): Promise<void> => {
    rootElement.value = root
    toolPanelElement.value = toolPanel || null
    sidePanelElement.value = sidePanel || null

    const { sidePanelWidth, toolPanelWidth } = getPanelDimensions()

    const availableWidth = root.clientWidth - sidePanelWidth - toolPanelWidth
    const availableHeight = root.clientHeight

    const zoomRatioWidth = availableWidth / img.width
    const zoomRatioHeight = availableHeight / img.height

    const aspectRatio = img.width / img.height

    let finalWidth: number
    let finalHeight: number

    const panOffset: Offset = { x: toolPanelWidth, y: 0 }

    if (zoomRatioHeight > zoomRatioWidth) {
      finalWidth = availableWidth
      finalHeight = finalWidth / aspectRatio
      panOffset.y = (availableHeight - finalHeight) / 2
    } else {
      finalHeight = availableHeight
      finalWidth = finalHeight * aspectRatio
      panOffset.x = (availableWidth - finalWidth) / 2 + toolPanelWidth
    }

    if (image.value === null) {
      image.value = img
    }

    imageRootWidth.value = finalWidth
    imageRootHeight.value = finalHeight

    zoom_ratio.value = Math.min(zoomRatioWidth, zoomRatioHeight)
    pan_offset.value = panOffset

    penPointerIdList.value = []

    await invalidatePanZoom()
  }

  watch(
    () => store.resetZoomTrigger,
    async () => {
      if (interpolatedZoomRatio.value === 1) return
      await smoothResetView()
    }
  )

  const addPenPointerId = (pointerId: number): void => {
    if (!penPointerIdList.value.includes(pointerId)) {
      penPointerIdList.value.push(pointerId)
    }
  }

  const removePenPointerId = (pointerId: number): void => {
    const index = penPointerIdList.value.indexOf(pointerId)
    if (index !== -1) {
      penPointerIdList.value.splice(index, 1)
    }
  }

  return {
    initializeCanvasPanZoom,
    handlePanStart,
    handlePanMove,
    handleTouchStart,
    handleTouchMove,
    handleTouchEnd,
    updateCursorPosition,
    zoom,
    invalidatePanZoom,
    addPenPointerId,
    removePenPointerId
  }
}
