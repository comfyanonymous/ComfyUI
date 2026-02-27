import { LGraphCanvas, LiteGraph } from '@/lib/litegraph/src/litegraph'

import { app } from '@/scripts/app'

let touchZooming = false
let touchCount = 0

app.registerExtension({
  name: 'Comfy.SimpleTouchSupport',
  setup() {
    let touchDist: number | null = null
    let touchTime: Date | null = null
    let lastTouch: { clientX: number; clientY: number } | null = null
    let lastScale: number | null = null
    function getMultiTouchPos(e: TouchEvent) {
      return Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      )
    }

    function getMultiTouchCenter(e: TouchEvent) {
      return {
        clientX: (e.touches[0].clientX + e.touches[1].clientX) / 2,
        clientY: (e.touches[0].clientY + e.touches[1].clientY) / 2
      }
    }

    app.canvasEl.parentElement?.addEventListener(
      'touchstart',
      (e: TouchEvent) => {
        touchCount += e.changedTouches.length

        lastTouch = null
        lastScale = null
        if (e.touches?.length === 1) {
          // Store start time for press+hold for context menu
          touchTime = new Date()
          lastTouch = e.touches[0]
        } else {
          touchTime = null
          if (e.touches?.length === 2) {
            // Store center pos for zoom
            lastScale = app.canvas.ds.scale
            lastTouch = getMultiTouchCenter(e)

            touchDist = getMultiTouchPos(e)
            app.canvas.pointer.isDown = false
          }
        }
      },
      true
    )

    app.canvasEl.parentElement?.addEventListener(
      'touchend',
      (e: TouchEvent) => {
        touchCount -= e.changedTouches.length

        if (e.touches?.length !== 1) touchZooming = false
        if (touchTime && !e.touches?.length) {
          if (new Date().getTime() - touchTime.getTime() > 600) {
            if (e.target === app.canvasEl) {
              const touch = {
                button: 2, // Right click
                clientX: e.changedTouches[0].clientX,
                clientY: e.changedTouches[0].clientY,
                pointerId: 1, // changedTouches' id is 0, set it to any number
                isPrimary: true // changedTouches' isPrimary is false, so set it to true
              }
              // context menu info set in 'pointerdown' event
              app.canvasEl.dispatchEvent(new PointerEvent('pointerdown', touch))
              // then, context menu open after 'pointerup' event
              setTimeout(() => {
                app.canvasEl.dispatchEvent(new PointerEvent('pointerup', touch))
              })
              e.preventDefault()
            }
          }
          touchTime = null
        }
      }
    )

    const resetTouchState = () => {
      touchCount = 0
      touchZooming = false
      touchTime = null
      lastTouch = null
      lastScale = null
      touchDist = null
    }

    // Reset touch state when page loses visibility (e.g., switching apps on iPad)
    // This prevents touchCount from getting stuck when touchend events are missed
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        resetTouchState()
      }
    })

    // Also handle touchcancel which fires when touch is interrupted
    app.canvasEl.parentElement?.addEventListener('touchcancel', resetTouchState)

    app.canvasEl.parentElement?.addEventListener(
      'touchmove',
      (e) => {
        // make a threshold for touchmove to prevent clear touchTime for long press
        if (touchTime && lastTouch && e.touches?.length === 1) {
          const onlyTouch = e.touches[0]
          const deltaX = onlyTouch.clientX - lastTouch.clientX
          const deltaY = onlyTouch.clientY - lastTouch.clientY
          if (deltaX * deltaX + deltaY * deltaY > 30) {
            touchTime = null
          }
        }
        if (e.touches?.length === 2 && lastTouch && !e.ctrlKey && !e.shiftKey) {
          e.preventDefault() // Prevent browser from zooming when two textareas are touched
          app.canvas.pointer.isDown = false
          touchZooming = true

          LiteGraph.closeAllContextMenus(window)
          // @ts-expect-error
          app.canvas.search_box?.close()
          const newTouchDist = getMultiTouchPos(e)

          const center = getMultiTouchCenter(e)

          if (lastScale === null || touchDist === null) return
          let scale = (lastScale * newTouchDist) / touchDist

          const newX = (center.clientX - lastTouch.clientX) / scale
          const newY = (center.clientY - lastTouch.clientY) / scale

          // Code from LiteGraph
          if (scale < app.canvas.ds.min_scale) {
            scale = app.canvas.ds.min_scale
          } else if (scale > app.canvas.ds.max_scale) {
            scale = app.canvas.ds.max_scale
          }

          const oldScale = app.canvas.ds.scale

          app.canvas.ds.scale = scale

          // Code from LiteGraph
          if (Math.abs(app.canvas.ds.scale - 1) < 0.01) {
            app.canvas.ds.scale = 1
          }

          const newScale = app.canvas.ds.scale

          const convertScaleToOffset = (scale: number) => [
            center.clientX / scale - app.canvas.ds.offset[0],
            center.clientY / scale - app.canvas.ds.offset[1]
          ]
          var oldCenter = convertScaleToOffset(oldScale)
          var newCenter = convertScaleToOffset(newScale)

          app.canvas.ds.offset[0] += newX + newCenter[0] - oldCenter[0]
          app.canvas.ds.offset[1] += newY + newCenter[1] - oldCenter[1]

          lastTouch.clientX = center.clientX
          lastTouch.clientY = center.clientY

          app.canvas.setDirty(true, true)
        }
      },
      true
    )
  }
})

const processMouseDown = LGraphCanvas.prototype.processMouseDown
LGraphCanvas.prototype.processMouseDown = function (e: PointerEvent) {
  if (touchZooming || touchCount) {
    return
  }
  app.canvas.pointer.isDown = false // Prevent context menu from opening on second tap
  return processMouseDown.apply(this, [e])
}

const processMouseMove = LGraphCanvas.prototype.processMouseMove
LGraphCanvas.prototype.processMouseMove = function (e: PointerEvent) {
  if (touchZooming || touchCount > 1) {
    return
  }
  return processMouseMove.apply(this, [e])
}
