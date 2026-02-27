// @ts-expect-error Polyfill
Symbol.dispose ??= Symbol('Symbol.dispose')
// @ts-expect-error Polyfill
Symbol.asyncDispose ??= Symbol('Symbol.asyncDispose')

// API *************************************************
// like rect but rounded corners
export function loadPolyfills() {
  if (
    typeof window != 'undefined' &&
    window.CanvasRenderingContext2D &&
    !window.CanvasRenderingContext2D.prototype.roundRect
  ) {
    // @ts-expect-error Slightly broken polyfill - radius_low not impl. anywhere
    window.CanvasRenderingContext2D.prototype.roundRect = function (
      x: number,
      y: number,
      w: number,
      h: number,
      radius: number | number[],
      radius_low: number | number[]
    ) {
      let top_left_radius = 0
      let top_right_radius = 0
      let bottom_left_radius = 0
      let bottom_right_radius = 0

      if (radius === 0) {
        this.rect(x, y, w, h)
        return
      }

      if (radius_low === undefined) radius_low = radius

      // make it compatible with official one
      if (Array.isArray(radius)) {
        if (radius.length == 1) {
          top_left_radius =
            top_right_radius =
            bottom_left_radius =
            bottom_right_radius =
              radius[0]
        } else if (radius.length == 2) {
          top_left_radius = bottom_right_radius = radius[0]
          top_right_radius = bottom_left_radius = radius[1]
        } else if (radius.length == 4) {
          top_left_radius = radius[0]
          top_right_radius = radius[1]
          bottom_left_radius = radius[2]
          bottom_right_radius = radius[3]
        } else {
          return
        }
      } else {
        // old using numbers
        top_left_radius = radius || 0
        top_right_radius = radius || 0

        const low = !Array.isArray(radius_low) && radius_low ? radius_low : 0
        bottom_left_radius = low
        bottom_right_radius = low
      }

      // top right
      this.moveTo(x + top_left_radius, y)
      this.lineTo(x + w - top_right_radius, y)
      this.quadraticCurveTo(x + w, y, x + w, y + top_right_radius)

      // bottom right
      this.lineTo(x + w, y + h - bottom_right_radius)
      this.quadraticCurveTo(x + w, y + h, x + w - bottom_right_radius, y + h)

      // bottom left
      this.lineTo(x + bottom_right_radius, y + h)
      this.quadraticCurveTo(x, y + h, x, y + h - bottom_left_radius)

      // top left
      this.lineTo(x, y + bottom_left_radius)
      this.quadraticCurveTo(x, y, x + top_left_radius, y)
    }
  }

  if (typeof window != 'undefined' && !window['requestAnimationFrame']) {
    window.requestAnimationFrame =
      // @ts-expect-error Legacy code
      window.webkitRequestAnimationFrame ||
      // @ts-expect-error Legacy code
      window.mozRequestAnimationFrame ||
      function (callback) {
        window.setTimeout(callback, 1000 / 60)
      }
  }
}
