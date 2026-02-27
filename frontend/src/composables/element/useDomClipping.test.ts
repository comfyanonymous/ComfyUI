import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'

import { useDomClipping } from './useDomClipping'

function createMockElement(rect: {
  left: number
  top: number
  width: number
  height: number
}): HTMLElement {
  return {
    getBoundingClientRect: vi.fn(
      () =>
        ({
          ...rect,
          x: rect.left,
          y: rect.top,
          right: rect.left + rect.width,
          bottom: rect.top + rect.height,
          toJSON: () => ({})
        }) as DOMRect
    )
  } as unknown as HTMLElement
}

function createMockCanvas(rect: {
  left: number
  top: number
  width: number
  height: number
}): HTMLCanvasElement {
  return {
    getBoundingClientRect: vi.fn(
      () =>
        ({
          ...rect,
          x: rect.left,
          y: rect.top,
          right: rect.left + rect.width,
          bottom: rect.top + rect.height,
          toJSON: () => ({})
        }) as DOMRect
    )
  } as unknown as HTMLCanvasElement
}

describe('useDomClipping', () => {
  let rafCallbacks: Map<number, FrameRequestCallback>
  let nextRafId: number

  beforeEach(() => {
    rafCallbacks = new Map()
    nextRafId = 1

    vi.stubGlobal(
      'requestAnimationFrame',
      vi.fn((cb: FrameRequestCallback) => {
        const id = nextRafId++
        rafCallbacks.set(id, cb)
        return id
      })
    )

    vi.stubGlobal(
      'cancelAnimationFrame',
      vi.fn((id: number) => {
        rafCallbacks.delete(id)
      })
    )
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  function flushRaf() {
    const callbacks = [...rafCallbacks.values()]
    rafCallbacks.clear()
    for (const cb of callbacks) {
      cb(performance.now())
    }
  }

  it('coalesces multiple rapid calls into a single getBoundingClientRect read', () => {
    const { updateClipPath } = useDomClipping()
    const element = createMockElement({
      left: 10,
      top: 10,
      width: 100,
      height: 50
    })
    const canvas = createMockCanvas({
      left: 0,
      top: 0,
      width: 800,
      height: 600
    })

    updateClipPath(element, canvas, true)
    updateClipPath(element, canvas, true)
    updateClipPath(element, canvas, true)

    expect(element.getBoundingClientRect).not.toHaveBeenCalled()

    flushRaf()

    expect(element.getBoundingClientRect).toHaveBeenCalledTimes(1)
    expect(canvas.getBoundingClientRect).toHaveBeenCalledTimes(1)
  })

  it('updates style ref after RAF fires', () => {
    const { style, updateClipPath } = useDomClipping()
    const element = createMockElement({
      left: 10,
      top: 10,
      width: 100,
      height: 50
    })
    const canvas = createMockCanvas({
      left: 0,
      top: 0,
      width: 800,
      height: 600
    })

    updateClipPath(element, canvas, true)

    expect(style.value).toEqual({})

    flushRaf()

    expect(style.value).toEqual({
      clipPath: 'none',
      willChange: 'clip-path'
    })
  })

  it('cancels previous RAF when called again before it fires', () => {
    const { style, updateClipPath } = useDomClipping()
    const element1 = createMockElement({
      left: 10,
      top: 10,
      width: 100,
      height: 50
    })
    const element2 = createMockElement({
      left: 20,
      top: 20,
      width: 200,
      height: 100
    })
    const canvas = createMockCanvas({
      left: 0,
      top: 0,
      width: 800,
      height: 600
    })

    updateClipPath(element1, canvas, true)
    updateClipPath(element2, canvas, true)

    expect(cancelAnimationFrame).toHaveBeenCalledTimes(1)

    flushRaf()

    expect(element1.getBoundingClientRect).not.toHaveBeenCalled()
    expect(element2.getBoundingClientRect).toHaveBeenCalledTimes(1)
    expect(style.value).toEqual({
      clipPath: 'none',
      willChange: 'clip-path'
    })
  })

  it('generates clip-path polygon when element intersects unselected area', () => {
    const { style, updateClipPath } = useDomClipping()
    const element = createMockElement({
      left: 50,
      top: 50,
      width: 100,
      height: 100
    })
    const canvas = createMockCanvas({
      left: 0,
      top: 0,
      width: 800,
      height: 600
    })
    const selectedArea = {
      x: 40,
      y: 40,
      width: 200,
      height: 200,
      scale: 1,
      offset: [0, 0] as [number, number]
    }

    updateClipPath(element, canvas, false, selectedArea)
    flushRaf()

    expect(style.value.clipPath).toContain('polygon')
    expect(style.value.willChange).toBe('clip-path')
  })

  it('does not read layout before RAF fires', () => {
    const { updateClipPath } = useDomClipping()
    const element = createMockElement({
      left: 0,
      top: 0,
      width: 50,
      height: 50
    })
    const canvas = createMockCanvas({
      left: 0,
      top: 0,
      width: 800,
      height: 600
    })

    updateClipPath(element, canvas, true)

    expect(element.getBoundingClientRect).not.toHaveBeenCalled()
    expect(canvas.getBoundingClientRect).not.toHaveBeenCalled()
  })
})
