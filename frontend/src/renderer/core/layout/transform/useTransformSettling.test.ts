import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, ref } from 'vue'

import { useTransformSettling } from '@/renderer/core/layout/transform/useTransformSettling'

describe('useTransformSettling', () => {
  let element: HTMLDivElement

  beforeEach(() => {
    vi.useFakeTimers()
    element = document.createElement('div')
    document.body.appendChild(element)
  })

  afterEach(() => {
    vi.useRealTimers()
    document.body.removeChild(element)
  })

  it('should track wheel events and settle after delay', async () => {
    const { isTransforming } = useTransformSettling(element, {
      settleDelay: 200
    })

    // Initially not transforming
    expect(isTransforming.value).toBe(false)

    // Dispatch wheel event
    element.dispatchEvent(new WheelEvent('wheel', { bubbles: true }))
    await nextTick()

    // Should be transforming
    expect(isTransforming.value).toBe(true)

    // Advance time but not past settle delay
    vi.advanceTimersByTime(100)
    expect(isTransforming.value).toBe(true)

    // Advance past settle delay
    vi.advanceTimersByTime(150)
    expect(isTransforming.value).toBe(false)
  })

  it('should reset settle timer on subsequent wheel events', async () => {
    const { isTransforming } = useTransformSettling(element, {
      settleDelay: 300
    })

    // First wheel event
    element.dispatchEvent(new WheelEvent('wheel', { bubbles: true }))
    await nextTick()
    expect(isTransforming.value).toBe(true)

    // Advance time partially
    vi.advanceTimersByTime(200)
    expect(isTransforming.value).toBe(true)

    // Another wheel event should reset the timer
    element.dispatchEvent(new WheelEvent('wheel', { bubbles: true }))
    await nextTick()

    // Advance 200ms more - should still be transforming
    vi.advanceTimersByTime(200)
    expect(isTransforming.value).toBe(true)

    // Need another 100ms to settle (300ms total from last event)
    vi.advanceTimersByTime(100)
    expect(isTransforming.value).toBe(false)
  })

  it('should not track pan events', async () => {
    const { isTransforming } = useTransformSettling(element)

    // Pointer events should not trigger transform
    element.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
    element.dispatchEvent(new PointerEvent('pointermove', { bubbles: true }))
    await nextTick()

    expect(isTransforming.value).toBe(false)
  })

  it('should work with ref target', async () => {
    const targetRef = ref<HTMLElement | null>(null)
    const { isTransforming } = useTransformSettling(targetRef, {
      settleDelay: 200
    })

    // No target yet
    expect(isTransforming.value).toBe(false)

    // Set target
    targetRef.value = element
    await nextTick()

    // Now events should work
    element.dispatchEvent(new WheelEvent('wheel', { bubbles: true }))
    await nextTick()
    expect(isTransforming.value).toBe(true)

    vi.advanceTimersByTime(200)
    expect(isTransforming.value).toBe(false)
  })

  it('should use capture phase for events', async () => {
    const captureHandler = vi.fn()
    const bubbleHandler = vi.fn()

    // Add handlers to verify capture phase
    element.addEventListener('wheel', captureHandler, true)
    element.addEventListener('wheel', bubbleHandler, false)

    const { isTransforming } = useTransformSettling(element)

    // Create child element
    const child = document.createElement('div')
    element.appendChild(child)

    // Dispatch event on child
    child.dispatchEvent(new WheelEvent('wheel', { bubbles: true }))
    await nextTick()

    // Capture handler should be called before bubble handler
    expect(captureHandler).toHaveBeenCalled()
    expect(isTransforming.value).toBe(true)

    element.removeEventListener('wheel', captureHandler, true)
    element.removeEventListener('wheel', bubbleHandler, false)
  })

  it('should clean up event listeners when component unmounts', async () => {
    const removeEventListenerSpy = vi.spyOn(element, 'removeEventListener')

    // Create a test component
    const TestComponent = {
      setup() {
        const { isTransforming } = useTransformSettling(element)
        return { isTransforming }
      },
      template: '<div>{{ isTransforming }}</div>'
    }

    const wrapper = mount(TestComponent)
    await nextTick()

    // Unmount component
    wrapper.unmount()

    // Should have removed wheel event listener
    expect(removeEventListenerSpy).toHaveBeenCalledWith(
      'wheel',
      expect.any(Function),
      expect.objectContaining({ capture: true })
    )
  })

  it('should use passive listeners when specified', async () => {
    const addEventListenerSpy = vi.spyOn(element, 'addEventListener')

    useTransformSettling(element, {
      passive: true
    })

    // Check that passive option was used for wheel event
    expect(addEventListenerSpy).toHaveBeenCalledWith(
      'wheel',
      expect.any(Function),
      expect.objectContaining({ passive: true, capture: true })
    )
  })
})
