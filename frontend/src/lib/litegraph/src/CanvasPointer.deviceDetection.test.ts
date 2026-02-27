/**
 * Test-Driven Design (TDD) tests for device detection functionality.
 *
 * These tests describe the expected behavior for device detection between
 * mouse and trackpad inputs using an efficient timestamp-based approach.
 *
 * Design Philosophy:
 * - Uses timestamps (performance.now()) instead of creating timers for every event
 * - Creates at most ONE timer (for Linux buffer timeout), not one per wheel event
 * - Handles potentially thousands of wheel events per second efficiently
 *
 * Expected new properties on CanvasPointer:
 * - detectedDevice: 'mouse' | 'trackpad'
 * - lastWheelEventTime: number  // timestamp, not the event itself
 * - bufferedLinuxEvent: WheelEvent | undefined
 * - bufferedLinuxEventTime: number
 * - linuxBufferTimeoutId: number | undefined  // single timer handle
 *
 * Expected new methods on CanvasPointer:
 * - detectDevice(event: WheelEvent): void
 * - clearLinuxBuffer(): void
 *
 * Performance: This design can handle 10,000+ events without creating any timers
 * (except one for Linux detection), ensuring smooth scrolling performance.
 *
 * @vitest-environment jsdom
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { CanvasPointer } from '@/lib/litegraph/src/CanvasPointer'

describe('CanvasPointer Device Detection - Efficient Timestamp-Based TDD Tests', () => {
  let element: HTMLDivElement
  let pointer: CanvasPointer

  beforeEach(() => {
    element = document.createElement('div')
    pointer = new CanvasPointer(element)
    // Mock performance.now() for timestamp-based testing
    vi.spyOn(performance, 'now').mockReturnValue(0)
    vi.spyOn(global, 'setTimeout')
    vi.spyOn(global, 'clearTimeout')
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.clearAllTimers()
  })

  describe('Initial State', () => {
    it('should start in mouse detected mode immediately after loading', () => {
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should have no last wheel event time immediately after loading', () => {
      expect(pointer.lastWheelEventTime).toBe(0)
      expect(pointer.hasReceivedWheelEvent).toBe(false)
    })

    it('should have no buffered Linux event immediately after loading', () => {
      expect(pointer.bufferedLinuxEvent).toBeUndefined()
      expect(pointer.bufferedLinuxEventTime).toBe(0)
      expect(pointer.linuxBufferTimeoutId).toBeUndefined()
    })
  })

  describe('First Event Detection', () => {
    describe('switching to trackpad on first event', () => {
      it('should switch to trackpad if first event is pinch-to-zoom with deltaY < 10', () => {
        const event = new WheelEvent('wheel', {
          ctrlKey: true,
          deltaY: 9.5,
          deltaX: 0
        })

        pointer.isTrackpadGesture(event)
        expect(pointer.detectedDevice).toBe('trackpad')
        expect(pointer.lastWheelEventTime).toBe(0) // Records current time
      })

      it('should switch to trackpad if first event is pinch-to-zoom with deltaY = 9.999', () => {
        const event = new WheelEvent('wheel', {
          ctrlKey: true,
          deltaY: 9.999,
          deltaX: 0
        })

        pointer.isTrackpadGesture(event)
        expect(pointer.detectedDevice).toBe('trackpad')
      })

      it('should NOT switch to trackpad if first event is pinch-to-zoom with deltaY = 10', () => {
        const event = new WheelEvent('wheel', {
          ctrlKey: true,
          deltaY: 10,
          deltaX: 0
        })

        pointer.isTrackpadGesture(event)
        expect(pointer.detectedDevice).toBe('mouse')
      })

      it('should switch to trackpad if first event is two-finger panning with integer values', () => {
        const event = new WheelEvent('wheel', {
          ctrlKey: false,
          deltaY: 5,
          deltaX: -3
        })

        pointer.isTrackpadGesture(event)
        expect(pointer.detectedDevice).toBe('trackpad')
      })

      it('should switch to trackpad if first event is two-finger panning with ctrlKey true', () => {
        const event = new WheelEvent('wheel', {
          ctrlKey: true,
          deltaY: 7,
          deltaX: 4
        })

        pointer.isTrackpadGesture(event)
        expect(pointer.detectedDevice).toBe('trackpad')
      })

      it('should switch to trackpad if first event is negative pinch-to-zoom with deltaY > -10', () => {
        const event = new WheelEvent('wheel', {
          ctrlKey: true,
          deltaY: -9.5,
          deltaX: 0
        })

        pointer.isTrackpadGesture(event)
        expect(pointer.detectedDevice).toBe('trackpad')
      })
    })

    describe('remaining in mouse mode on first event', () => {
      it('should remain in mouse mode if first event is pinch-to-zoom with deltaY >= 10', () => {
        const event = new WheelEvent('wheel', {
          ctrlKey: true,
          deltaY: 10.1,
          deltaX: 0
        })

        pointer.isTrackpadGesture(event)
        expect(pointer.detectedDevice).toBe('mouse')
      })

      it('should remain in mouse mode if first event is mouse wheel with deltaY = 120', () => {
        const event = new WheelEvent('wheel', {
          ctrlKey: false,
          deltaY: 120,
          deltaX: 0
        })

        pointer.isTrackpadGesture(event)
        expect(pointer.detectedDevice).toBe('mouse')
      })

      it('should remain in mouse mode if first event has only deltaY (no deltaX)', () => {
        const event = new WheelEvent('wheel', {
          ctrlKey: false,
          deltaY: 30,
          deltaX: 0
        })

        pointer.isTrackpadGesture(event)
        expect(pointer.detectedDevice).toBe('mouse')
      })
    })
  })

  describe('Mode Switching from Mouse to Trackpad', () => {
    beforeEach(() => {
      // Ensure we start in mouse mode
      pointer.detectedDevice = 'mouse'
      // Simulate a previous event to establish timing
      pointer.lastWheelEventTime = 0
      pointer.hasReceivedWheelEvent = true
    })

    it('should switch to trackpad on two-finger panning with non-zero deltaX and deltaY', () => {
      // Simulate 500ms has passed since last event
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: 15,
        deltaX: 8
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('trackpad')
    })

    it('should NOT switch to trackpad on two-finger panning with zero deltaX', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: 15,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should NOT switch to trackpad on two-finger panning with zero deltaY', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: 0,
        deltaX: 15
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should switch to trackpad on pinch-to-zoom with deltaY < 10', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: true,
        deltaY: 9.99,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('trackpad')
    })

    it('should switch to trackpad on pinch-to-zoom with deltaY = -5.5', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: true,
        deltaY: -5.5,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('trackpad')
    })

    it('should NOT switch to trackpad on pinch-to-zoom with deltaY = 10', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: true,
        deltaY: 10,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should NOT switch to trackpad on pinch-to-zoom with deltaY = -10', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: true,
        deltaY: -10,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('mouse')
    })
  })

  describe('Mode Switching from Trackpad to Mouse', () => {
    beforeEach(() => {
      // Set to trackpad mode
      pointer.detectedDevice = 'trackpad'
      pointer.lastWheelEventTime = 0
      pointer.hasReceivedWheelEvent = true
    })

    it('should switch to mouse on clear mouse wheel event with deltaY > 80', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: 80.1,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should switch to mouse on clear mouse wheel event with deltaY = 120', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: 120,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should switch to mouse on clear mouse wheel event with negative deltaY < -80', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: -90,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should NOT switch to mouse with deltaY = 80', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: 80,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('trackpad')
    })

    it('should NOT switch to mouse with deltaY = -80', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: -80,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('trackpad')
    })

    it('should NOT switch to mouse with deltaY = 79.999', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: 79.999,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('trackpad')
    })
  })

  describe('500ms Cooldown Period', () => {
    it('should NOT allow switching from mouse to trackpad within 500ms', () => {
      pointer.detectedDevice = 'mouse'

      // First event at time 0
      vi.spyOn(performance, 'now').mockReturnValue(0)
      const event1 = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: 60,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)
      expect(pointer.lastWheelEventTime).toBe(0)

      // Try to switch after 499ms - should fail
      vi.spyOn(performance, 'now').mockReturnValue(499)
      const event2 = new WheelEvent('wheel', {
        ctrlKey: true,
        deltaY: 5,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should allow switching from mouse to trackpad after 500ms', () => {
      pointer.detectedDevice = 'mouse'

      // First event at time 0
      vi.spyOn(performance, 'now').mockReturnValue(0)
      const event1 = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: 60,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)

      // Try to switch after 500ms - should succeed
      vi.spyOn(performance, 'now').mockReturnValue(500)
      const event2 = new WheelEvent('wheel', {
        ctrlKey: true,
        deltaY: 5,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)
      expect(pointer.detectedDevice).toBe('trackpad')
    })

    it('should NOT allow switching from trackpad to mouse within 500ms', () => {
      pointer.detectedDevice = 'trackpad'

      // First event at time 0
      vi.spyOn(performance, 'now').mockReturnValue(0)
      const event1 = new WheelEvent('wheel', {
        ctrlKey: true,
        deltaY: 5,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)

      // Try to switch after 400ms - should fail
      vi.spyOn(performance, 'now').mockReturnValue(400)
      const event2 = new WheelEvent('wheel', {
        ctrlKey: false,
        deltaY: 120,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)
      expect(pointer.detectedDevice).toBe('trackpad')
    })

    it('should maintain cooldown even with multiple events', () => {
      pointer.detectedDevice = 'mouse'

      // Series of events that would normally trigger trackpad
      const trackpadEvents = [
        { deltaY: 5, deltaX: 3 },
        { deltaY: -7, deltaX: 2 },
        { deltaY: 8, deltaX: -4 }
      ]

      // Send first mouse event at time 0
      vi.spyOn(performance, 'now').mockReturnValue(0)
      pointer.isTrackpadGesture(
        new WheelEvent('wheel', { deltaY: 60, deltaX: 0 })
      )

      // Send trackpad events within 500ms window
      trackpadEvents.forEach((eventData, index) => {
        vi.spyOn(performance, 'now').mockReturnValue((index + 1) * 100) // 100ms, 200ms, 300ms
        const event = new WheelEvent('wheel', eventData)
        pointer.isTrackpadGesture(event)
        expect(pointer.detectedDevice).toBe('mouse') // Should remain mouse
      })

      // After 500ms from last event (300ms + 500ms = 800ms), should be able to switch
      vi.spyOn(performance, 'now').mockReturnValue(800)
      const switchEvent = new WheelEvent('wheel', { deltaY: 5, deltaX: 3 })
      pointer.isTrackpadGesture(switchEvent)
      expect(pointer.detectedDevice).toBe('trackpad')
    })
  })

  describe('Linux Wheel Event Buffering', () => {
    beforeEach(() => {
      pointer.detectedDevice = 'trackpad'
      pointer.lastWheelEventTime = 0
      pointer.hasReceivedWheelEvent = true
      vi.clearAllMocks()
    })

    it('should buffer possible Linux wheel event and create single timeout', () => {
      const setTimeoutSpy = vi.spyOn(global, 'setTimeout')
      vi.spyOn(performance, 'now').mockReturnValue(500) // Allow mode switching

      const event = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.bufferedLinuxEvent).toBe(event)
      expect(pointer.bufferedLinuxEventTime).toBe(500)
      expect(pointer.detectedDevice).toBe('trackpad') // No immediate switch

      // Should create exactly ONE timeout for buffer clearing
      expect(setTimeoutSpy).toHaveBeenCalledTimes(1)
      expect(setTimeoutSpy).toHaveBeenCalledWith(expect.any(Function), 10)
    })

    it('should reuse timer when buffering new Linux event', () => {
      const setTimeoutSpy = vi.spyOn(global, 'setTimeout')
      const clearTimeoutSpy = vi.spyOn(global, 'clearTimeout')

      // First Linux event
      vi.spyOn(performance, 'now').mockReturnValue(500)
      const event1 = new WheelEvent('wheel', { deltaY: 15, deltaX: 0 })
      pointer.isTrackpadGesture(event1)
      const firstTimeoutId = pointer.linuxBufferTimeoutId

      // Second Linux event before timeout
      vi.spyOn(performance, 'now').mockReturnValue(505)
      const event2 = new WheelEvent('wheel', { deltaY: 10, deltaX: 0 })
      pointer.isTrackpadGesture(event2)

      // Should clear the first timeout and create a new one
      expect(clearTimeoutSpy).toHaveBeenCalledWith(firstTimeoutId)
      expect(setTimeoutSpy).toHaveBeenCalledTimes(2)
      expect(pointer.bufferedLinuxEvent).toBe(event2)
    })

    it('should buffer negative Linux wheel values', () => {
      const setTimeoutSpy = vi.spyOn(global, 'setTimeout')
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        deltaY: -10,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.bufferedLinuxEvent).toBe(event)
      expect(pointer.detectedDevice).toBe('trackpad')
      expect(setTimeoutSpy).toHaveBeenCalledTimes(1)
    })

    it('should NOT buffer event with deltaY < 10', () => {
      const setTimeoutSpy = vi.spyOn(global, 'setTimeout')
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        deltaY: 9,
        deltaX: 0
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.bufferedLinuxEvent).toBeUndefined()
      expect(pointer.detectedDevice).toBe('trackpad')
      expect(setTimeoutSpy).not.toHaveBeenCalled() // No timer created
    })

    it('should NOT buffer event with non-zero deltaX', () => {
      const setTimeoutSpy = vi.spyOn(global, 'setTimeout')
      vi.spyOn(performance, 'now').mockReturnValue(500)

      const event = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 1
      })

      pointer.isTrackpadGesture(event)
      expect(pointer.bufferedLinuxEvent).toBeUndefined()
      expect(pointer.detectedDevice).toBe('trackpad')
      expect(setTimeoutSpy).not.toHaveBeenCalled() // No timer created
    })

    it('should switch to mouse if follow-up event has same deltaY within 10ms', () => {
      const clearTimeoutSpy = vi.spyOn(global, 'clearTimeout')

      // First event - buffered at time 500
      vi.spyOn(performance, 'now').mockReturnValue(500)
      const event1 = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)
      expect(pointer.bufferedLinuxEvent).toBe(event1)
      const timeoutId = pointer.linuxBufferTimeoutId

      // Follow-up within 10ms with same deltaY
      vi.spyOn(performance, 'now').mockReturnValue(509)
      const event2 = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)

      expect(pointer.detectedDevice).toBe('mouse')
      expect(pointer.bufferedLinuxEvent).toBeUndefined()
      expect(clearTimeoutSpy).toHaveBeenCalledWith(timeoutId) // Timer cleared
    })

    it('should switch to mouse if follow-up event is divisible by original deltaY within 10ms', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // First event - buffered
      const event1 = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)

      // Follow-up within 10ms with deltaY divisible by 10
      vi.spyOn(performance, 'now').mockReturnValue(505)
      const event2 = new WheelEvent('wheel', {
        deltaY: 30,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)

      expect(pointer.detectedDevice).toBe('mouse')
      expect(pointer.bufferedLinuxEvent).toBeUndefined()
    })

    it('should switch to mouse if follow-up deltaY is divisible by original (base 15)', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // First event with base 15
      const event1 = new WheelEvent('wheel', {
        deltaY: 15,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)

      // Follow-up with multiple of 15
      vi.spyOn(performance, 'now').mockReturnValue(508)
      const event2 = new WheelEvent('wheel', {
        deltaY: 45,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)

      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should switch to mouse if original deltaY is divisible by follow-up', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // First event with larger value
      const event1 = new WheelEvent('wheel', {
        deltaY: 30,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)

      // Follow-up with divisor
      vi.spyOn(performance, 'now').mockReturnValue(507)
      const event2 = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)

      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should NOT switch to mouse if follow-up is not divisible', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // First event
      const event1 = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)

      // Follow-up with non-divisible value
      vi.spyOn(performance, 'now').mockReturnValue(505)
      const event2 = new WheelEvent('wheel', {
        deltaY: 13,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)

      expect(pointer.detectedDevice).toBe('trackpad')
    })

    it('should NOT switch to mouse if follow-up comes after 10ms', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // First event
      const event1 = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)

      // Follow-up after 10ms
      vi.spyOn(performance, 'now').mockReturnValue(511)
      const event2 = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)

      expect(pointer.detectedDevice).toBe('trackpad')
    })

    it('should call clearLinuxBuffer method after 10ms timeout', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)
      vi.useFakeTimers() // Use fake timers just for this test

      const event = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event)
      expect(pointer.bufferedLinuxEvent).toBe(event)

      // Simulate timeout firing
      vi.runOnlyPendingTimers()
      expect(pointer.bufferedLinuxEvent).toBeUndefined()
      expect(pointer.linuxBufferTimeoutId).toBeUndefined()

      vi.useRealTimers() // Restore for other tests
    })

    it('should handle negative Linux wheel values', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // First negative event
      const event1 = new WheelEvent('wheel', {
        deltaY: -15,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)

      // Follow-up with same negative value
      vi.spyOn(performance, 'now').mockReturnValue(505)
      const event2 = new WheelEvent('wheel', {
        deltaY: -15,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)

      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should handle mixed sign Linux wheel values if divisible', () => {
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // First positive event
      const event1 = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)

      // Follow-up with negative multiple
      vi.spyOn(performance, 'now').mockReturnValue(505)
      const event2 = new WheelEvent('wheel', {
        deltaY: -30,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)

      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should allow buffering during 500ms cooldown as exception', () => {
      pointer.detectedDevice = 'trackpad'

      // Send initial event at time 0
      vi.spyOn(performance, 'now').mockReturnValue(0)
      const event1 = new WheelEvent('wheel', {
        deltaY: 5,
        deltaX: 2
      })
      pointer.isTrackpadGesture(event1)

      // Within cooldown at 100ms, but Linux buffering should still work
      vi.spyOn(performance, 'now').mockReturnValue(100)
      const event2 = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)
      expect(pointer.bufferedLinuxEvent).toBe(event2)

      // Follow-up for Linux detection at 105ms
      vi.spyOn(performance, 'now').mockReturnValue(105)
      const event3 = new WheelEvent('wheel', {
        deltaY: 20,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event3)

      // Should switch despite being within original 500ms window
      expect(pointer.detectedDevice).toBe('mouse')
    })
  })

  describe('Performance and Efficiency', () => {
    it('should not create timers for regular wheel events', () => {
      const setTimeoutSpy = vi.spyOn(global, 'setTimeout')
      pointer.detectedDevice = 'mouse'

      // Simulate rapid scrolling without Linux-like patterns
      for (let i = 0; i < 100; i++) {
        vi.spyOn(performance, 'now').mockReturnValue(i * 16) // 60fps scrolling
        const event = new WheelEvent('wheel', {
          deltaY: 120, // Clear mouse wheel value
          deltaX: 0
        })
        pointer.isTrackpadGesture(event)
      }

      // Should create NO timers for regular mouse wheel events
      expect(setTimeoutSpy).not.toHaveBeenCalled()
    })

    it('should create at most one timer for Linux detection', () => {
      const setTimeoutSpy = vi.spyOn(global, 'setTimeout')
      pointer.detectedDevice = 'trackpad'

      // Send a Linux-like event that requires buffering
      vi.spyOn(performance, 'now').mockReturnValue(500)
      const event1 = new WheelEvent('wheel', { deltaY: 10, deltaX: 0 })
      pointer.isTrackpadGesture(event1)

      // Should create exactly one timer
      expect(setTimeoutSpy).toHaveBeenCalledTimes(1)

      // Send more regular events
      for (let i = 1; i <= 10; i++) {
        vi.spyOn(performance, 'now').mockReturnValue(500 + i * 100)
        const event = new WheelEvent('wheel', { deltaY: 5, deltaX: 3 })
        pointer.isTrackpadGesture(event)
      }

      // Still only one timer (the Linux buffer timeout)
      expect(setTimeoutSpy).toHaveBeenCalledTimes(1)
    })

    it('should handle thousands of events efficiently', () => {
      const setTimeoutSpy = vi.spyOn(global, 'setTimeout')
      let maxTimersCreated = 0

      // Simulate extended scrolling session with mixed inputs
      for (let i = 0; i < 10000; i++) {
        vi.spyOn(performance, 'now').mockReturnValue(i)

        // Mix of different event types
        const eventType = i % 3
        let event: WheelEvent

        if (eventType === 0) {
          // Mouse wheel
          event = new WheelEvent('wheel', {
            deltaY: 120,
            deltaX: 0
          })
        } else if (eventType === 1) {
          // Trackpad two-finger
          event = new WheelEvent('wheel', {
            deltaY: Math.floor(Math.random() * 20),
            deltaX: Math.floor(Math.random() * 20)
          })
        } else {
          // Pinch to zoom
          event = new WheelEvent('wheel', {
            ctrlKey: true,
            deltaY: Math.random() * 10,
            deltaX: 0
          })
        }

        pointer.isTrackpadGesture(event)

        // Track maximum timers created
        maxTimersCreated = Math.max(
          maxTimersCreated,
          setTimeoutSpy.mock.calls.length
        )
      }

      // Should create at most a few timers for Linux detection, not thousands
      expect(maxTimersCreated).toBeLessThan(10)
    })

    it('should use minimal memory with timestamp approach', () => {
      // This test verifies the implementation uses timestamps, not stored events
      const initialProps = Object.keys(pointer).length

      // Process many events
      for (let i = 0; i < 1000; i++) {
        vi.spyOn(performance, 'now').mockReturnValue(i * 10)
        const event = new WheelEvent('wheel', {
          deltaY: 60 + Math.random() * 100,
          deltaX: Math.random() * 50
        })
        pointer.isTrackpadGesture(event)
      }

      // Should only have a few properties for tracking state
      const finalProps = Object.keys(pointer).length
      expect(finalProps - initialProps).toBeLessThanOrEqual(5) // Only added minimal tracking properties

      // Verify we store timestamps, not events (except Linux buffer)
      expect(typeof pointer.lastWheelEventTime).toBe('number')
      expect(typeof pointer.bufferedLinuxEventTime).toBe('number')
    })

    it('should handle rapid mode switching efficiently', () => {
      const setTimeoutSpy = vi.spyOn(global, 'setTimeout')

      // Rapidly switch between mouse and trackpad modes
      for (let i = 0; i < 100; i++) {
        const baseTime = i * 600 // Every 600ms to allow switching

        // Mouse event
        vi.spyOn(performance, 'now').mockReturnValue(baseTime)
        pointer.isTrackpadGesture(
          new WheelEvent('wheel', { deltaY: 120, deltaX: 0 })
        )

        // Trackpad event
        vi.spyOn(performance, 'now').mockReturnValue(baseTime + 500)
        pointer.isTrackpadGesture(
          new WheelEvent('wheel', { deltaY: 5, deltaX: 3 })
        )
      }

      // Should create minimal or no timers despite 200 events
      expect(setTimeoutSpy.mock.calls.length).toBeLessThan(5)
    })
  })

  describe('Edge Cases and Complex Scenarios', () => {
    it('should handle float values correctly for mouse detection', () => {
      pointer.detectedDevice = 'trackpad'
      pointer.lastWheelEventTime = 0
      pointer.hasReceivedWheelEvent = true
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // Float value <= 80 should NOT switch from trackpad
      const event1 = new WheelEvent('wheel', {
        deltaY: 60.5,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)
      expect(pointer.detectedDevice).toBe('trackpad')

      // Float value > 80 should switch to mouse
      vi.spyOn(performance, 'now').mockReturnValue(1000) // 500ms later
      const event2 = new WheelEvent('wheel', {
        deltaY: 80.1,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should handle integer values correctly for trackpad detection', () => {
      pointer.detectedDevice = 'mouse'
      pointer.lastWheelEventTime = 0
      pointer.hasReceivedWheelEvent = true
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // Integer values in two-finger panning
      const event = new WheelEvent('wheel', {
        deltaY: 5,
        deltaX: 3
      })
      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('trackpad')
    })

    it('should correctly identify pinch-to-zoom with ctrlKey', () => {
      const event = new WheelEvent('wheel', {
        ctrlKey: true,
        deltaY: 250.5,
        deltaX: 0
      })

      // This is pinch-to-zoom but deltaY > 10, so stays as mouse on first event
      pointer.isTrackpadGesture(event)
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should handle rapid event sequences', () => {
      pointer.detectedDevice = 'mouse'
      pointer.lastWheelEventTime = 0

      // Simulate rapid scrolling
      for (let i = 0; i < 10; i++) {
        vi.spyOn(performance, 'now').mockReturnValue(i * 30) // 30ms between events
        const event = new WheelEvent('wheel', {
          deltaY: 60,
          deltaX: 0
        })
        pointer.isTrackpadGesture(event)
        expect(pointer.detectedDevice).toBe('mouse')
      }
    })

    it('should handle boundary values for pinch-to-zoom detection', () => {
      // Test deltaY = 10 (boundary)
      const event1 = new WheelEvent('wheel', {
        ctrlKey: true,
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)
      expect(pointer.detectedDevice).toBe('mouse')

      // Reset and test deltaY = 9.999999
      pointer = new CanvasPointer(element)
      const event2 = new WheelEvent('wheel', {
        ctrlKey: true,
        deltaY: 9.999999,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)
      expect(pointer.detectedDevice).toBe('trackpad')
    })

    it('should handle boundary values for mouse wheel detection', () => {
      pointer.detectedDevice = 'trackpad'
      pointer.lastWheelEventTime = 0
      pointer.hasReceivedWheelEvent = true
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // Test deltaY = 80 (boundary)
      const event1 = new WheelEvent('wheel', {
        deltaY: 80,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)
      expect(pointer.detectedDevice).toBe('trackpad')

      // Test deltaY = 80.000001
      vi.spyOn(performance, 'now').mockReturnValue(1000) // 500ms later
      const event2 = new WheelEvent('wheel', {
        deltaY: 80.000001,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should handle Linux wheel detection with various multiples', () => {
      pointer.detectedDevice = 'trackpad'
      pointer.lastWheelEventTime = 0
      pointer.hasReceivedWheelEvent = true
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // Test with base 10 and multiple 50
      const event1 = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event1)

      vi.spyOn(performance, 'now').mockReturnValue(505) // 5ms later
      const event2 = new WheelEvent('wheel', {
        deltaY: 50,
        deltaX: 0
      })
      pointer.isTrackpadGesture(event2)
      expect(pointer.detectedDevice).toBe('mouse')
    })

    it('should not confuse trackpad integers with Linux wheel', () => {
      pointer.detectedDevice = 'trackpad'
      pointer.lastWheelEventTime = 0
      pointer.hasReceivedWheelEvent = true
      vi.spyOn(performance, 'now').mockReturnValue(500)

      // Trackpad two-finger panning with integers
      const event1 = new WheelEvent('wheel', {
        deltaY: 10,
        deltaX: 5 // Non-zero deltaX
      })
      pointer.isTrackpadGesture(event1)

      // Should not buffer this as Linux event
      expect(pointer.bufferedLinuxEvent).toBeUndefined()
      expect(pointer.detectedDevice).toBe('trackpad')
    })
  })

  describe('Input Type Validation', () => {
    describe('Two-finger panning validation', () => {
      it('should accept integer deltaY values', () => {
        const values = [0, 1, -1, 100, -100, 999, -999]
        values.forEach((deltaY) => {
          const event = new WheelEvent('wheel', {
            ctrlKey: false,
            deltaY,
            deltaX: 5
          })
          expect(Number.isInteger(event.deltaY)).toBe(true)
        })
      })

      it('should accept integer deltaX values', () => {
        const values = [0, 1, -1, 100, -100, 999, -999]
        values.forEach((deltaX) => {
          const event = new WheelEvent('wheel', {
            ctrlKey: false,
            deltaY: 5,
            deltaX
          })
          expect(Number.isInteger(event.deltaX)).toBe(true)
        })
      })

      it('should handle ctrlKey true or false', () => {
        ;[true, false].forEach((ctrlKey) => {
          const event = new WheelEvent('wheel', {
            ctrlKey,
            deltaY: 5,
            deltaX: 3
          })
          expect(typeof event.ctrlKey).toBe('boolean')
        })
      })
    })

    describe('Pinch-to-zoom validation', () => {
      it('should always have ctrlKey true', () => {
        const event = new WheelEvent('wheel', {
          ctrlKey: true,
          deltaY: 5.5,
          deltaX: 0
        })
        expect(event.ctrlKey).toBe(true)
      })

      it('should accept float deltaY values in range -1000 to 1000', () => {
        const values = [-1000, -999.99, -0.1, 0, 0.1, 999.99, 1000]
        values.forEach((deltaY) => {
          const event = new WheelEvent('wheel', {
            ctrlKey: true,
            deltaY,
            deltaX: 0
          })
          expect(event.deltaY).toBeGreaterThanOrEqual(-1000)
          expect(event.deltaY).toBeLessThanOrEqual(1000)
        })
      })

      it('should always have deltaX = 0', () => {
        const event = new WheelEvent('wheel', {
          ctrlKey: true,
          deltaY: 5.5,
          deltaX: 0
        })
        expect(event.deltaX).toBe(0)
      })
    })

    describe('Mouse input validation', () => {
      it('should accept float deltaX values in range -1000 to 1000', () => {
        const values = [-1000, -500.5, 0, 500.5, 1000]
        values.forEach((deltaX) => {
          const event = new WheelEvent('wheel', {
            deltaY: 120,
            deltaX
          })
          expect(event.deltaX).toBeGreaterThanOrEqual(-1000)
          expect(event.deltaX).toBeLessThanOrEqual(1000)
        })
      })

      it('should have deltaY >= 60 for Windows/Mac mouse', () => {
        const values = [60, 60.1, 80, 120, 240]
        values.forEach((deltaY) => {
          const event = new WheelEvent('wheel', {
            deltaY,
            deltaX: 0
          })
          expect(event.deltaY).toBeGreaterThanOrEqual(60)
        })
      })

      it('should have integer deltaY as multiples of 10 or 15 for Linux', () => {
        // Base 10 multiples
        const base10Values = [10, 20, 30, 40, 50, -10, -20, -30]
        base10Values.forEach((deltaY) => {
          expect(Number.isInteger(deltaY)).toBe(true)
          // Use Math.abs to avoid JavaScript's -0 vs 0 issue with modulo on negative numbers
          expect(Math.abs(deltaY) % 10).toBe(0)
        })

        // Base 15 multiples
        const base15Values = [15, 30, 45, 60, -15, -30, -45]
        base15Values.forEach((deltaY) => {
          expect(Number.isInteger(deltaY)).toBe(true)
          // Use Math.abs to avoid JavaScript's -0 vs 0 issue with modulo on negative numbers
          expect(Math.abs(deltaY) % 15).toBe(0)
        })
      })
    })

    describe('Float vs Integer understanding', () => {
      it('should recognize that integers are valid float values', () => {
        const integerValues = [0, 1, -1, 10, -10, 100]
        integerValues.forEach((value) => {
          expect(Number.isInteger(value)).toBe(true)
          expect(typeof value === 'number').toBe(true) // Valid as float
        })
      })

      it('should recognize that decimals are NOT valid integer values', () => {
        const decimalValues = [0.1, -0.1, 10.5, -10.5, 99.99]
        decimalValues.forEach((value) => {
          expect(Number.isInteger(value)).toBe(false)
          expect(typeof value === 'number').toBe(true) // Still valid as float
        })
      })

      it('should correctly validate pinch-to-zoom deltaY as float', () => {
        // These are all valid float values for pinch-to-zoom
        const validValues = [0, 1, -1, 0.5, -0.5, 999, -999, 500.123]
        validValues.forEach((value) => {
          const event = new WheelEvent('wheel', {
            ctrlKey: true,
            deltaY: value,
            deltaX: 0
          })
          expect(typeof event.deltaY === 'number').toBe(true)
          expect(event.deltaY >= -1000 && event.deltaY <= 1000).toBe(true)
        })
      })
    })
  })
})
