import type { CompassCorners } from './interfaces'
import { dist2 } from './measure'
import type { CanvasPointerEvent } from './types/events'

/**
 * Allows click and drag actions to be declared ahead of time during a pointerdown event.
 *
 * By default, it retains the most recent event of each type until it is reset (on pointerup).
 * - {@link eDown}
 * - {@link eMove}
 * - {@link eUp}
 *
 * Depending on whether the user clicks or drags the pointer, only the appropriate callbacks are called:
 * - {@link onClick}
 * - {@link onDoubleClick}
 * - {@link onDragStart}
 * - {@link onDrag}
 * - {@link onDragEnd}
 * - {@link finally}
 * @see
 * - {@link LGraphCanvas.processMouseDown}
 * - {@link LGraphCanvas.processMouseMove}
 * - {@link LGraphCanvas.processMouseUp}
 */
export class CanvasPointer {
  /** Maximum time in milliseconds to ignore click drift */
  static bufferTime = 150

  /** Maximum gap between pointerup and pointerdown events to be considered as a double click */
  static doubleClickTime = 300

  /** Maximum offset from click location */
  static get maxClickDrift() {
    return this._maxClickDrift
  }

  static set maxClickDrift(value) {
    this._maxClickDrift = value
    this._maxClickDrift2 = value * value
  }

  private static _maxClickDrift = 6
  /** {@link maxClickDrift} squared.  Used to calculate click drift without `sqrt`. */
  private static _maxClickDrift2 = this._maxClickDrift ** 2

  /** Assume that "wheel" events with both deltaX and deltaY less than this value are trackpad gestures. */
  static trackpadThreshold = 60

  /**
   * The minimum time between "wheel" events to allow switching between trackpad
   * and mouse modes.
   *
   * This prevents trackpad "flick" panning from registering as regular mouse wheel.
   * After a flick gesture is complete, the automatic wheel events are sent with
   * reduced frequency, but much higher deltaX and deltaY values.
   */
  static trackpadMaxGap = 500

  /** The maximum time in milliseconds to buffer a high-res wheel event. */
  static maxHighResBufferTime = 10

  /** The element this PointerState should capture input against when dragging. */
  element: Element
  /** Pointer ID used by drag capture. */
  pointerId?: number

  /** Set to true when if the pointer moves far enough after a down event, before the corresponding up event is fired. */
  dragStarted: boolean = false

  /** The {@link eUp} from the last successful click */
  eLastDown?: CanvasPointerEvent

  /** Used downstream for touch event support. */
  isDouble: boolean = false
  /** Used downstream for touch event support. */
  isDown: boolean = false

  /** The resize handle currently being hovered or dragged */
  resizeDirection?: CompassCorners

  /**
   * If `true`, {@link eDown}, {@link eMove}, and {@link eUp} will be set to
   * `undefined` when {@link reset} is called.
   *
   * Default: `true`
   */
  clearEventsOnReset: boolean = true

  /** The last pointerdown event for the primary button */
  eDown?: CanvasPointerEvent
  /** The last pointermove event for the primary button */
  eMove?: CanvasPointerEvent
  /** The last pointerup event for the primary button */
  eUp?: CanvasPointerEvent

  /** Currently detected input device type */
  detectedDevice: 'mouse' | 'trackpad' = 'mouse'

  /** Timestamp of last wheel event for cooldown tracking */
  lastWheelEventTime: number = 0

  /** Flag to track if we've received the first wheel event */
  hasReceivedWheelEvent: boolean = false

  /** Buffered Linux wheel event awaiting confirmation */
  bufferedLinuxEvent?: WheelEvent

  /** Timestamp when Linux event was buffered */
  bufferedLinuxEventTime: number = 0

  /** Timer ID for Linux buffer clearing */
  linuxBufferTimeoutId?: ReturnType<typeof setTimeout>

  /**
   * If set, as soon as the mouse moves outside the click drift threshold, this action is run once.
   * @param pointer [DEPRECATED] This parameter will be removed in a future release.
   * @param eMove The pointermove event of this ongoing drag action.
   *
   * It is possible for no `pointermove` events to occur, but still be far from
   * the original `pointerdown` event. In this case, {@link eMove} will be null, and
   * {@link onDragEnd} will be called immediately after {@link onDragStart}.
   */
  onDragStart?(pointer: this, eMove?: CanvasPointerEvent): unknown

  /**
   * Called on pointermove whilst dragging.
   * @param eMove The pointermove event of this ongoing drag action
   */
  onDrag?(eMove: CanvasPointerEvent): unknown

  /**
   * Called on pointerup after dragging (i.e. not called if clicked).
   * @param upEvent The pointerup or pointermove event that triggered this callback
   */
  onDragEnd?(upEvent: CanvasPointerEvent): unknown

  /**
   * Callback that will be run once, the next time a pointerup event appears to be a normal click.
   * @param upEvent The pointerup or pointermove event that triggered this callback
   */
  onClick?(upEvent: CanvasPointerEvent): unknown

  /**
   * Callback that will be run once, the next time a pointerup event appears to be a normal click.
   * @param upEvent The pointerup or pointermove event that triggered this callback
   */
  onDoubleClick?(upEvent: CanvasPointerEvent): unknown

  /**
   * Run-once callback, called at the end of any click or drag, whether or not it was successful in any way.
   *
   * The setter of this callback will call the existing value before replacing it.
   * Therefore, simply setting this value twice will execute the first callback.
   */
  get finally() {
    return this._finally
  }

  set finally(value) {
    try {
      this._finally?.()
    } finally {
      this._finally = value
    }
  }

  private _finally?: () => unknown

  constructor(element: Element) {
    this.element = element
  }

  /**
   * Callback for `pointerdown` events.  To be used as the event handler (or called by it).
   * @param e The `pointerdown` event
   */
  down(e: CanvasPointerEvent): void {
    this.reset()
    this.eDown = e
    this.pointerId = e.pointerId
    this.element.setPointerCapture(e.pointerId)
  }

  /**
   * Callback for `pointermove` events.  To be used as the event handler (or called by it).
   * @param e The `pointermove` event
   */
  move(e: CanvasPointerEvent): void {
    const { eDown } = this
    if (!eDown) return

    // No buttons down, but eDown exists - clean up & leave
    if (!e.buttons) {
      this.reset()
      return
    }

    // Primary button released - treat as pointerup.
    if (!(e.buttons & eDown.buttons)) {
      this._completeClick(e)
      this.reset()
      return
    }
    this.eMove = e
    this.onDrag?.(e)

    // Dragging, but no callback to run
    if (this.dragStarted) return

    const longerThanBufferTime =
      e.timeStamp - eDown.timeStamp > CanvasPointer.bufferTime
    if (longerThanBufferTime || !this._hasSamePosition(e, eDown)) {
      this._setDragStarted(e)
    }
  }

  /**
   * Callback for `pointerup` events.  To be used as the event handler (or called by it).
   * @param e The `pointerup` event
   */
  up(e: CanvasPointerEvent): boolean {
    if (e.button !== this.eDown?.button) return false

    this._completeClick(e)
    const { dragStarted } = this
    this.reset()
    return !dragStarted
  }

  private _completeClick(e: CanvasPointerEvent): void {
    const { eDown } = this
    if (!eDown) return

    this.eUp = e

    if (this.dragStarted) {
      // A move event already started drag
      this.onDragEnd?.(e)
    } else if (!this._hasSamePosition(e, eDown)) {
      // Teleport without a move event (e.g. tab out, move, tab back)
      this._setDragStarted()
      this.onDragEnd?.(e)
    } else if (this.onDoubleClick && this._isDoubleClick()) {
      // Double-click event
      this.onDoubleClick(e)
      this.eLastDown = undefined
    } else {
      // Normal click event
      this.onClick?.(e)
      this.eLastDown = eDown
    }
  }

  /**
   * Checks if two events occurred near each other - not further apart than the maximum click drift.
   * @param a The first event to compare
   * @param b The second event to compare
   * @param tolerance2 The maximum distance (squared) before the positions are considered different
   * @returns `true` if the two events were no more than {@link maxClickDrift} apart, otherwise `false`
   */
  private _hasSamePosition(
    a: PointerEvent,
    b: PointerEvent,
    tolerance2 = CanvasPointer._maxClickDrift2
  ): boolean {
    const drift = dist2(a.clientX, a.clientY, b.clientX, b.clientY)
    return drift <= tolerance2
  }

  /**
   * Checks whether the pointer is currently past the max click drift threshold.
   * @returns `true` if the latest pointer event is past the the click drift threshold
   */
  private _isDoubleClick(): boolean {
    const { eDown, eLastDown } = this
    if (!eDown || !eLastDown) return false

    // Use thrice the drift distance for double-click gap
    const tolerance2 = (3 * CanvasPointer._maxClickDrift) ** 2
    const diff = eDown.timeStamp - eLastDown.timeStamp
    return (
      diff > 0 &&
      diff < CanvasPointer.doubleClickTime &&
      this._hasSamePosition(eDown, eLastDown, tolerance2)
    )
  }

  private _setDragStarted(eMove?: CanvasPointerEvent): void {
    this.dragStarted = true
    this.onDragStart?.(this, eMove)
    delete this.onDragStart
  }

  /**
   * Checks if the given wheel event is part of a trackpad gesture.
   * This method now uses the new device detection internally for improved accuracy.
   * @param e The wheel event to check
   * @returns `true` if the event is part of a trackpad gesture, otherwise `false`
   */
  isTrackpadGesture(e: WheelEvent): boolean {
    // Use the new device detection
    const now = performance.now()
    const timeSinceLastEvent = Math.max(0, now - this.lastWheelEventTime)
    this.lastWheelEventTime = now

    if (this._isHighResWheelEvent(e, now)) {
      this.detectedDevice = 'mouse'
    } else if (this._isWithinCooldown(timeSinceLastEvent)) {
      if (this._shouldBufferLinuxEvent(e)) {
        this._bufferLinuxEvent(e, now)
      }
    } else {
      this._updateDeviceMode(e, now)
      this.hasReceivedWheelEvent = true
    }

    return this.detectedDevice === 'trackpad'
  }

  /**
   * Validates buffered high res wheel events and switches to mouse mode if pattern matches.
   * @returns `true` if switched to mouse mode
   */
  private _isHighResWheelEvent(event: WheelEvent, now: number): boolean {
    if (!this.bufferedLinuxEvent || this.bufferedLinuxEventTime <= 0) {
      return false
    }

    const timeSinceBuffer = now - this.bufferedLinuxEventTime

    if (timeSinceBuffer > CanvasPointer.maxHighResBufferTime) {
      this._clearLinuxBuffer()
      return false
    }

    if (
      event.deltaX === 0 &&
      this._isLinuxWheelPattern(this.bufferedLinuxEvent.deltaY, event.deltaY)
    ) {
      this._clearLinuxBuffer()
      return true
    }

    return false
  }

  /**
   * Checks if we're within the cooldown period where mode switching is disabled.
   */
  private _isWithinCooldown(timeSinceLastEvent: number): boolean {
    const isFirstEvent = !this.hasReceivedWheelEvent
    const cooldownExpired = timeSinceLastEvent >= CanvasPointer.trackpadMaxGap
    return !isFirstEvent && !cooldownExpired
  }

  /**
   * Updates the device mode based on event patterns.
   */
  private _updateDeviceMode(event: WheelEvent, now: number): void {
    if (this._isTrackpadPattern(event)) {
      this.detectedDevice = 'trackpad'
    } else if (this._isMousePattern(event)) {
      this.detectedDevice = 'mouse'
    } else if (
      this.detectedDevice === 'trackpad' &&
      this._shouldBufferLinuxEvent(event)
    ) {
      this._bufferLinuxEvent(event, now)
    }
  }

  /**
   * Clears the buffered Linux wheel event and associated timer.
   */
  private _clearLinuxBuffer(): void {
    this.bufferedLinuxEvent = undefined
    this.bufferedLinuxEventTime = 0
    if (this.linuxBufferTimeoutId !== undefined) {
      clearTimeout(this.linuxBufferTimeoutId)
      this.linuxBufferTimeoutId = undefined
    }
  }

  /**
   * Checks if the event matches trackpad input patterns.
   * @param event The wheel event to check
   */
  private _isTrackpadPattern(event: WheelEvent): boolean {
    // Two-finger panning: non-zero deltaX AND deltaY
    if (event.deltaX !== 0 && event.deltaY !== 0) return true

    // Pinch-to-zoom: ctrlKey with small deltaY
    if (event.ctrlKey && Math.abs(event.deltaY) < 10) return true

    return false
  }

  /**
   * Checks if the event matches mouse wheel input patterns.
   * @param event The wheel event to check
   */
  private _isMousePattern(event: WheelEvent): boolean {
    const absoluteDeltaY = Math.abs(event.deltaY)

    // Primary threshold for switching from trackpad to mouse
    if (absoluteDeltaY > 80) return true

    // Secondary threshold when already in mouse mode
    return (
      absoluteDeltaY >= 60 &&
      event.deltaX === 0 &&
      this.detectedDevice === 'mouse'
    )
  }

  /**
   * Checks if the event should be buffered as a potential Linux wheel event.
   * @param event The wheel event to check
   */
  private _shouldBufferLinuxEvent(event: WheelEvent): boolean {
    const absoluteDeltaY = Math.abs(event.deltaY)
    const isInLinuxRange = absoluteDeltaY >= 10 && absoluteDeltaY < 60
    const isVerticalOnly = event.deltaX === 0
    const hasIntegerDelta = Number.isInteger(event.deltaY)

    return (
      this.detectedDevice === 'trackpad' &&
      isInLinuxRange &&
      isVerticalOnly &&
      hasIntegerDelta
    )
  }

  /**
   * Buffers a potential Linux wheel event for later confirmation.
   * @param event The event to buffer
   * @param now The current timestamp
   */
  private _bufferLinuxEvent(event: WheelEvent, now: number): void {
    if (this.linuxBufferTimeoutId !== undefined) {
      clearTimeout(this.linuxBufferTimeoutId)
    }

    this.bufferedLinuxEvent = event
    this.bufferedLinuxEventTime = now

    // Set timeout to clear buffer after 10ms
    this.linuxBufferTimeoutId = setTimeout(() => {
      this._clearLinuxBuffer()
    }, CanvasPointer.maxHighResBufferTime)
  }

  /**
   * Checks if two deltaY values follow a Linux wheel pattern (divisibility).
   * @param deltaY1 The first deltaY value
   * @param deltaY2 The second deltaY value
   */
  private _isLinuxWheelPattern(deltaY1: number, deltaY2: number): boolean {
    const absolute1 = Math.abs(deltaY1)
    const absolute2 = Math.abs(deltaY2)

    if (absolute1 === 0 || absolute2 === 0) return false
    if (absolute1 === absolute2) return true

    // Check if one value is a multiple of the other
    return absolute1 % absolute2 === 0 || absolute2 % absolute1 === 0
  }

  /**
   * Resets the state of this {@link CanvasPointer} instance.
   *
   * The {@link finally} callback is first executed, then all callbacks and intra-click
   * state is cleared.
   */
  reset(): void {
    // The setter executes the callback before clearing it
    this.finally = undefined
    delete this.onClick
    delete this.onDoubleClick
    delete this.onDragStart
    delete this.onDrag
    delete this.onDragEnd

    this.isDown = false
    this.isDouble = false
    this.dragStarted = false
    this.resizeDirection = undefined

    if (this.clearEventsOnReset) {
      this.eDown = undefined
      this.eMove = undefined
      this.eUp = undefined
    }

    const { element, pointerId } = this
    this.pointerId = undefined
    if (typeof pointerId === 'number' && element.hasPointerCapture(pointerId)) {
      element.releasePointerCapture(pointerId)
    }
  }
}
