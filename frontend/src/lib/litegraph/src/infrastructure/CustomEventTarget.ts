import type { NeverNever, PickNevers } from '@/lib/litegraph/src/types/utility'

type EventListeners<T> = {
  readonly [K in keyof T]:
    | ((this: EventTarget, ev: CustomEvent<T[K]>) => unknown)
    | EventListenerObject
    | null
}

/**
 * Has strongly-typed overrides of {@link EventTarget.addEventListener} and {@link EventTarget.removeEventListener}.
 */
export interface ICustomEventTarget<
  EventMap extends Record<Keys, unknown>,
  Keys extends keyof EventMap & string = keyof EventMap & string
> {
  addEventListener<K extends Keys>(
    type: K,
    listener: EventListeners<EventMap>[K],
    options?: boolean | AddEventListenerOptions
  ): void

  removeEventListener<K extends Keys>(
    type: K,
    listener: EventListeners<EventMap>[K],
    options?: boolean | EventListenerOptions
  ): void

  /** @deprecated Use {@link dispatch}. */
  dispatchEvent(event: never): boolean
}

/**
 * Capable of dispatching strongly-typed events via {@link dispatch}.
 * Overloads are used to ensure detail param is correctly optional.
 */
export interface CustomEventDispatcher<
  EventMap extends Record<Keys, unknown>,
  Keys extends keyof EventMap & string = keyof EventMap & string
> {
  dispatch<T extends keyof NeverNever<EventMap>>(
    type: T,
    detail: EventMap[T]
  ): boolean
  dispatch<T extends keyof PickNevers<EventMap>>(type: T): boolean
}

/**
 * A strongly-typed, custom {@link EventTarget} that can dispatch and listen for events.
 *
 * 1. Define an event map
 *    ```ts
 *    export interface CustomEventMap {
 *      "my-event": { message: string }
 *      "simple-event": never
 *    }
 *    ```
 *
 * 2. Create an event emitter
 *    ```ts
 *    // By subclassing
 *    class MyClass extends CustomEventTarget<CustomEventMap> {
 *      // ...
 *    }
 *
 *    // Or simply create an instance:
 *    const events = new CustomEventTarget<CustomEventMap>()
 *    ```
 *
 * 3. Dispatch events
 *    ```ts
 *    // Extended class
 *    const myClass = new MyClass()
 *    myClass.dispatch("my-event", { message: "Hello, world!" })
 *    myClass.dispatch("simple-event")
 *
 *    // Instance
 *    const events = new CustomEventTarget<CustomEventMap>()
 *    events.dispatch("my-event", { message: "Hello, world!" })
 *    events.dispatch("simple-event")
 *    ```
 */
export class CustomEventTarget<
  EventMap extends Record<Keys, unknown>,
  Keys extends keyof EventMap & string = keyof EventMap & string
>
  extends EventTarget
  implements ICustomEventTarget<EventMap, Keys>
{
  /**
   * Type-safe event dispatching.
   * @see {@link EventTarget.dispatchEvent}
   * @param type Name of the event to dispatch
   * @param detail A custom object to send with the event
   * @returns `true` if the event was dispatched successfully, otherwise `false`.
   */
  dispatch<T extends keyof NeverNever<EventMap>>(
    type: T,
    detail: EventMap[T]
  ): boolean
  dispatch<T extends keyof PickNevers<EventMap>>(type: T): boolean
  dispatch<T extends keyof EventMap>(type: T, detail?: EventMap[T]) {
    const event = new CustomEvent(type as string, { detail, cancelable: true })
    return super.dispatchEvent(event)
  }

  override addEventListener<K extends Keys>(
    type: K,
    listener: EventListeners<EventMap>[K],
    options?: boolean | AddEventListenerOptions
  ): void {
    // Assertion: Contravariance on CustomEvent => Event
    super.addEventListener(type as string, listener as EventListener, options)
  }

  override removeEventListener<K extends Keys>(
    type: K,
    listener: EventListeners<EventMap>[K],
    options?: boolean | EventListenerOptions
  ): void {
    // Assertion: Contravariance on CustomEvent => Event
    super.removeEventListener(
      type as string,
      listener as EventListener,
      options
    )
  }

  /** @deprecated Use {@link dispatch}. */
  override dispatchEvent(event: never): boolean {
    return super.dispatchEvent(event)
  }
}
