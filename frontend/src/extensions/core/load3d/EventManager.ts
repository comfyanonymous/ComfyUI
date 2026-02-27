import { type EventCallback, type EventManagerInterface } from './interfaces'

export class EventManager implements EventManagerInterface {
  private listeners: Record<string, EventCallback[]> = {}

  addEventListener<T>(event: string, callback: EventCallback<T>): void {
    if (!this.listeners[event]) {
      this.listeners[event] = []
    }
    this.listeners[event].push(callback as EventCallback)
  }

  removeEventListener<T>(event: string, callback: EventCallback<T>): void {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter(
        (cb) => cb !== callback
      )
    }
  }

  emitEvent<T>(event: string, data: T): void {
    if (this.listeners[event]) {
      this.listeners[event].forEach((callback) => callback(data))
    }
  }
}
