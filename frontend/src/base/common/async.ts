/**
 * Cross-browser async utilities for scheduling tasks during browser idle time
 * with proper fallbacks for browsers that don't support requestIdleCallback.
 *
 * Implementation based on:
 * https://github.com/microsoft/vscode/blob/main/src/vs/base/common/async.ts
 */

interface IdleDeadline {
  didTimeout: boolean
  timeRemaining(): number
}

interface IDisposable {
  dispose(): void
}
type GlobalWindow = typeof globalThis

/**
 * Internal implementation function that handles the actual scheduling logic.
 * Uses feature detection to determine whether to use native requestIdleCallback
 * or fall back to setTimeout-based implementation.
 */
let _runWhenIdle: (
  targetWindow: GlobalWindow,
  callback: (idle: IdleDeadline) => void,
  timeout?: number
) => IDisposable

/**
 * Execute the callback during the next browser idle period.
 * Falls back to setTimeout-based scheduling in browsers without native support.
 */
export let runWhenGlobalIdle: (
    callback: (idle: IdleDeadline) => void,
    timeout?: number
  ) => IDisposable

  // Self-invoking function to set up the idle callback implementation
;(function () {
  const safeGlobal: GlobalWindow = globalThis as GlobalWindow

  if (
    typeof safeGlobal.requestIdleCallback !== 'function' ||
    typeof safeGlobal.cancelIdleCallback !== 'function'
  ) {
    // Fallback implementation for browsers without native support (e.g., Safari)
    _runWhenIdle = (_targetWindow, runner, _timeout?) => {
      setTimeout(() => {
        if (disposed) {
          return
        }

        // Simulate IdleDeadline - give 15ms window (one frame at ~64fps)
        const end = Date.now() + 15
        const deadline: IdleDeadline = {
          didTimeout: true,
          timeRemaining() {
            return Math.max(0, end - Date.now())
          }
        }

        runner(Object.freeze(deadline))
      })

      let disposed = false
      return {
        dispose() {
          if (disposed) {
            return
          }
          disposed = true
        }
      }
    }
  } else {
    // Native requestIdleCallback implementation
    _runWhenIdle = (targetWindow: typeof safeGlobal, runner, timeout?) => {
      const handle: number = targetWindow.requestIdleCallback(
        runner,
        typeof timeout === 'number' ? { timeout } : undefined
      )

      let disposed = false
      return {
        dispose() {
          if (disposed) {
            return
          }
          disposed = true
          targetWindow.cancelIdleCallback(handle)
        }
      }
    }
  }

  runWhenGlobalIdle = (runner, timeout) =>
    _runWhenIdle(globalThis, runner, timeout)
})()
