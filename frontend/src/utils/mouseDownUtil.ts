import { useEventListener } from '@vueuse/core'

export const whileMouseDown = (
  elementOrEvent: HTMLElement | Event,
  callback: (iteration: number) => void,
  interval: number = 30
) => {
  const element =
    elementOrEvent instanceof HTMLElement
      ? elementOrEvent
      : (elementOrEvent.target as HTMLElement)

  let iteration = 0

  const intervalId = setInterval(() => {
    callback(iteration++)
  }, interval)

  const dispose = () => {
    clearInterval(intervalId)
    disposeGlobal()
    disposeLocal()
  }

  // Listen for mouseup globally to catch cases where user drags out of element
  const disposeGlobal = useEventListener(document, 'mouseup', dispose)
  const disposeLocal = useEventListener(element, 'mouseup', dispose)

  return {
    dispose: dispose
  }
}
