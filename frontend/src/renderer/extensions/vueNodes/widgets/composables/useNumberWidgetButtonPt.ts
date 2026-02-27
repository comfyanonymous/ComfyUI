import { cn } from '@comfyorg/tailwind-utils'

const sharedButtonClasses = cn(
  'inline-flex items-center justify-center border-0 bg-transparent text-inherit transition-colors duration-150 ease-in-out ',
  'hover:bg-node-component-surface-hovered active:bg-node-component-surface-selected',
  'disabled:bg-node-component-disabled disabled:text-node-icon-disabled disabled:cursor-not-allowed'
)

export function useNumberWidgetButtonPt(options?: {
  roundedLeft?: boolean
  roundedRight?: boolean
}) {
  const { roundedLeft = false, roundedRight = false } = options ?? {}

  const increment = cn(sharedButtonClasses, roundedRight && 'rounded-r-lg')
  const decrement = cn(sharedButtonClasses, roundedLeft && 'rounded-l-lg')

  return {
    incrementButton: {
      class: increment
    },
    decrementButton: {
      class: decrement
    }
  }
}
