import type { InjectionKey, Ref } from 'vue'

import type { VariantProps } from 'cva'
import { cva } from 'cva'

export const toggleGroupVariantKey: InjectionKey<
  Ref<ToggleGroupItemVariants['variant']>
> = Symbol('toggleGroupVariant')

export const toggleGroupVariants = cva({
  base: 'flex items-center justify-center gap-1',
  variants: {
    variant: {
      default: 'bg-transparent',
      outline: 'bg-transparent'
    }
  },
  defaultVariants: {
    variant: 'default'
  }
})

export const toggleGroupItemVariants = cva({
  base: [
    'inline-flex items-center justify-center rounded',
    'border-none cursor-pointer appearance-none',
    'text-center font-normal',
    'transition-all duration-150 ease-in-out',
    'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
    'disabled:pointer-events-none disabled:opacity-50 disabled:cursor-not-allowed',
    'data-[state=on]:bg-interface-menu-component-surface-selected data-[state=on]:text-text-primary'
  ],
  variants: {
    variant: {
      default:
        'bg-transparent hover:bg-interface-menu-component-surface-selected/50 text-text-secondary',
      outline:
        'border border-border-default bg-transparent hover:bg-secondary-background text-text-secondary'
    },
    size: {
      default: 'h-7 px-3 text-sm',
      sm: 'h-6 px-5 py-[5px] text-xs',
      lg: 'h-9 px-4 text-sm'
    }
  },
  defaultVariants: {
    variant: 'default',
    size: 'default'
  }
})

export type ToggleGroupVariants = VariantProps<typeof toggleGroupVariants>
export type ToggleGroupItemVariants = VariantProps<
  typeof toggleGroupItemVariants
>
