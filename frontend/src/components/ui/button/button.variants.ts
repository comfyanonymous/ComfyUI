import type { VariantProps } from 'cva'
import { cva } from 'cva'

export const buttonVariants = cva({
  base: 'relative inline-flex items-center justify-center gap-2 cursor-pointer whitespace-nowrap appearance-none border-none rounded-md text-sm font-medium font-inter transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
  variants: {
    variant: {
      secondary:
        'bg-secondary-background text-secondary-foreground hover:bg-secondary-background-hover',
      primary:
        'bg-primary-background text-base-foreground hover:bg-primary-background-hover',
      inverted:
        'bg-base-foreground text-base-background hover:bg-base-foreground/80',
      destructive:
        'bg-destructive-background text-base-foreground hover:bg-destructive-background-hover',
      textonly:
        'text-base-foreground bg-transparent hover:bg-secondary-background-hover',
      'muted-textonly':
        'text-muted-foreground bg-transparent hover:bg-secondary-background-hover',
      'destructive-textonly':
        'text-destructive-background bg-transparent hover:bg-destructive-background/10',
      'overlay-white': 'bg-white text-gray-600 hover:bg-white/90'
    },
    size: {
      sm: 'h-6 rounded-sm px-2 py-1 text-xs',
      md: 'h-8 rounded-lg p-2 text-xs',
      lg: 'h-10 rounded-lg px-4 py-2 text-sm',
      icon: 'size-8',
      'icon-sm': 'size-5 p-0',
      unset: ''
    }
  },

  defaultVariants: {
    variant: 'secondary',
    size: 'md'
  }
})

export type ButtonVariants = VariantProps<typeof buttonVariants>

const variants = [
  'secondary',
  'primary',
  'inverted',
  'destructive',
  'textonly',
  'muted-textonly',
  'destructive-textonly',
  'overlay-white'
] as const satisfies Array<ButtonVariants['variant']>
const sizes = ['sm', 'md', 'lg', 'icon', 'icon-sm'] as const satisfies Array<
  ButtonVariants['size']
>

export const FOR_STORIES = { variants, sizes } as const
