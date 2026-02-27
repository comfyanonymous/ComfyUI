import { cn } from '@/utils/tailwindUtil'

export const WidgetInputBaseClass = cn([
  // Background
  'not-disabled:bg-component-node-widget-background',
  'not-disabled:text-component-node-foreground',
  // Outline
  'border-none',
  // Rounded
  'rounded-lg'
])
