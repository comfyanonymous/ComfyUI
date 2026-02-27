<template>
  <div :class="containerClasses">
    <slot name="top"></slot>
    <slot name="bottom"></slot>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const {
  size = 'regular',
  variant = 'default',
  rounded = 'md',
  customAspectRatio,
  hasBorder = true,
  hasBackground = true,
  hasShadow = true,
  hasCursor = true,
  class: customClass = ''
} = defineProps<{
  size?: 'mini' | 'compact' | 'regular' | 'portrait' | 'tall'
  variant?: 'default' | 'ghost' | 'outline'
  rounded?: 'none' | 'md' | 'lg' | 'xl'
  customAspectRatio?: string
  hasBorder?: boolean
  hasBackground?: boolean
  hasShadow?: boolean
  hasCursor?: boolean
  class?: string
}>()

// Base structure classes
const structureClasses = 'flex flex-col overflow-hidden'

// Rounded corners
const roundedClasses = {
  none: 'rounded-none',
  md: 'rounded',
  lg: 'rounded-lg',
  xl: 'rounded-xl'
} as const

const containerClasses = computed(() => {
  // Variant styles
  const variantClasses = {
    default: cn(
      hasBackground && 'bg-modal-card-background',
      hasBorder && 'border border-border-default',
      hasShadow && 'shadow-sm',
      hasCursor && 'cursor-pointer'
    ),
    ghost: cn(
      hasCursor && 'cursor-pointer',
      'p-2 transition-colors duration-200'
    ),
    outline: cn(
      hasBorder && 'border-2 border-border-subtle',
      hasCursor && 'cursor-pointer',
      'hover:border-border-subtle/50 transition-colors'
    )
  }

  // Size/aspect ratio
  const aspectRatio = customAspectRatio
    ? `aspect-[${customAspectRatio}]`
    : {
        mini: 'aspect-100/120',
        compact: 'aspect-240/311',
        regular: 'aspect-256/308',
        portrait: 'aspect-256/325',
        tall: 'aspect-256/353'
      }[size]

  return cn(
    structureClasses,
    roundedClasses[rounded],
    variantClasses[variant],
    aspectRatio,
    customClass
  )
})
</script>
