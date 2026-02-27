<script setup lang="ts">
import { cn } from '@comfyorg/tailwind-utils'

defineProps<{
  label: string
  tooltip?: string
  singleline?: boolean
}>()
</script>

<template>
  <div
    :class="
      cn('flex gap-2', singleline ? 'items-center justify-between' : 'flex-col')
    "
  >
    <span
      v-tooltip.left="
        tooltip
          ? {
              value: tooltip,
              showDelay: 300
            }
          : null
      "
      :class="
        cn(
          'text-sm text-muted-foreground truncate group',
          tooltip ? 'cursor-help' : '',
          singleline ? 'flex-1' : ''
        )
      "
    >
      {{ label }}

      <i
        v-if="tooltip"
        class="icon-[lucide--info] ml-0.5 size-3 relative top-[1px] group-hover:text-primary"
      />
    </span>
    <slot />
  </div>
</template>
