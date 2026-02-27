<script setup lang="ts">
import type { SelectTriggerProps } from 'reka-ui'
import { SelectIcon, SelectTrigger } from 'reka-ui'
import type { HTMLAttributes } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const { class: className, ...restProps } = defineProps<
  SelectTriggerProps & { class?: HTMLAttributes['class'] }
>()
</script>

<template>
  <SelectTrigger
    v-bind="restProps"
    :class="
      cn(
        'flex h-10 w-full cursor-pointer select-none items-center justify-between',
        'rounded-lg px-4 py-2 text-sm',
        'bg-secondary-background text-base-foreground',
        'border-[2.5px] border-solid border-transparent',
        'transition-all duration-200 ease-in-out',
        'focus:border-node-component-border focus:outline-none',
        'data-[placeholder]:text-muted-foreground',
        'disabled:cursor-not-allowed disabled:opacity-60',
        '[&>span]:truncate',
        className
      )
    "
  >
    <slot />
    <SelectIcon as-child>
      <i class="icon-[lucide--chevron-down] shrink-0 text-muted-foreground" />
    </SelectIcon>
  </SelectTrigger>
</template>
