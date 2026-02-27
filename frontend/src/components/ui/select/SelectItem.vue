<script setup lang="ts">
import type { SelectItemProps } from 'reka-ui'
import { SelectItem, SelectItemIndicator, SelectItemText } from 'reka-ui'
import type { HTMLAttributes } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const { class: className, ...restProps } = defineProps<
  SelectItemProps & { class?: HTMLAttributes['class'] }
>()
</script>

<template>
  <SelectItem
    v-bind="restProps"
    :class="
      cn(
        'relative flex w-full cursor-pointer select-none items-center justify-between',
        'gap-3 rounded px-2 py-3 text-sm outline-none',
        'hover:bg-secondary-background-hover',
        'focus:bg-secondary-background-hover',
        'data-[state=checked]:bg-secondary-background-selected',
        'data-[state=checked]:hover:bg-secondary-background-selected',
        'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
        className
      )
    "
  >
    <SelectItemText class="truncate">
      <slot />
    </SelectItemText>

    <SelectItemIndicator class="flex shrink-0 items-center justify-center">
      <i class="icon-[lucide--check] text-base-foreground" aria-hidden="true" />
    </SelectItemIndicator>
  </SelectItem>
</template>
