<script setup lang="ts">
import type { SelectLabelProps } from 'reka-ui'
import { SelectLabel } from 'reka-ui'
import type { HTMLAttributes } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const { class: className, ...restProps } = defineProps<
  SelectLabelProps & { class?: HTMLAttributes['class'] }
>()
</script>

<template>
  <SelectLabel
    v-bind="restProps"
    :class="
      cn(
        'px-3 py-2 text-xs uppercase tracking-wide text-muted-foreground',
        className
      )
    "
  >
    <slot />
  </SelectLabel>
</template>
