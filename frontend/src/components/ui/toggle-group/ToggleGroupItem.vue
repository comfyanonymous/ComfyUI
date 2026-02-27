<script setup lang="ts">
import type { ToggleGroupItemProps } from 'reka-ui'
import { ToggleGroupItem, useForwardProps } from 'reka-ui'
import type { HTMLAttributes } from 'vue'
import { computed, inject, ref } from 'vue'

import { cn } from '@/utils/tailwindUtil'

import type { ToggleGroupItemVariants } from './toggleGroup.variants'
import {
  toggleGroupItemVariants,
  toggleGroupVariantKey
} from './toggleGroup.variants'

interface Props extends ToggleGroupItemProps {
  class?: HTMLAttributes['class']
  variant?: ToggleGroupItemVariants['variant']
  size?: ToggleGroupItemVariants['size']
}

const {
  class: className,
  variant,
  size = 'default',
  ...restProps
} = defineProps<Props>()

const contextVariant = inject(toggleGroupVariantKey, ref('default'))

const forwardedProps = useForwardProps(restProps)

const resolvedVariant = computed(
  () => variant ?? contextVariant.value ?? 'default'
)
</script>

<template>
  <ToggleGroupItem
    v-bind="forwardedProps"
    :class="
      cn(
        toggleGroupItemVariants({ variant: resolvedVariant, size }),
        'flex-1 min-w-0 truncate',
        className
      )
    "
  >
    <slot />
  </ToggleGroupItem>
</template>
