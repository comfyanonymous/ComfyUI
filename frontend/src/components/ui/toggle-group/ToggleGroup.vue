<script setup lang="ts">
import type { ToggleGroupRootEmits, ToggleGroupRootProps } from 'reka-ui'
import { ToggleGroupRoot, useForwardPropsEmits } from 'reka-ui'
import type { HTMLAttributes } from 'vue'
import { provide, toRef } from 'vue'

import { cn } from '@/utils/tailwindUtil'

import type { ToggleGroupVariants } from './toggleGroup.variants'
import {
  toggleGroupVariantKey,
  toggleGroupVariants
} from './toggleGroup.variants'

interface Props extends ToggleGroupRootProps {
  class?: HTMLAttributes['class']
  variant?: ToggleGroupVariants['variant']
}

const {
  class: className,
  variant = 'default',
  ...restProps
} = defineProps<Props>()

const emits = defineEmits<ToggleGroupRootEmits>()

const forwarded = useForwardPropsEmits(restProps, emits)

provide(
  toggleGroupVariantKey,
  toRef(() => variant)
)
</script>

<template>
  <ToggleGroupRoot
    v-bind="forwarded"
    :class="cn(toggleGroupVariants({ variant }), className)"
  >
    <slot />
  </ToggleGroupRoot>
</template>
