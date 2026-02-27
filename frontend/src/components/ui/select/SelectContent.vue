<script setup lang="ts">
import type { SelectContentEmits, SelectContentProps } from 'reka-ui'
import {
  SelectContent,
  SelectPortal,
  SelectViewport,
  useForwardPropsEmits
} from 'reka-ui'
import type { HTMLAttributes } from 'vue'
import { computed } from 'vue'

import { cn } from '@/utils/tailwindUtil'

import SelectScrollDownButton from './SelectScrollDownButton.vue'
import SelectScrollUpButton from './SelectScrollUpButton.vue'

defineOptions({
  inheritAttrs: false
})

const {
  position = 'popper',
  // Safari has issues with click events on portaled content inside dialogs.
  // Set disablePortal to true when using Select inside a Dialog on Safari.
  // See: https://github.com/chakra-ui/ark/issues/1782
  disablePortal = false,
  class: className,
  ...restProps
} = defineProps<
  SelectContentProps & {
    class?: HTMLAttributes['class']
    disablePortal?: boolean
  }
>()
const emits = defineEmits<SelectContentEmits>()

const delegatedProps = computed(() => ({
  position,
  ...restProps
}))

const forwarded = useForwardPropsEmits(delegatedProps, emits)
</script>

<template>
  <SelectPortal :disabled="disablePortal">
    <SelectContent
      v-bind="{ ...forwarded, ...$attrs }"
      :class="
        cn(
          'relative z-3000 max-h-96 min-w-32 overflow-hidden',
          'mt-2 rounded-lg p-2',
          'bg-base-background text-base-foreground',
          'border border-solid border-border-default',
          'shadow-md',
          'data-[state=open]:animate-in data-[state=closed]:animate-out',
          'data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
          'data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95',
          'data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2',
          'data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
          position === 'popper' &&
            'data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1',
          className
        )
      "
    >
      <SelectScrollUpButton />
      <SelectViewport
        :class="
          cn(
            'scrollbar-custom flex flex-col gap-0',
            position === 'popper' &&
              'h-(--reka-select-trigger-height) w-full min-w-(--reka-select-trigger-width)'
          )
        "
      >
        <slot />
      </SelectViewport>
      <SelectScrollDownButton />
    </SelectContent>
  </SelectPortal>
</template>
