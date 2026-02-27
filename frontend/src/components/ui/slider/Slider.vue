<script setup lang="ts">
import { reactiveOmit } from '@vueuse/core'
import type { SliderRootEmits, SliderRootProps } from 'reka-ui'
import {
  SliderRange,
  SliderRoot,
  SliderThumb,
  SliderTrack,
  useForwardPropsEmits
} from 'reka-ui'
import { ref } from 'vue'
import type { HTMLAttributes } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const props = defineProps<
  // eslint-disable-next-line vue/no-unused-properties
  SliderRootProps & { class?: HTMLAttributes['class'] }
>()

const pressed = ref(false)
const setPressed = (val: boolean) => {
  pressed.value = val
}

const emits = defineEmits<SliderRootEmits>()

const delegatedProps = reactiveOmit(props, 'class')

const forwarded = useForwardPropsEmits(delegatedProps, emits)
</script>

<template>
  <SliderRoot
    v-slot="{ modelValue }"
    data-slot="slider"
    :class="
      cn(
        'relative flex w-full touch-none items-center select-none data-[disabled]:opacity-50',
        'data-[orientation=vertical]:h-full data-[orientation=vertical]:min-h-44 data-[orientation=vertical]:w-auto data-[orientation=vertical]:flex-col',
        props.class
      )
    "
    v-bind="forwarded"
    @slide-start="() => setPressed(true)"
    @slide-move="() => setPressed(true)"
    @slide-end="() => setPressed(false)"
  >
    <SliderTrack
      data-slot="slider-track"
      :class="
        cn(
          'bg-node-stroke relative grow overflow-hidden rounded-full',
          'cursor-pointer overflow-visible',
          `before:absolute before:-inset-2 before:block before:bg-transparent`,
          'data-[orientation=horizontal]:h-0.5 data-[orientation=horizontal]:w-full',
          'data-[orientation=vertical]:h-full data-[orientation=vertical]:w-0.5'
        )
      "
    >
      <SliderRange
        data-slot="slider-range"
        class="absolute bg-node-component-surface-highlight data-[orientation=horizontal]:h-full data-[orientation=vertical]:w-full"
      />
    </SliderTrack>

    <SliderThumb
      v-for="(_, key) in modelValue"
      :key="key"
      data-slot="slider-thumb"
      :class="
        cn(
          'bg-node-component-surface-highlight ring-node-component-surface-selected block size-3.5 shrink-0 rounded-full shadow-sm transition-[color,box-shadow]',
          'cursor-grab',
          'before:absolute before:-inset-1 before:block before:bg-transparent before:rounded-full',
          'hover:ring-2 focus-visible:ring-2 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50',
          { 'cursor-grabbing': pressed }
        )
      "
    />
  </SliderRoot>
</template>
