<script setup lang="ts">
import type { SimplifiedWidget } from '@/types/simplifiedWidget'
import { useHideLayoutField } from '@/types/widgetTypes'
import { cn } from '@/utils/tailwindUtil'

const { rootClass } = defineProps<{
  widget: Pick<
    SimplifiedWidget<string | number | undefined>,
    'name' | 'label' | 'borderStyle'
  >
  rootClass?: string
}>()

const hideLayoutField = useHideLayoutField()
</script>

<template>
  <div
    :class="
      cn(
        'grid grid-cols-subgrid min-w-0 justify-between gap-1 text-node-component-slot-text',
        rootClass
      )
    "
  >
    <div v-if="!hideLayoutField" class="truncate content-center-safe">
      <template v-if="widget.name">
        {{ widget.label || widget.name }}
      </template>
    </div>
    <!-- basis-full grow -->
    <div class="relative min-w-0 flex-1">
      <div
        :class="
          cn(
            'cursor-default min-w-0 rounded-lg focus-within:ring focus-within:ring-component-node-widget-background-highlighted transition-all',
            widget.borderStyle
          )
        "
        @pointerdown.stop
        @pointermove.stop
        @pointerup.stop
      >
        <slot />
      </div>
    </div>
  </div>
</template>
