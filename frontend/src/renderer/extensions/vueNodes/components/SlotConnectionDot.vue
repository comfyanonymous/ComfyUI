<script setup lang="ts">
import { computed, useTemplateRef } from 'vue'

import { getSlotColor } from '@/constants/slotColors'
import type { INodeSlot } from '@/lib/litegraph/src/litegraph'
import { RenderShape } from '@/lib/litegraph/src/types/globalEnums'
import { cn } from '@/utils/tailwindUtil'
import type { ClassValue } from '@/utils/tailwindUtil'

const props = defineProps<{
  slotData?: INodeSlot
  class?: ClassValue
  hasError?: boolean
  multi?: boolean
}>()

const clipPath = computed(() => {
  switch (props.slotData?.shape) {
    case 6:
      return 'url(#square)'
    case 7:
      return 'url(#hollow)'
    default:
      return undefined
  }
})

const slotElRef = useTemplateRef('slot-el')

const types = computed(() => {
  if (props.hasError) return ['var(--color-error)']
  //TODO Support connected/disconnected colors?
  if (!props.slotData) return [getSlotColor()]
  if (props.slotData.type === '*') return ['']
  const typesSet = new Set(
    `${props.slotData.type}`.split(',').map(getSlotColor)
  )
  return [...typesSet].slice(0, 3)
})

defineExpose({
  slotElRef
})

const isListShape = computed(() => props.slotData?.shape === RenderShape.GRID)

const slotClass = computed(() =>
  cn(
    'bg-slate-300 slot-dot',
    isListShape.value ? 'rounded-[1px]' : 'rounded-full',
    'transition-all duration-150',
    'border border-solid border-node-component-slot-dot-outline',
    props.multi
      ? 'w-3 h-6'
      : 'size-3 cursor-crosshair group-hover/slot:[--node-component-slot-dot-outline-opacity-mult:5] group-hover/slot:scale-125'
  )
)
</script>

<template>
  <div
    :class="
      cn(
        'after:absolute after:inset-y-0 after:w-5/2 relative size-6 flex items-center justify-center group/slot',
        props.class
      )
    "
  >
    <div
      v-if="types.length === 1 && (slotData?.shape == undefined || isListShape)"
      ref="slot-el"
      :style="{ backgroundColor: types.length === 1 ? types[0] : undefined }"
      :class="slotClass"
    />
    <svg
      v-else
      ref="slot-el"
      :class="slotClass"
      viewBox="0 0 100 100"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <clipPath id="square">
          <rect x="20" y="20" width="60" height="60" />
        </clipPath>
        <clipPath id="hollow">
          <path
            d="M-50 50
           A100 100 0 0 1 150 50
           A100 100 0 0 1 -50 50
           M30 50
           A20 20 0 0 0 70 50
           A20 20 0 0 0 30 50"
          />
        </clipPath>
      </defs>
      <circle
        v-if="types.length === 1"
        :clip-path
        cx="50"
        cy="50"
        r="50"
        :fill="types[0]"
      />
      <g v-else-if="types.length === 2" :clip-path stroke-width="4">
        <path d="M0 50 A 50 50 0 0 1 100 50" :fill="types[0]" />
        <path d="M100 50 A 50 50 0 0 1 0 50" :fill="types[1]" />
        <path d="M0 50L100 50" stroke="var(--inner-stroke, black)" />
        <path
          d="M50 2A48 48 0 0 1 50 98A48 48 0 0 1 50 2"
          fill="transparent"
          stroke="var(--outer-stroke, transparent)"
        />
      </g>
      <g v-else :clip-path stroke-width="4">
        <path d="M0 50A50 50 0 0 0 75 93L50 50" :fill="types[0]" />
        <path d="M75 93A50 50 0 0 0 75 7L50 50" :fill="types[1]" />
        <path d="M75 7A50 50 0 0 0 0 50L50 50" :fill="types[2]" />
        <path
          d="M50 50L0 50M50 50L75 93M50 50L75 7"
          stroke="var(--inner-stroke, black)"
        />
        <path
          d="M50 2A48 48 0 0 1 50 98A48 48 0 0 1 50 2"
          fill="transparent"
          stroke="var(--outer-stroke, transparent)"
        />
      </g>
    </svg>
  </div>
</template>
