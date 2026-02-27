<template>
  <!-- Icon-only mode with Popover -->
  <div
    v-if="displayMode === 'icon-only'"
    class="relative inline-flex h-full shrink-0 items-center justify-center px-2"
    :class="clickableClasses"
    :style="menuBackgroundStyle"
    @click="togglePopover"
  >
    <i
      v-if="iconClass"
      :class="['shrink-0 text-base', iconClass, iconColorClass]"
    />
    <div
      v-else-if="badge.label"
      class="shrink-0 rounded-full px-1.5 py-0.5 text-xxxs font-semibold"
      :class="labelClasses"
    >
      {{ badge.label }}
    </div>
    <div v-else class="size-2 shrink-0 rounded-full" :class="dotClasses" />
    <Popover
      ref="popover"
      append-to="body"
      :auto-z-index="true"
      :base-z-index="1000"
      :dismissable="true"
      :close-on-escape="true"
      unstyled
      :pt="popoverPt"
    >
      <div class="flex max-w-xs min-w-40 flex-col gap-2 p-3">
        <div
          v-if="badge.label"
          class="w-fit rounded-full px-1.5 py-0.5 text-xxxs font-semibold"
          :class="labelClasses"
        >
          {{ badge.label }}
        </div>
        <div class="text-sm font-inter">{{ badge.text }}</div>
        <div v-if="badge.tooltip" class="text-xs">
          {{ badge.tooltip }}
        </div>
      </div>
    </Popover>
  </div>

  <!-- Compact mode: Icon + Label only with Popover -->
  <div
    v-else-if="displayMode === 'compact'"
    class="relative inline-flex h-full"
    :style="menuBackgroundStyle"
  >
    <div
      class="flex h-full shrink-0 items-center gap-2 whitespace-nowrap"
      :class="[
        { 'flex-row-reverse': reverseOrder },
        noPadding ? '' : 'px-3',
        clickableClasses
      ]"
      @click="togglePopover"
    >
      <i
        v-if="iconClass"
        :class="['shrink-0 text-base', iconClass, iconColorClass]"
      />
      <div
        v-if="badge.label"
        class="shrink-0 rounded-full px-1.5 py-0.5 text-xxxs font-semibold"
        :class="labelClasses"
      >
        {{ badge.label }}
      </div>
    </div>
    <Popover
      ref="popover"
      append-to="body"
      :auto-z-index="true"
      :base-z-index="1000"
      :dismissable="true"
      :close-on-escape="true"
      unstyled
      :pt="popoverPt"
    >
      <div class="flex max-w-xs min-w-40 flex-col gap-2 p-3">
        <div
          v-if="badge.label"
          class="w-fit rounded-full px-1.5 py-0.5 text-xxxs font-semibold"
          :class="labelClasses"
        >
          {{ badge.label }}
        </div>
        <div class="text-sm font-inter">{{ badge.text }}</div>
        <div v-if="badge.tooltip" class="text-xs">
          {{ badge.tooltip }}
        </div>
      </div>
    </Popover>
  </div>

  <!-- Full mode: Icon + Label + Text -->
  <div
    v-else
    v-tooltip="badge.tooltip"
    class="flex h-full shrink-0 items-center gap-2 whitespace-nowrap"
    :class="[{ 'flex-row-reverse': reverseOrder }, noPadding ? '' : 'px-3']"
    :style="menuBackgroundStyle"
  >
    <i
      v-if="iconClass"
      :class="['shrink-0 text-base', iconClass, iconColorClass]"
    />
    <div
      v-if="badge.label"
      class="shrink-0 rounded-full px-1.5 py-0.5 text-xxxs font-semibold"
      :class="labelClasses"
    >
      {{ badge.label }}
    </div>
    <div class="font-inter text-sm" :class="textClasses">
      {{ badge.text }}
    </div>
  </div>
</template>
<script setup lang="ts">
import Popover from 'primevue/popover'
import { computed, ref } from 'vue'

import type { TopbarBadge } from '@/types/comfy'
import { cn } from '@/utils/tailwindUtil'

const {
  badge,
  displayMode = 'full',
  reverseOrder = false,
  noPadding = false,
  backgroundColor = 'var(--comfy-menu-bg)'
} = defineProps<{
  badge: TopbarBadge
  displayMode?: 'full' | 'compact' | 'icon-only'
  reverseOrder?: boolean
  noPadding?: boolean
  backgroundColor?: string
}>()

const popover = ref<InstanceType<typeof Popover>>()

const togglePopover = (event: Event) => {
  popover.value?.toggle(event)
}

const variant = computed(() => badge.variant ?? 'info')

const menuBackgroundStyle = computed(() => ({
  backgroundColor: backgroundColor
}))

const labelClasses = computed(() => {
  switch (variant.value) {
    case 'error':
      return 'bg-danger-100 text-white'
    case 'warning':
      return 'bg-gold-600 text-black'
    case 'info':
    default:
      return 'bg-white text-black'
  }
})

const textClasses = computed(() => {
  switch (variant.value) {
    case 'error':
      return 'text-danger-100'
    case 'warning':
      return 'text-warning-background'
    case 'info':
    default:
      return 'text-text-primary'
  }
})

const iconColorClass = computed(() => textClasses.value)

const iconClass = computed(() => {
  if (badge.icon) {
    return badge.icon
  }
  switch (variant.value) {
    case 'error':
      return 'pi pi-exclamation-circle'
    case 'warning':
      return 'icon-[lucide--triangle-alert]'
    case 'info':
    default:
      return undefined
  }
})

const clickableClasses = 'cursor-pointer transition-opacity hover:opacity-80'

const dotClasses = computed(() => {
  switch (variant.value) {
    case 'error':
      return 'bg-danger-100'
    case 'warning':
      return 'bg-gold-600'
    case 'info':
    default:
      return 'bg-slate-100'
  }
})

const popoverPt = computed(() => ({
  root: {
    class: cn('absolute z-50')
  },
  content: {
    class: cn(
      'mt-1 rounded-lg',
      'bg-base-background',
      'text-base-foreground',
      'shadow-lg',
      'border border-border-default'
    )
  }
}))
</script>
