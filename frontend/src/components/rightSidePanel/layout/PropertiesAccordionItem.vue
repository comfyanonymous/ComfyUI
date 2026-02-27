<script lang="ts" setup>
import { computed } from 'vue'

import { cn } from '@/utils/tailwindUtil'

import TransitionCollapse from './TransitionCollapse.vue'

const {
  disabled,
  label,
  enableEmptyState,
  tooltip,
  size = 'default',
  class: className
} = defineProps<{
  disabled?: boolean
  label?: string
  enableEmptyState?: boolean
  tooltip?: string
  size?: 'default' | 'lg'
  class?: string
}>()

const isCollapse = defineModel<boolean>('collapse', { default: false })

const isExpanded = computed(() => !isCollapse.value && !disabled)

const tooltipConfig = computed(() => {
  if (!tooltip) return undefined
  return { value: tooltip, showDelay: 1000 }
})
</script>

<template>
  <div :class="cn('flex flex-col bg-comfy-menu-bg', className)">
    <div
      class="sticky top-0 z-10 flex items-center justify-between backdrop-blur-xl bg-inherit"
    >
      <button
        v-tooltip="tooltipConfig"
        type="button"
        :class="
          cn(
            'group bg-transparent border-0 outline-0 ring-0 w-full text-left flex items-center justify-between pl-4 pr-3',
            size === 'lg' ? 'min-h-16' : 'min-h-12',
            !disabled && 'cursor-pointer'
          )
        "
        :disabled="disabled"
        @click="isCollapse = !isCollapse"
      >
        <span class="text-sm font-semibold line-clamp-2 flex-1">
          <slot name="label">
            {{ label }}
          </slot>
        </span>

        <i
          :class="
            cn(
              'text-muted-foreground group-hover:text-base-foreground group-has-[.subbutton:hover]:text-muted-foreground group-focus:text-base-foreground icon-[lucide--chevron-up] size-4 transition-all',
              isCollapse && '-rotate-180',
              disabled && 'opacity-0'
            )
          "
        />
      </button>
    </div>
    <TransitionCollapse>
      <div v-if="isExpanded" class="pb-4">
        <slot />
      </div>
      <slot v-else-if="enableEmptyState && disabled" name="empty">
        <div>
          {{ $t('g.empty') }}
        </div>
      </slot>
    </TransitionCollapse>
  </div>
</template>
