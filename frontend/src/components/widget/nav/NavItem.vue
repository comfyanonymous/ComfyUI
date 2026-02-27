<template>
  <div
    v-tooltip.right="{
      value: tooltipText,
      disabled: !isOverflowing,
      pt: { text: { class: 'w-max whitespace-nowrap' } }
    }"
    class="flex cursor-pointer select-none items-center-safe gap-2 rounded-md px-4 py-3 text-sm transition-colors text-base-foreground"
    :class="
      active
        ? 'bg-interface-menu-component-surface-selected'
        : 'hover:bg-interface-menu-component-surface-hovered'
    "
    role="button"
    @mouseenter="checkOverflow"
    @click="onClick"
  >
    <NavIcon v-if="icon" :icon="icon" />
    <i v-else class="text-neutral icon-[lucide--folder] text-xs shrink-0" />
    <span ref="textRef" class="min-w-0 truncate">
      <slot />
    </span>
    <StatusBadge
      v-if="badge !== undefined"
      :label="String(badge)"
      severity="contrast"
      variant="circle"
      class="ml-auto"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

import StatusBadge from '@/components/common/StatusBadge.vue'
import type { NavItemData } from '@/types/navTypes'

import NavIcon from './NavIcon.vue'

const { icon, badge, active, onClick } = defineProps<{
  icon: NavItemData['icon']
  badge?: NavItemData['badge']
  active?: boolean
  onClick: () => void
}>()

const textRef = ref<HTMLElement | null>(null)
const isOverflowing = ref(false)

const checkOverflow = () => {
  if (!textRef.value) return
  isOverflowing.value =
    textRef.value.scrollWidth > textRef.value.clientWidth + 1
}

const tooltipText = computed(() => textRef.value?.textContent ?? '')
</script>
