<script setup lang="ts">
import { DropdownMenuItem, DropdownMenuSeparator } from 'reka-ui'
import type { Component } from 'vue'

import OverlayIcon from '@/components/common/OverlayIcon.vue'
import type { WorkflowMenuItem } from '@/types/workflowMenuItem'
import { cn } from '@/utils/tailwindUtil'

const {
  items,
  itemComponent = DropdownMenuItem,
  separatorComponent = DropdownMenuSeparator
} = defineProps<{
  items: WorkflowMenuItem[]
  itemComponent?: Component
  separatorComponent?: Component
}>()
</script>

<template>
  <template v-for="(item, index) in items" :key="index">
    <component
      :is="separatorComponent"
      v-if="item.separator"
      class="border-b w-full border-border-subtle my-1"
    />
    <component
      :is="itemComponent"
      v-else
      :disabled="item.disabled"
      :class="
        cn(
          'flex min-h-6 p-2 items-center gap-2 self-stretch rounded-sm outline-none',
          !item.disabled && item.command && 'cursor-pointer',
          'data-[highlighted]:bg-secondary-background-hover',
          !item.disabled && 'hover:bg-secondary-background-hover',
          'data-[disabled]:opacity-50 data-[disabled]:cursor-default'
        )
      "
      @select="() => item.command?.()"
    >
      <OverlayIcon v-if="item.overlayIcon" v-bind="item.overlayIcon" />
      <i v-else-if="item.icon" :class="item.icon" />
      <span class="flex-1">{{ item.label }}</span>
      <span
        v-if="item.badge"
        class="rounded-full uppercase ml-3 flex items-center gap-1 bg-[var(--primary-background)] px-1.5 py-0.5 text-xxs text-base-foreground"
      >
        {{ item.badge }}
      </span>
    </component>
  </template>
</template>
