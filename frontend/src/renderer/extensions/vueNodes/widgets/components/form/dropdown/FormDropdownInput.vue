<script setup lang="ts">
import { computed } from 'vue'

import { cn } from '@/utils/tailwindUtil'

import { WidgetInputBaseClass } from '../../layout'
import type { FormDropdownItem } from './types'

interface Props {
  isOpen?: boolean
  placeholder?: string
  items: FormDropdownItem[]
  /** Items used for display in the input field. Falls back to items if not provided. */
  displayItems?: FormDropdownItem[]
  selected: Set<string>
  maxSelectable: number
  uploadable: boolean
  disabled: boolean
  accept?: string
}

const {
  isOpen = false,
  placeholder = 'Select...',
  items,
  displayItems,
  selected,
  maxSelectable,
  uploadable,
  disabled,
  accept
} = defineProps<Props>()

const emit = defineEmits<{
  (e: 'select-click', event: MouseEvent): void
  (e: 'file-change', event: Event): void
}>()

const selectedItems = computed(() => {
  const itemsToSearch = displayItems ?? items
  return itemsToSearch.filter((item) => selected.has(item.id))
})

const theButtonStyle = computed(() =>
  cn(
    'border-0 bg-component-node-widget-background outline-none text-text-secondary',
    disabled
      ? 'cursor-not-allowed'
      : 'hover:bg-component-node-widget-background-hovered cursor-pointer',
    selectedItems.value.length > 0 && 'text-text-primary'
  )
)
</script>

<template>
  <div
    :class="
      cn(WidgetInputBaseClass, 'flex text-base leading-none', {
        'opacity-50 cursor-not-allowed outline-zinc-300/10': disabled
      })
    "
  >
    <button
      :class="
        cn(
          theButtonStyle,
          'flex justify-between items-center flex-1 min-w-0 h-8',
          {
            'rounded-l-lg': uploadable,
            'rounded-lg': !uploadable
          }
        )
      "
      @click="emit('select-click', $event)"
    >
      <span class="min-w-0 flex-1 px-1 py-2 text-left truncate">
        <span v-if="!selectedItems.length">
          {{ placeholder }}
        </span>
        <span v-else>
          {{ selectedItems.map((item) => item.label ?? item.name).join(', ') }}
        </span>
      </span>
      <i
        class="icon-[lucide--chevron-down]"
        :class="
          cn(
            'mr-2 size-4 transition-transform duration-200 flex-shrink-0 text-component-node-foreground-secondary',
            isOpen && 'rotate-180'
          )
        "
      />
    </button>
    <label
      v-if="uploadable"
      :class="
        cn(
          theButtonStyle,
          'relative',
          'size-8 flex justify-center items-center border-l rounded-r-lg border-zinc-300/10'
        )
      "
    >
      <i class="icon-[lucide--folder-search] size-4" />
      <input
        type="file"
        class="absolute inset-0 -z-1 opacity-0"
        :multiple="maxSelectable > 1"
        :disabled="disabled"
        :accept="accept"
        @change="emit('file-change', $event)"
      />
    </label>
  </div>
</template>
