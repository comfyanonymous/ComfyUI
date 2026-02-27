<template>
  <Chip removable @remove="emit('remove', $event)">
    <Badge size="small" :class="semanticBadgeClass">
      {{ badge }}
    </Badge>
    {{ text }}
  </Chip>
</template>

<script setup lang="ts">
import Badge from 'primevue/badge'
import Chip from 'primevue/chip'
import { computed } from 'vue'

export interface SearchFilter {
  text: string
  badge: string
  badgeClass: string
  id: string | number
}

const semanticClassMap: Record<string, string> = {
  'i-badge': 'bg-green-500 text-white',
  'o-badge': 'bg-red-500 text-white',
  'c-badge': 'bg-blue-500 text-white',
  's-badge': 'bg-yellow-500'
}

const props = defineProps<Omit<SearchFilter, 'id'>>()
const emit = defineEmits<{
  (e: 'remove', event: Event): void
}>()

const semanticBadgeClass = computed(() => {
  return semanticClassMap[props.badgeClass] ?? props.badgeClass
})
</script>
