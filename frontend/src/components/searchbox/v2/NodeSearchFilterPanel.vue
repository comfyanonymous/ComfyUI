<template>
  <div
    id="filter-options-list"
    ref="listRef"
    role="listbox"
    class="flex-1 overflow-y-auto py-2"
  >
    <div
      v-for="(option, index) in options"
      :id="`filter-option-${index}`"
      :key="option"
      role="option"
      data-testid="filter-option"
      :aria-selected="index === selectedIndex"
      :class="
        cn(
          'cursor-pointer px-6 py-1.5',
          index === selectedIndex && 'bg-secondary-background-hover'
        )
      "
      @click="emit('apply', option)"
      @mouseenter="selectedIndex = index"
    >
      <span class="text-base font-semibold text-foreground">
        <span class="text-2xl mr-1" :style="{ color: getLinkTypeColor(option) }"
          >&bull;</span
        >
        {{ option }}
      </span>
    </div>
    <div
      v-if="options.length === 0"
      class="px-4 py-8 text-center text-muted-foreground"
    >
      {{ $t('g.noResults') }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'

import type { FilterChip } from '@/components/searchbox/v2/NodeSearchFilterBar.vue'
import { getLinkTypeColor } from '@/utils/litegraphUtil'
import { cn } from '@/utils/tailwindUtil'

const { chip } = defineProps<{
  chip: FilterChip
}>()

const query = defineModel<string>('query', { required: true })

const emit = defineEmits<{
  apply: [value: string]
}>()

const listRef = ref<HTMLElement>()
const selectedIndex = ref(0)

const options = computed(() => {
  const { fuseSearch } = chip.filter
  if (query.value) {
    return fuseSearch.search(query.value).slice(0, 64)
  }
  return fuseSearch.data.slice().sort()
})

watch(query, () => {
  selectedIndex.value = 0
})

function navigate(direction: number) {
  const newIndex = selectedIndex.value + direction
  if (newIndex >= 0 && newIndex < options.value.length) {
    selectedIndex.value = newIndex
    nextTick(() => {
      listRef.value
        ?.querySelector(`#filter-option-${newIndex}`)
        ?.scrollIntoView({ block: 'nearest' })
    })
  }
}

function selectCurrent() {
  const option = options.value[selectedIndex.value]
  if (option) emit('apply', option)
}

defineExpose({ navigate, selectCurrent })
</script>
