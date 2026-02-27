<template>
  <div class="p-3">
    <TagsInputRoot
      :model-value="tagValues"
      delimiter=""
      class="flex cursor-text flex-wrap items-center gap-2 rounded-lg bg-secondary-background px-4 py-3"
      @remove-tag="onRemoveTag"
      @click="inputRef?.focus()"
    >
      <!-- Active filter label (filter selection mode) -->
      <span
        v-if="activeFilter"
        class="inline-flex shrink-0 items-center gap-1 rounded-lg bg-base-background px-2 py-1 -my-1 text-sm opacity-80 text-foreground"
      >
        {{ activeFilter.label }}:
        <button
          type="button"
          data-testid="cancel-filter"
          class="cursor-pointer border-none bg-transparent text-muted-foreground hover:text-base-foreground rounded-full aspect-square"
          :aria-label="$t('g.remove')"
          @click="emit('cancelFilter')"
        >
          <i class="pi pi-times text-xs" />
        </button>
      </span>
      <!-- Applied filter chips -->
      <template v-if="!activeFilter">
        <TagsInputItem
          v-for="filter in filters"
          :key="filterKey(filter)"
          :value="filterKey(filter)"
          data-testid="filter-chip"
          class="inline-flex items-center gap-1 rounded-lg bg-base-background px-2 py-1 -my-1 data-[state=active]:ring-2 data-[state=active]:ring-primary"
        >
          <span class="text-sm opacity-80">
            {{ t(`g.${filter.filterDef.id}`) }}:
          </span>
          <span :style="{ color: getLinkTypeColor(filter.value) }">
            &bull;
          </span>
          <span class="text-sm">{{ filter.value }}</span>
          <TagsInputItemDelete
            as="button"
            type="button"
            data-testid="chip-delete"
            :aria-label="$t('g.remove')"
            class="ml-1 cursor-pointer border-none bg-transparent text-muted-foreground hover:text-base-foreground rounded-full aspect-square"
          >
            <i class="pi pi-times text-xs" />
          </TagsInputItemDelete>
        </TagsInputItem>
      </template>
      <TagsInputInput as-child>
        <input
          ref="inputRef"
          v-model="inputValue"
          type="text"
          role="combobox"
          aria-autocomplete="list"
          :aria-expanded="true"
          :aria-controls="activeFilter ? 'filter-options-list' : 'results-list'"
          :aria-label="inputPlaceholder"
          :placeholder="inputPlaceholder"
          class="h-6 min-w-[min(300px,80vw)] flex-1 border-none bg-transparent text-foreground text-sm outline-none placeholder:text-muted-foreground"
          @keydown.enter.prevent="emit('selectCurrent')"
          @keydown.down.prevent="emit('navigateDown')"
          @keydown.up.prevent="emit('navigateUp')"
        />
      </TagsInputInput>
    </TagsInputRoot>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  TagsInputInput,
  TagsInputItem,
  TagsInputItemDelete,
  TagsInputRoot
} from 'reka-ui'

import type { FilterChip } from '@/components/searchbox/v2/NodeSearchFilterBar.vue'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import type { FuseFilterWithValue } from '@/utils/fuseUtil'
import { getLinkTypeColor } from '@/utils/litegraphUtil'

const { filters, activeFilter } = defineProps<{
  filters: FuseFilterWithValue<ComfyNodeDefImpl, string>[]
  activeFilter: FilterChip | null
}>()

const searchQuery = defineModel<string>('searchQuery', { required: true })
const filterQuery = defineModel<string>('filterQuery', { required: true })

const emit = defineEmits<{
  removeFilter: [filter: FuseFilterWithValue<ComfyNodeDefImpl, string>]
  cancelFilter: []
  navigateDown: []
  navigateUp: []
  selectCurrent: []
}>()

const { t } = useI18n()
const inputRef = ref<HTMLInputElement>()

const inputValue = computed({
  get: () => (activeFilter ? filterQuery.value : searchQuery.value),
  set: (value: string) => {
    if (activeFilter) {
      filterQuery.value = value
    } else {
      searchQuery.value = value
    }
  }
})

const inputPlaceholder = computed(() =>
  activeFilter
    ? t('g.filterByType', { type: activeFilter.label.toLowerCase() })
    : t('g.addNode')
)

const tagValues = computed(() => filters.map(filterKey))

function filterKey(filter: FuseFilterWithValue<ComfyNodeDefImpl, string>) {
  return `${filter.filterDef.id}:${filter.value}`
}

function onRemoveTag(tagValue: string) {
  const filter = filters.find((f) => filterKey(f) === tagValue)
  if (filter) emit('removeFilter', filter)
}

function focus() {
  inputRef.value?.focus()
}

onMounted(() => {
  focus()
})

defineExpose({ focus })
</script>
