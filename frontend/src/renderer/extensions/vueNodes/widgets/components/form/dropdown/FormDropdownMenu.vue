<script setup lang="ts">
import type { CSSProperties, MaybeRefOrGetter } from 'vue'
import { computed } from 'vue'

import VirtualGrid from '@/components/common/VirtualGrid.vue'

import type {
  FilterOption,
  OwnershipFilterOption,
  OwnershipOption
} from '@/platform/assets/types/filterTypes'

import FormDropdownMenuActions from './FormDropdownMenuActions.vue'
import FormDropdownMenuFilter from './FormDropdownMenuFilter.vue'
import FormDropdownMenuItem from './FormDropdownMenuItem.vue'
import type { FormDropdownItem, LayoutMode, SortOption } from './types'

interface Props {
  items: FormDropdownItem[]
  isSelected: (item: FormDropdownItem, index: number) => boolean
  filterOptions: FilterOption[]
  sortOptions: SortOption[]
  searcher?: (
    query: string,
    onCleanup: (cleanupFn: () => void) => void
  ) => Promise<void>
  updateKey?: MaybeRefOrGetter<unknown>
  showOwnershipFilter?: boolean
  ownershipOptions?: OwnershipFilterOption[]
  showBaseModelFilter?: boolean
  baseModelOptions?: FilterOption[]
}

const {
  items,
  isSelected,
  filterOptions,
  sortOptions,
  searcher,
  updateKey,
  showOwnershipFilter,
  ownershipOptions,
  showBaseModelFilter,
  baseModelOptions
} = defineProps<Props>()
const emit = defineEmits<{
  (e: 'item-click', item: FormDropdownItem, index: number): void
}>()

const filterSelected = defineModel<string>('filterSelected')
const layoutMode = defineModel<LayoutMode>('layoutMode')
const sortSelected = defineModel<string>('sortSelected')
const searchQuery = defineModel<string>('searchQuery')
const ownershipSelected = defineModel<OwnershipOption>('ownershipSelected')
const baseModelSelected = defineModel<Set<string>>('baseModelSelected')

type LayoutConfig = {
  maxColumns: number
  itemHeight: number
  itemWidth: number
  gap: string
}

const LAYOUT_CONFIGS: Record<LayoutMode, LayoutConfig> = {
  grid: { maxColumns: 4, itemHeight: 120, itemWidth: 89, gap: '1rem 0.5rem' },
  list: { maxColumns: 1, itemHeight: 64, itemWidth: 380, gap: '0.5rem' },
  'list-small': {
    maxColumns: 1,
    itemHeight: 40,
    itemWidth: 380,
    gap: '0.25rem'
  }
}

const layoutConfig = computed<LayoutConfig>(
  () => LAYOUT_CONFIGS[layoutMode.value ?? 'grid']
)

const gridStyle = computed<CSSProperties>(() => ({
  display: 'grid',
  gap: layoutConfig.value.gap,
  padding: '1rem',
  width: '100%'
}))

type VirtualDropdownItem = FormDropdownItem & { key: string }
const virtualItems = computed<VirtualDropdownItem[]>(() =>
  items.map((item) => ({
    ...item,
    key: String(item.id)
  }))
)
</script>

<template>
  <div
    class="flex h-[640px] w-103 flex-col rounded-lg bg-component-node-background pt-4 outline outline-offset-[-1px] outline-node-component-border"
  >
    <FormDropdownMenuFilter
      v-if="filterOptions.length > 0"
      v-model:filter-selected="filterSelected"
      :filter-options
    />
    <FormDropdownMenuActions
      v-model:layout-mode="layoutMode"
      v-model:sort-selected="sortSelected"
      v-model:search-query="searchQuery"
      v-model:ownership-selected="ownershipSelected"
      v-model:base-model-selected="baseModelSelected"
      :sort-options
      :searcher
      :update-key
      :show-ownership-filter
      :ownership-options
      :show-base-model-filter
      :base-model-options
    />
    <div
      v-if="items.length === 0"
      class="flex h-50 items-center justify-center"
    >
      <i
        :title="$t('g.noItems')"
        :aria-label="$t('g.noItems')"
        class="icon-[lucide--circle-off] size-30 text-muted-foreground/20"
      />
    </div>
    <VirtualGrid
      v-else
      :key="layoutMode"
      :items="virtualItems"
      :grid-style
      :max-columns="layoutConfig.maxColumns"
      :default-item-height="layoutConfig.itemHeight"
      :default-item-width="layoutConfig.itemWidth"
      :buffer-rows="2"
      class="mt-2 min-h-0 flex-1"
    >
      <template #item="{ item, index }">
        <FormDropdownMenuItem
          :index
          :selected="isSelected(item, index)"
          :preview-url="item.preview_url ?? ''"
          :name="item.name"
          :label="item.label"
          :layout="layoutMode"
          @click="emit('item-click', item, index)"
        />
      </template>
    </VirtualGrid>
  </div>
</template>
