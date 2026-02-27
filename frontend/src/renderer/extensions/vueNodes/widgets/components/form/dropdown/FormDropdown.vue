<script setup lang="ts">
import Popover from 'primevue/popover'
import { computed, ref, useTemplateRef } from 'vue'
import { useI18n } from 'vue-i18n'

import { useToastStore } from '@/platform/updates/common/toastStore'

import type {
  FilterOption,
  OwnershipFilterOption,
  OwnershipOption
} from '@/platform/assets/types/filterTypes'

import FormDropdownInput from './FormDropdownInput.vue'
import FormDropdownMenu from './FormDropdownMenu.vue'
import { defaultSearcher, getDefaultSortOptions } from './shared'
import type { FormDropdownItem, LayoutMode, SortOption } from './types'

interface Props {
  items: FormDropdownItem[]
  /** Items used for display in the input field. Falls back to items if not provided. */
  displayItems?: FormDropdownItem[]
  placeholder?: string
  /**
   * If true, allows multiple selections. If a number is provided,
   * it specifies the maximum number of selections allowed.
   */
  multiple?: boolean | number

  uploadable?: boolean
  disabled?: boolean
  accept?: string
  filterOptions?: FilterOption[]
  sortOptions?: SortOption[]
  showOwnershipFilter?: boolean
  ownershipOptions?: OwnershipFilterOption[]
  showBaseModelFilter?: boolean
  baseModelOptions?: FilterOption[]
  isSelected?: (
    selected: Set<string>,
    item: FormDropdownItem,
    index: number
  ) => boolean
  searcher?: (
    query: string,
    items: FormDropdownItem[],
    onCleanup: (cleanupFn: () => void) => void
  ) => Promise<FormDropdownItem[]>
}

const { t } = useI18n()

const {
  placeholder,
  multiple = false,
  uploadable = false,
  disabled = false,
  accept,
  filterOptions = [],
  sortOptions = getDefaultSortOptions(),
  showOwnershipFilter,
  ownershipOptions,
  showBaseModelFilter,
  baseModelOptions,
  isSelected = (selected, item, _index) => selected.has(item.id),
  searcher = defaultSearcher,
  items
} = defineProps<Props>()

const placeholderText = computed(
  () => placeholder ?? t('widgets.uploadSelect.placeholder')
)

const selected = defineModel<Set<string>>('selected', {
  default: () => new Set()
})
const filterSelected = defineModel<string>('filterSelected', { default: '' })
const sortSelected = defineModel<string>('sortSelected', {
  default: 'default'
})
const layoutMode = defineModel<LayoutMode>('layoutMode', {
  default: 'grid'
})
const files = defineModel<File[]>('files', { default: () => [] })
const searchQuery = defineModel<string>('searchQuery', { default: '' })
const ownershipSelected = defineModel<OwnershipOption>('ownershipSelected', {
  default: 'all'
})
const baseModelSelected = defineModel<Set<string>>('baseModelSelected', {
  default: () => new Set()
})

const toastStore = useToastStore()
const popoverRef = ref<InstanceType<typeof Popover>>()
const triggerRef = useTemplateRef('triggerRef')
const isOpen = ref(false)

const maxSelectable = computed(() => {
  if (multiple === true) return Infinity
  if (typeof multiple === 'number') return multiple
  return 1
})

const itemsKey = computed(() => items.map((item) => item.id).join('|'))

const filteredItems = ref<FormDropdownItem[]>([])

const defaultSorter = computed<SortOption['sorter']>(() => {
  const sorter = sortOptions.find((option) => option.id === 'default')?.sorter
  return sorter || (({ items: i }) => i.slice())
})
const selectedSorter = computed<SortOption['sorter']>(() => {
  if (sortSelected.value === 'default') return defaultSorter.value
  const sorter = sortOptions.find(
    (option) => option.id === sortSelected.value
  )?.sorter
  return sorter || defaultSorter.value
})
const sortedItems = computed(() => {
  return selectedSorter.value({ items: filteredItems.value }) || []
})

function internalIsSelected(item: FormDropdownItem, index: number): boolean {
  return isSelected(selected.value, item, index)
}

const toggleDropdown = (event: Event) => {
  if (disabled) return
  if (popoverRef.value && triggerRef.value) {
    popoverRef.value.toggle(event, triggerRef.value)
    isOpen.value = !isOpen.value
  }
}

const closeDropdown = () => {
  if (popoverRef.value) {
    popoverRef.value.hide()
    isOpen.value = false
  }
}

function handleFileChange(event: Event) {
  if (disabled) return
  const target = event.target
  if (!(target instanceof HTMLInputElement)) return
  if (target.files) {
    files.value = Array.from(target.files)
  }
  target.value = ''
}

function handleSelection(item: FormDropdownItem, index: number) {
  if (disabled) return
  const sel = selected.value
  if (internalIsSelected(item, index)) {
    sel.delete(item.id)
  } else {
    if (sel.size < maxSelectable.value) {
      sel.add(item.id)
    } else if (maxSelectable.value === 1) {
      sel.clear()
      sel.add(item.id)
    } else {
      toastStore.addAlert(t('widgets.uploadSelect.maxSelectionReached'))
      return
    }
  }
  selected.value = new Set(sel)

  if (maxSelectable.value === 1) {
    closeDropdown()
  }
}

async function customSearcher(
  query: string,
  onCleanup: (cleanupFn: () => void) => void
) {
  let isCleanup = false
  let cleanupFn: undefined | (() => void)
  onCleanup(() => {
    isCleanup = true
    cleanupFn?.()
  })
  await searcher(query, items, (cb) => (cleanupFn = cb)).then((results) => {
    if (!isCleanup) filteredItems.value = results
  })
}
</script>

<template>
  <div ref="triggerRef">
    <FormDropdownInput
      :files
      :is-open
      :placeholder="placeholderText"
      :items
      :display-items
      :max-selectable
      :selected
      :uploadable
      :disabled
      :accept
      @select-click="toggleDropdown"
      @file-change="handleFileChange"
    />
    <Popover
      ref="popoverRef"
      :dismissable="true"
      :close-on-escape="true"
      unstyled
      :pt="{
        root: {
          class: 'absolute z-50'
        },
        content: {
          class: ['bg-transparent border-none p-0 pt-2 rounded-lg shadow-lg']
        }
      }"
      @hide="isOpen = false"
    >
      <FormDropdownMenu
        v-model:filter-selected="filterSelected"
        v-model:layout-mode="layoutMode"
        v-model:sort-selected="sortSelected"
        v-model:search-query="searchQuery"
        v-model:ownership-selected="ownershipSelected"
        v-model:base-model-selected="baseModelSelected"
        :filter-options
        :sort-options
        :show-ownership-filter
        :ownership-options
        :show-base-model-filter
        :base-model-options
        :disabled
        :searcher="customSearcher"
        :items="sortedItems"
        :is-selected="internalIsSelected"
        :max-selectable
        :update-key="itemsKey"
        @close="closeDropdown"
        @item-click="handleSelection"
      />
    </Popover>
  </div>
</template>
