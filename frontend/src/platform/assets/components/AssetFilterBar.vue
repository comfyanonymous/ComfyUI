<template>
  <div
    class="flex gap-4 items-center justify-between px-6 pt-2 pb-6"
    data-component-id="asset-filter-bar"
  >
    <div
      class="flex gap-4 items-center"
      data-component-id="asset-filter-bar-left"
    >
      <MultiSelect
        v-if="availableFileFormats.length > 0"
        v-model="activeFileFormatObjects"
        :label="$t('assetBrowser.fileFormats')"
        :options="availableFileFormats"
        class="min-w-32"
        data-component-id="asset-filter-file-formats"
        @update:model-value="handleFilterChange"
      />

      <MultiSelect
        v-if="availableBaseModels.length > 0"
        v-model="activeBaseModelObjects"
        :label="$t('assetBrowser.baseModels')"
        :options="availableBaseModels"
        class="min-w-32"
        data-component-id="asset-filter-base-models"
        @update:model-value="handleFilterChange"
      />

      <SingleSelect
        v-if="showOwnershipFilter"
        v-model="ownership"
        :label="$t('assetBrowser.ownership')"
        :options="ownershipOptions"
        class="min-w-32"
        data-component-id="asset-filter-ownership"
        @update:model-value="handleFilterChange"
      />
    </div>

    <div class="flex items-center" data-component-id="asset-filter-bar-right">
      <SingleSelect
        v-model="sortBy"
        :label="$t('assetBrowser.sortBy')"
        :options="sortOptions"
        class="min-w-32"
        data-component-id="asset-filter-sort"
        @update:model-value="handleFilterChange"
      >
        <template #icon>
          <i class="icon-[lucide--arrow-up-down]" />
        </template>
      </SingleSelect>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import MultiSelect from '@/components/input/MultiSelect.vue'
import SingleSelect from '@/components/input/SingleSelect.vue'
import type { SelectOption } from '@/components/input/types'
import { useAssetFilterOptions } from '@/platform/assets/composables/useAssetFilterOptions'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import type {
  AssetFilterState,
  AssetSortOption,
  OwnershipOption
} from '@/platform/assets/types/filterTypes'

const { t } = useI18n()

const sortOptions = computed(() => [
  { name: t('assetBrowser.sortRecent'), value: 'recent' as const },
  { name: t('assetBrowser.sortAZ'), value: 'name-asc' as const },
  { name: t('assetBrowser.sortZA'), value: 'name-desc' as const }
])

const { assets = [], showOwnershipFilter = false } = defineProps<{
  assets?: AssetItem[]
  showOwnershipFilter?: boolean
}>()

const selectedFileFormats = ref<SelectOption[]>([])
const selectedBaseModels = ref<SelectOption[]>([])
const sortBy = ref<AssetSortOption>('recent')
const ownership = ref<OwnershipOption>('all')

const { availableFileFormats, availableBaseModels, ownershipOptions } =
  useAssetFilterOptions(() => assets)

// Only show selected items that exist in the current scope
const activeFileFormatObjects = computed({
  get() {
    return selectedFileFormats.value.filter((opt) =>
      availableFileFormats.value.some((a) => a.value === opt.value)
    )
  },
  set(value: SelectOption[]) {
    selectedFileFormats.value = value
  }
})

const activeBaseModelObjects = computed({
  get() {
    return selectedBaseModels.value.filter((opt) =>
      availableBaseModels.value.some((a) => a.value === opt.value)
    )
  },
  set(value: SelectOption[]) {
    selectedBaseModels.value = value
  }
})

const emit = defineEmits<{
  filterChange: [filters: AssetFilterState]
}>()

function handleFilterChange() {
  emit('filterChange', {
    fileFormats: activeFileFormatObjects.value.map((opt) => opt.value),
    baseModels: activeBaseModelObjects.value.map((opt) => opt.value),
    sortBy: sortBy.value,
    ownership: ownership.value
  })
}
</script>
