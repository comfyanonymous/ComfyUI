<template>
  <div
    class="comfy-vue-node-search-container flex w-full min-w-96 items-center justify-center"
  >
    <div
      v-if="enableNodePreview && hoveredSuggestion"
      class="comfy-vue-node-preview-container absolute top-[50px] left-[-375px] z-50 cursor-pointer"
      @mousedown.stop="onAddNode(hoveredSuggestion!, $event)"
    >
      <NodePreview
        :key="hoveredSuggestion?.name || ''"
        :node-def="hoveredSuggestion"
      />
    </div>

    <Button
      variant="secondary"
      :aria-label="$t('g.addNodeFilterCondition')"
      class="filter-button z-10"
      @click="nodeSearchFilterVisible = true"
    >
      <i class="pi pi-filter" />
    </Button>
    <Dialog
      v-model:visible="nodeSearchFilterVisible"
      class="min-w-96"
      dismissable-mask
      modal
      @hide="reFocusInput"
    >
      <template #header>
        <h3>{{ $t('g.addNodeFilterCondition') }}</h3>
      </template>
      <div class="_dialog-body">
        <NodeSearchFilter @add-filter="onAddFilter" />
      </div>
    </Dialog>

    <AutoCompletePlus
      ref="autoCompletePlus"
      :model-value="filters"
      class="comfy-vue-node-search-box z-10 grow"
      scroll-height="40vh"
      :placeholder="placeholder"
      :input-id="inputId"
      append-to="self"
      :suggestions="suggestions"
      :delay="100"
      :loading="!nodeFrequencyStore.isLoaded"
      complete-on-focus
      auto-option-focus
      force-selection
      multiple
      option-label="display_name"
      @complete="search($event.query)"
      @option-select="onAddNode($event.value)"
      @focused-option-changed="setHoverSuggestion($event)"
    >
      <template #option="{ option }">
        <NodeSearchItem :node-def="option" :current-query="currentQuery" />
      </template>
      <!-- FilterAndValue -->
      <template #chip="{ value }">
        <SearchFilterChip
          v-if="value.filterDef && value.value"
          :key="`${value.filterDef.id}-${value.value}`"
          :text="value.value"
          :badge="value.filterDef.invokeSequence.toUpperCase()"
          :badge-class="value.filterDef.invokeSequence + '-badge'"
          @remove="
            onRemoveFilter(
              $event,
              value as FuseFilterWithValue<ComfyNodeDefImpl, string>
            )
          "
        />
      </template>
    </AutoCompletePlus>
  </div>
</template>

<script setup lang="ts">
import { debounce } from 'es-toolkit/compat'
import Dialog from 'primevue/dialog'
import { computed, nextTick, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import NodePreview from '@/components/node/NodePreview.vue'
import AutoCompletePlus from '@/components/primevueOverride/AutoCompletePlus.vue'
import NodeSearchFilter from '@/components/searchbox/NodeSearchFilter.vue'
import NodeSearchItem from '@/components/searchbox/NodeSearchItem.vue'
import Button from '@/components/ui/button/Button.vue'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useTelemetry } from '@/platform/telemetry'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { useNodeDefStore, useNodeFrequencyStore } from '@/stores/nodeDefStore'
import type { FuseFilterWithValue } from '@/utils/fuseUtil'

import SearchFilterChip from '../common/SearchFilterChip.vue'

const settingStore = useSettingStore()
const { t } = useI18n()
const telemetry = useTelemetry()

const enableNodePreview = computed(() =>
  settingStore.get('Comfy.NodeSearchBoxImpl.NodePreview')
)

const { filters, searchLimit = 64 } = defineProps<{
  filters: FuseFilterWithValue<ComfyNodeDefImpl, string>[]
  searchLimit?: number
}>()

const autoCompletePlus = ref()
const nodeSearchFilterVisible = ref(false)
const inputId = `comfy-vue-node-search-box-input-${Math.random()}`
const suggestions = ref<ComfyNodeDefImpl[]>([])
const hoveredSuggestion = ref<ComfyNodeDefImpl | null>(null)
const currentQuery = ref('')
const placeholder = computed(() => {
  return filters.length === 0
    ? t('g.searchPlaceholder', { subject: t('g.nodes') })
    : ''
})

const nodeDefStore = useNodeDefStore()
const nodeFrequencyStore = useNodeFrequencyStore()

// Debounced search tracking (500ms as per implementation plan)
const debouncedTrackSearch = debounce((query: string) => {
  if (query.trim()) {
    telemetry?.trackNodeSearch({ query })
  }
}, 500)

const search = (query: string) => {
  const queryIsEmpty = query === '' && filters.length === 0
  currentQuery.value = query
  suggestions.value = queryIsEmpty
    ? nodeFrequencyStore.topNodeDefs
    : [
        ...nodeDefStore.nodeSearchService.searchNode(query, filters, {
          limit: searchLimit
        })
      ]

  // Track search queries with debounce
  debouncedTrackSearch(query)
}

const emit = defineEmits<{
  addFilter: [filter: FuseFilterWithValue<ComfyNodeDefImpl, string>]
  removeFilter: [filter: FuseFilterWithValue<ComfyNodeDefImpl, string>]
  addNode: [nodeDef: ComfyNodeDefImpl, dragEvent?: MouseEvent]
}>()

// Track node selection and emit addNode event
function onAddNode(nodeDef: ComfyNodeDefImpl, event?: MouseEvent) {
  telemetry?.trackNodeSearchResultSelected({
    node_type: nodeDef.name,
    last_query: currentQuery.value
  })
  emit('addNode', nodeDef, event)
}

let inputElement: HTMLInputElement | null = null
const reFocusInput = async () => {
  inputElement ??= document.getElementById(inputId) as HTMLInputElement
  if (inputElement) {
    inputElement.blur()
    await nextTick(() => inputElement?.focus())
  }
}

onMounted(() => {
  inputElement ??= document.getElementById(inputId) as HTMLInputElement
  if (inputElement) inputElement.focus()
  autoCompletePlus.value.hide = () => search('')
  search('')
  autoCompletePlus.value.show()
})
const onAddFilter = (
  filterAndValue: FuseFilterWithValue<ComfyNodeDefImpl, string>
) => {
  nodeSearchFilterVisible.value = false
  emit('addFilter', filterAndValue)
}
const onRemoveFilter = async (
  event: Event,
  filterAndValue: FuseFilterWithValue<ComfyNodeDefImpl, string>
) => {
  event.stopPropagation()
  event.preventDefault()
  emit('removeFilter', filterAndValue)
  await reFocusInput()
}
const setHoverSuggestion = (index: number) => {
  if (index === -1) {
    hoveredSuggestion.value = null
    return
  }
  const value = suggestions.value[index]
  hoveredSuggestion.value = value
}
</script>
