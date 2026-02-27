<template>
  <div class="flex items-center gap-2 px-2 py-1.5">
    <button
      v-for="chip in chips"
      :key="chip.key"
      type="button"
      :aria-pressed="activeChipKey === chip.key"
      :class="
        cn(
          'cursor-pointer rounded-md border px-3 py-1 text-sm transition-colors flex-auto border-secondary-background',
          activeChipKey === chip.key
            ? 'bg-secondary-background text-foreground'
            : 'bg-transparent text-muted-foreground hover:border-base-foreground/60 hover:text-base-foreground/60'
        )
      "
      @click="emit('selectChip', chip)"
    >
      {{ chip.label }}
    </button>
  </div>
</template>

<script lang="ts">
import type { FuseFilter } from '@/utils/fuseUtil'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

export interface FilterChip {
  key: string
  label: string
  filter: FuseFilter<ComfyNodeDefImpl, string>
}
</script>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { useNodeDefStore } from '@/stores/nodeDefStore'
import { cn } from '@/utils/tailwindUtil'

const { activeChipKey = null } = defineProps<{
  activeChipKey?: string | null
}>()

const emit = defineEmits<{
  selectChip: [chip: FilterChip]
}>()

const { t } = useI18n()
const nodeDefStore = useNodeDefStore()

const chips = computed<FilterChip[]>(() => {
  const searchService = nodeDefStore.nodeSearchService
  return [
    {
      key: 'input',
      label: t('g.input'),
      filter: searchService.inputTypeFilter
    },
    {
      key: 'output',
      label: t('g.output'),
      filter: searchService.outputTypeFilter
    },
    {
      key: 'source',
      label: t('g.source'),
      filter: searchService.nodeSourceFilter
    }
  ]
})
</script>
