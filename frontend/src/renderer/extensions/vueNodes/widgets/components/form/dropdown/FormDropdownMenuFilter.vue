<script setup lang="ts">
import { computed } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { useModelUpload } from '@/platform/assets/composables/useModelUpload'
import type { FilterOption } from '@/platform/assets/types/filterTypes'
import { cn } from '@/utils/tailwindUtil'

const { filterOptions } = defineProps<{
  filterOptions: FilterOption[]
}>()

const filterSelected = defineModel<string>('filterSelected')

const { isUploadButtonEnabled, showUploadDialog } = useModelUpload()

const singleFilterOption = computed(() => filterOptions.length === 1)
</script>

<template>
  <div class="text-secondary mb-4 flex gap-1 px-4 justify-start">
    <button
      v-for="option in filterOptions"
      :key="option.value"
      type="button"
      :disabled="singleFilterOption"
      :class="
        cn(
          'px-4 py-2 rounded-md inline-flex justify-center items-center select-none appearance-none border-0 text-base-foreground',
          !singleFilterOption &&
            'transition-all duration-150 hover:text-base-foreground hover:bg-interface-menu-component-surface-hovered cursor-pointer active:scale-95',
          !singleFilterOption && filterSelected === option.value
            ? '!bg-interface-menu-component-surface-selected text-base-foreground'
            : 'bg-transparent'
        )
      "
      @click="filterSelected = option.value"
    >
      {{ option.name }}
    </button>
    <Button
      v-if="isUploadButtonEnabled && singleFilterOption"
      class="ml-auto"
      size="md"
      variant="textonly"
      @click="showUploadDialog"
    >
      <i class="icon-[lucide--folder-input]" />
      <span>{{ $t('g.import') }}</span>
    </Button>
  </div>
</template>
