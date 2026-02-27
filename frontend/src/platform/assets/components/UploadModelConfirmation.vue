<template>
  <div class="flex flex-col gap-4 text-sm text-muted-foreground">
    <div class="flex flex-col gap-2">
      <p class="m-0">
        {{ $t('assetBrowser.modelAssociatedWithLink') }}
      </p>
      <div
        class="flex items-center gap-3 rounded-lg bg-secondary-background px-4 py-2"
      >
        <img
          v-if="previewImage"
          :src="previewImage"
          :alt="metadata?.filename || metadata?.name || 'Model preview'"
          class="size-14 flex-shrink-0 rounded object-cover"
        />
        <p class="m-0 min-w-0 flex-1 truncate text-base-foreground">
          {{ metadata?.filename || metadata?.name }}
        </p>
      </div>
    </div>

    <!-- Model Type Selection -->
    <div class="flex flex-col gap-2">
      <div class="flex items-center gap-2">
        <label>
          {{ $t('assetBrowser.modelTypeSelectorLabel') }}
        </label>
        <i class="icon-[lucide--circle-question-mark] text-muted-foreground" />
        <span class="text-muted-foreground">
          {{ $t('assetBrowser.notSureLeaveAsIs') }}
        </span>
      </div>
      <SingleSelect
        v-model="modelValue"
        :label="
          isLoading
            ? $t('g.loading')
            : $t('assetBrowser.modelTypeSelectorPlaceholder')
        "
        :options="modelTypes"
        :disabled="isLoading"
        data-attr="upload-model-step2-type-selector"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import SingleSelect from '@/components/input/SingleSelect.vue'
import { useModelTypes } from '@/platform/assets/composables/useModelTypes'
import type { AssetMetadata } from '@/platform/assets/schemas/assetSchema'

defineProps<{
  metadata?: AssetMetadata
  previewImage?: string
}>()

const modelValue = defineModel<string | undefined>()

const { modelTypes, isLoading } = useModelTypes()
</script>
