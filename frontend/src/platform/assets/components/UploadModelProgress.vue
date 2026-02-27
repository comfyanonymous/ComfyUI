<template>
  <div class="flex flex-1 flex-col gap-6 text-sm text-muted-foreground">
    <!-- Processing State (202 async download in progress) -->
    <div v-if="result === 'processing'" class="flex flex-col gap-2">
      <p class="m-0 font-bold">
        {{ $t('assetBrowser.processingModel') }}
      </p>
      <p class="m-0">
        {{ $t('assetBrowser.processingModelDescription') }}
      </p>

      <div
        class="flex flex-row items-center gap-3 rounded-lg bg-modal-card-background p-4"
      >
        <img
          v-if="previewImage"
          :src="previewImage"
          :alt="metadata?.filename || metadata?.name || 'Model preview'"
          class="size-14 flex-shrink-0 rounded object-cover"
        />
        <div
          class="flex min-w-0 flex-1 flex-col items-start justify-center gap-1"
        >
          <p class="m-0 w-full truncate text-base-foreground">
            {{ metadata?.filename || metadata?.name }}
          </p>
          <p class="m-0 text-sm text-muted">
            {{ modelType }}
          </p>
        </div>
      </div>
    </div>

    <!-- Success State -->
    <div v-else-if="result === 'success'" class="flex flex-col gap-2">
      <p class="m-0 font-bold">
        {{ $t('assetBrowser.modelUploaded') }}
      </p>
      <p class="m-0">
        {{ $t('assetBrowser.findInLibrary', { type: modelType }) }}
      </p>

      <div
        class="flex flex-row items-center gap-3 rounded-lg bg-modal-card-background p-4"
      >
        <img
          v-if="previewImage"
          :src="previewImage"
          :alt="metadata?.filename || metadata?.name || 'Model preview'"
          class="size-14 flex-shrink-0 rounded object-cover"
        />
        <div
          class="flex min-w-0 flex-1 flex-col items-start justify-center gap-1"
        >
          <p class="m-0 w-full truncate text-base-foreground">
            {{ metadata?.filename || metadata?.name }}
          </p>
          <p class="m-0 text-sm text-muted">
            {{ modelType }}
          </p>
        </div>
      </div>
    </div>

    <!-- Error State -->
    <div
      v-else-if="result === 'error'"
      class="flex flex-1 flex-col items-center justify-center gap-6"
    >
      <i class="icon-[lucide--x-circle] text-6xl text-error" />
      <div class="text-center">
        <p class="m-0 text-sm font-bold">
          {{ $t('assetBrowser.uploadFailed') }}
        </p>
        <p v-if="error" class="text-sm text-muted mb-0">
          {{ error }}
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { AssetMetadata } from '@/platform/assets/schemas/assetSchema'

defineProps<{
  result: 'processing' | 'success' | 'error'
  error?: string
  metadata?: AssetMetadata
  modelType?: string
  previewImage?: string
}>()
</script>
