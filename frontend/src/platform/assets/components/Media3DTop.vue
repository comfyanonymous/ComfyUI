<template>
  <div class="relative size-full overflow-hidden rounded">
    <img
      v-if="!thumbnailError"
      :src="thumbnailSrc"
      :alt="asset?.name"
      class="size-full object-contain transition-transform duration-300 group-hover:scale-105 group-data-[selected=true]:scale-105"
      @error="thumbnailError = true"
    />
    <div
      v-else
      class="flex size-full flex-col items-center justify-center gap-2 bg-modal-card-placeholder-background transition-transform duration-300 group-hover:scale-105 group-data-[selected=true]:scale-105"
    >
      <i class="icon-[lucide--box] text-3xl text-muted-foreground" />
      <span class="text-sm text-base-foreground">
        {{ $t('assetBrowser.media.threeDModelPlaceholder') }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

import type { AssetMeta } from '../schemas/mediaAssetSchema'

const { asset } = defineProps<{ asset: AssetMeta }>()

const thumbnailError = ref(false)

const thumbnailSrc = computed(() => {
  if (!asset?.src) return ''
  return asset.src.replace(/([?&]filename=)([^&]*)/, '$1$2.png')
})
</script>
