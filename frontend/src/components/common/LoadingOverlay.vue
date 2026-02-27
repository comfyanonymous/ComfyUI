<template>
  <Transition name="fade">
    <div
      v-if="loading"
      class="absolute inset-0 z-50 flex items-center justify-center bg-backdrop/50"
    >
      <div class="flex flex-col items-center">
        <div class="grid place-items-center">
          <div
            :class="
              cn(
                'col-start-1 row-start-1 animate-spin rounded-full border-muted-foreground border-t-base-foreground',
                spinnerSizeClass
              )
            "
          />
          <div class="col-start-1 row-start-1">
            <slot />
          </div>
        </div>
        <div v-if="loadingMessage" class="mt-4 text-lg text-base-foreground">
          {{ loadingMessage }}
        </div>
      </div>
    </div>
  </Transition>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const { size = 'md' } = defineProps<{
  loading: boolean
  loadingMessage?: string
  size?: 'sm' | 'md'
}>()

const spinnerSizeClass = computed(() => {
  switch (size) {
    case 'sm':
      return 'h-6 w-6 border-2'
    case 'md':
    default:
      return 'h-12 w-12 border-4'
  }
})
</script>
