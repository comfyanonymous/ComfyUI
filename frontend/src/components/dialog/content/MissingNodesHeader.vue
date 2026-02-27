<template>
  <div class="flex w-full items-center justify-between p-4">
    <div class="flex items-center gap-2">
      <i class="icon-[lucide--triangle-alert] text-warning-background"></i>
      <p class="m-0 text-sm">
        {{
          isCloud
            ? $t('missingNodes.cloud.title')
            : $t('missingNodes.oss.title')
        }}
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { isCloud } from '@/platform/distribution/types'
</script>
