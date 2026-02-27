<template>
  <div class="flex w-full items-center justify-between px-3 pb-4">
    <div class="flex w-full items-start justify-end gap-2 pr-1">
      <Button variant="secondary" @click="handleCopyError">
        {{ $t('importFailed.copyError') }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { useCopyToClipboard } from '@/composables/useCopyToClipboard'
import type { ConflictDetectionResult } from '@/workbench/extensions/manager/types/conflictDetectionTypes'

const { conflictedPackages = [] } = defineProps<{
  conflictedPackages?: ConflictDetectionResult[]
}>()

const { copyToClipboard } = useCopyToClipboard()

const formatErrorText = computed(() => {
  const errorParts: string[] = []

  conflictedPackages.forEach((pkg) => {
    const importFailedConflict = pkg.conflicts.find(
      (conflict) => conflict.type === 'import_failed'
    )

    if (importFailedConflict?.required_value) {
      errorParts.push(importFailedConflict.required_value)
    }
  })

  return errorParts.join('\n\n')
})

const handleCopyError = () => {
  copyToClipboard(formatErrorText.value)
}
</script>
