<template>
  <div class="flex w-[490px] flex-col border-t-1 border-border-default">
    <div class="flex h-full w-full flex-col gap-4 p-4">
      <!-- Error Details -->
      <div v-if="importFailedPackages.length > 0" class="flex flex-col gap-3">
        <div
          v-for="pkg in importFailedPackages"
          :key="pkg.packageId"
          class="flex flex-col gap-2 max-h-60 overflow-x-hidden overflow-y-auto scrollbar-custom"
          role="region"
          :aria-label="`Error traceback for ${pkg.packageId}`"
          tabindex="0"
        >
          <!-- Error Message -->
          <div
            v-if="pkg.traceback || pkg.errorMessage"
            class="text-xs p-4 rounded-md bg-secondary-background font-mono"
          >
            {{ pkg.traceback || pkg.errorMessage }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import type { ConflictDetectionResult } from '@/workbench/extensions/manager/types/conflictDetectionTypes'

const { conflictedPackages } = defineProps<{
  conflictedPackages: ConflictDetectionResult[]
}>()

interface ImportFailedPackage {
  packageId: string
  packageName: string
  errorMessage: string
  traceback: string
}

const importFailedPackages = computed((): ImportFailedPackage[] => {
  return conflictedPackages
    .filter((pkg) =>
      pkg.conflicts.some((conflict) => conflict.type === 'import_failed')
    )
    .map((pkg) => {
      const importFailedConflict = pkg.conflicts.find(
        (conflict) => conflict.type === 'import_failed'
      )
      if (!importFailedConflict) {
        return {
          packageId: pkg.package_id,
          packageName: pkg.package_name,
          errorMessage: 'Unknown import error',
          traceback: ''
        }
      }

      return {
        packageId: pkg.package_id,
        packageName: pkg.package_name,
        errorMessage:
          importFailedConflict.current_value || 'Unknown import error',
        traceback: importFailedConflict.required_value || ''
      }
    })
})
</script>
