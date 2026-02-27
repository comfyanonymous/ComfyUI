<template>
  <div class="flex flex-col gap-3">
    <div
      v-for="(conflict, index) in conflictResult?.conflicts || []"
      :key="index"
      class="rounded-md bg-secondary-background/60"
    >
      <!-- Import failed conflicts show detailed error message -->
      <template v-if="conflict.type === 'import_failed'">
        <div v-if="conflict.required_value" class="overflow-x-hidden rounded">
          <p class="m-0 text-xs text-muted-foreground break-all font-mono">
            {{ conflict.required_value }}
          </p>
        </div>
      </template>

      <!-- Other conflict types use standard message -->
      <template v-else>
        <div class="text-sm break-words">
          {{ getConflictMessage(conflict, $t) }}
        </div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { ConflictDetectionResult } from '@/workbench/extensions/manager/types/conflictDetectionTypes'
import { getConflictMessage } from '@/workbench/extensions/manager/utils/conflictMessageUtil'

const { conflictResult } = defineProps<{
  conflictResult: ConflictDetectionResult | null | undefined
}>()
</script>
