<script setup lang="ts">
import { useI18n } from 'vue-i18n'

import Popover from '@/components/ui/Popover.vue'
import Button from '@/components/ui/button/Button.vue'
import { useCommandStore } from '@/stores/commandStore'

const { queueCount } = defineProps<{
  queueCount: number
}>()

const { t } = useI18n()
const commandStore = useCommandStore()

function clearQueue(close: () => void) {
  void commandStore.execute('Comfy.ClearPendingTasks')
  close()
}
</script>
<template>
  <div
    class="shrink-0 p-1 border-2 border-transparent relative"
    data-testid="linear-job"
  >
    <Popover side="top" :show-arrow="false" @focus-outside.prevent>
      <template #button>
        <Button
          v-tooltip.top="t('linearMode.queue.clickToClear')"
          :aria-label="t('linearMode.queue.clickToClear')"
          variant="textonly"
          size="unset"
          class="size-10 rounded-sm bg-secondary-background flex items-center justify-center"
        >
          <i
            class="icon-[lucide--loader-circle] size-4 animate-spin text-muted-foreground"
          />
        </Button>
      </template>
      <template #default="{ close }">
        <Button
          :disabled="queueCount === 0"
          variant="textonly"
          class="text-destructive-background px-4 text-sm"
          @click="clearQueue(close)"
        >
          <i class="icon-[lucide--list-x]" />
          {{ t('linearMode.queue.clear') }}
        </Button>
      </template>
    </Popover>
    <div
      v-if="queueCount > 1"
      aria-hidden="true"
      class="absolute top-0 right-0 min-w-4 h-4 flex justify-center items-center rounded-full bg-primary-background text-text-primary text-xs"
      v-text="queueCount"
    />
  </div>
</template>
