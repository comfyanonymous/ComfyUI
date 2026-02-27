<template>
  <div
    v-tooltip.bottom="{
      value: $t('menu.batchCount'),
      showDelay: 600
    }"
    class="batch-count"
    :aria-label="$t('menu.batchCount')"
  >
    <InputNumber
      v-model="batchCount"
      class="w-14"
      :min="minQueueCount"
      :max="maxQueueCount"
      fluid
      show-buttons
      :pt="{
        incrementButton: {
          class: 'w-6',
          onmousedown: () => {
            handleClick(true)
          }
        },
        decrementButton: {
          class: 'w-6',
          onmousedown: () => {
            handleClick(false)
          }
        }
      }"
    />
  </div>
</template>

<script lang="ts" setup>
import { storeToRefs } from 'pinia'
import InputNumber from 'primevue/inputnumber'
import { computed } from 'vue'

import { useSettingStore } from '@/platform/settings/settingStore'
import { useQueueSettingsStore } from '@/stores/queueStore'

const queueSettingsStore = useQueueSettingsStore()
const { batchCount } = storeToRefs(queueSettingsStore)
const minQueueCount = 1

const settingStore = useSettingStore()
const maxQueueCount = computed(() =>
  settingStore.get('Comfy.QueueButton.BatchCountLimit')
)

const handleClick = (increment: boolean) => {
  let newCount: number
  if (increment) {
    const originalCount = batchCount.value - 1
    newCount = Math.min(originalCount * 2, maxQueueCount.value)
  } else {
    const originalCount = batchCount.value + 1
    newCount = Math.floor(originalCount / 2)
  }

  batchCount.value = newCount
}
</script>

<style scoped>
:deep(.p-inputtext) {
  border-top-left-radius: 0;
  border-bottom-left-radius: 0;
}
</style>
