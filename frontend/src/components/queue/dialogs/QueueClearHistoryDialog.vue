<template>
  <section
    class="w-[360px] rounded-2xl border border-interface-stroke bg-interface-panel-surface text-text-primary shadow-interface font-inter"
  >
    <header
      class="flex items-center justify-between border-b border-interface-stroke px-4 py-4"
    >
      <p class="m-0 text-[14px] font-normal leading-none">
        {{ t('sideToolbar.queueProgressOverlay.clearHistoryDialogTitle') }}
      </p>
      <Button
        size="icon"
        variant="muted-textonly"
        :aria-label="t('g.close')"
        @click="onCancel"
      >
        <i class="icon-[lucide--x] block size-4 leading-none" />
      </Button>
    </header>

    <div class="flex flex-col gap-4 px-4 py-4 text-[14px] text-text-secondary">
      <p class="m-0">
        {{
          t('sideToolbar.queueProgressOverlay.clearHistoryDialogDescription')
        }}
      </p>
      <p class="m-0">
        {{ t('sideToolbar.queueProgressOverlay.clearHistoryDialogAssetsNote') }}
      </p>
    </div>

    <footer class="flex items-center justify-end px-4 py-4">
      <div class="flex items-center gap-4 leading-none">
        <Button variant="muted-textonly" size="lg" @click="onCancel">
          {{ t('g.cancel') }}
        </Button>
        <Button
          variant="secondary"
          size="lg"
          :disabled="isClearing"
          @click="onConfirm"
          >{{ t('g.clear') }}</Button
        >
      </div>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useErrorHandling } from '@/composables/useErrorHandling'
import { useDialogStore } from '@/stores/dialogStore'
import { useQueueStore } from '@/stores/queueStore'

const dialogStore = useDialogStore()
const queueStore = useQueueStore()
const { t } = useI18n()
const { wrapWithErrorHandlingAsync } = useErrorHandling()

const isClearing = ref(false)

const clearHistory = wrapWithErrorHandlingAsync(
  async () => {
    await queueStore.clear(['history'])
    dialogStore.closeDialog()
  },
  undefined,
  () => {
    isClearing.value = false
  }
)

const onConfirm = async () => {
  if (isClearing.value) return
  isClearing.value = true
  await clearHistory()
}

const onCancel = () => {
  dialogStore.closeDialog()
}
</script>
