<template>
  <div
    class="flex flex-col justify-between gap-10 p-4 border-t border-border-default w-auto max-w-[min(500px,90vw)]"
  >
    <UploadModelUpgradeModalBody />

    <UploadModelUpgradeModalFooter
      @close="handleClose"
      @subscribe="handleSubscribe"
    />
  </div>
</template>

<script setup lang="ts">
import { useBillingContext } from '@/composables/billing/useBillingContext'
import UploadModelUpgradeModalBody from '@/platform/assets/components/UploadModelUpgradeModalBody.vue'
import UploadModelUpgradeModalFooter from '@/platform/assets/components/UploadModelUpgradeModalFooter.vue'
import { useDialogStore } from '@/stores/dialogStore'

const dialogStore = useDialogStore()
const { showSubscriptionDialog } = useBillingContext()

function handleClose() {
  dialogStore.closeDialog({ key: 'upload-model-upgrade' })
}

function handleSubscribe() {
  showSubscriptionDialog()
}
</script>
