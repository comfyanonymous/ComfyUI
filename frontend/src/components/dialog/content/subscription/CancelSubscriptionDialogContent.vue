<template>
  <div
    class="flex w-full max-w-[400px] flex-col rounded-2xl border border-border-default bg-base-background"
  >
    <!-- Header -->
    <div
      class="flex h-12 items-center justify-between border-b border-border-default px-4"
    >
      <h2 class="m-0 text-sm font-normal text-base-foreground">
        {{ $t('subscription.cancelDialog.title') }}
      </h2>
      <button
        class="cursor-pointer rounded border-none bg-transparent p-0 text-muted-foreground transition-colors hover:text-base-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-secondary-foreground"
        :aria-label="$t('g.close')"
        :disabled="isLoading"
        @click="onClose"
      >
        <i class="pi pi-times size-4" />
      </button>
    </div>

    <!-- Body -->
    <div class="flex flex-col gap-4 px-4 py-4">
      <p class="m-0 text-sm text-muted-foreground">
        {{ description }}
      </p>
    </div>

    <!-- Footer -->
    <div class="flex items-center justify-end gap-4 px-4 py-4">
      <Button variant="muted-textonly" :disabled="isLoading" @click="onClose">
        {{ $t('subscription.cancelDialog.keepSubscription') }}
      </Button>
      <Button
        variant="destructive"
        size="lg"
        :loading="isLoading"
        @click="onConfirmCancel"
      >
        {{ $t('subscription.cancelDialog.confirmCancel') }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useToast } from 'primevue/usetoast'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { useDialogStore } from '@/stores/dialogStore'

const props = defineProps<{
  cancelAt?: string
}>()

const { t } = useI18n()
const dialogStore = useDialogStore()
const toast = useToast()
const { cancelSubscription, fetchStatus, subscription } = useBillingContext()

const isLoading = ref(false)

const formattedEndDate = computed(() => {
  const dateStr = props.cancelAt ?? subscription.value?.endDate
  if (!dateStr) return t('subscription.cancelDialog.endOfBillingPeriod')
  const date = new Date(dateStr)
  return date.toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric'
  })
})

const description = computed(() =>
  t('subscription.cancelDialog.description', { date: formattedEndDate.value })
)

function onClose() {
  if (isLoading.value) return
  dialogStore.closeDialog({ key: 'cancel-subscription' })
}

async function onConfirmCancel() {
  isLoading.value = true
  try {
    await cancelSubscription()
    await fetchStatus()
    dialogStore.closeDialog({ key: 'cancel-subscription' })
    toast.add({
      severity: 'success',
      summary: t('subscription.cancelSuccess'),
      life: 5000
    })
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: t('subscription.cancelDialog.failed'),
      detail: error instanceof Error ? error.message : t('g.unknownError'),
      life: 5000
    })
  } finally {
    isLoading.value = false
  }
}
</script>
