<template>
  <div
    v-if="showCustomPricingTable"
    class="relative flex flex-col p-4 pt-8 md:p-16 !overflow-y-auto h-full gap-8"
  >
    <Button
      size="icon"
      variant="muted-textonly"
      class="rounded-full shrink-0 text-text-secondary hover:bg-white/10 absolute right-2.5 top-2.5"
      :aria-label="$t('g.close')"
      @click="handleClose"
    >
      <i class="pi pi-times text-xl" />
    </Button>
    <div class="text-center">
      <h2 class="text-xl lg:text-2xl text-muted-foreground m-0">
        {{ $t('subscription.description') }}
      </h2>
    </div>

    <PricingTable class="flex-1" />

    <!-- Contact and Enterprise Links -->
    <div class="flex flex-col items-center gap-2">
      <p class="text-sm text-text-secondary m-0">
        {{ $t('subscription.haveQuestions') }}
      </p>
      <div class="flex items-center gap-1.5">
        <Button
          variant="muted-textonly"
          class="h-6 p-1 text-sm text-text-secondary hover:text-base-foreground"
          @click="handleContactUs"
        >
          {{ $t('subscription.contactUs') }}
          <i class="pi pi-comments" />
        </Button>
        <span class="text-sm text-text-secondary">{{ $t('g.or') }}</span>
        <Button
          variant="muted-textonly"
          class="h-6 p-1 text-sm text-text-secondary hover:text-base-foreground"
          @click="handleViewEnterprise"
        >
          {{ $t('subscription.viewEnterprise') }}
          <i class="pi pi-external-link" />
        </Button>
      </div>
    </div>
  </div>
  <div v-else class="legacy-dialog relative grid h-full grid-cols-5">
    <!-- Custom close button -->
    <Button
      size="icon"
      variant="muted-textonly"
      class="rounded-full absolute top-2.5 right-2.5 z-10 h-8 w-8 p-0 text-white hover:bg-white/20"
      :aria-label="$t('g.close')"
      @click="handleClose"
    >
      <i class="pi pi-times" />
    </Button>

    <div
      class="relative col-span-2 flex items-center justify-center overflow-hidden rounded-sm"
    >
      <video
        autoplay
        loop
        muted
        playsinline
        class="h-full min-w-[125%] object-cover p-0"
        style="margin-left: -20%"
      >
        <source
          src="/assets/images/cloud-subscription.webm"
          type="video/webm"
        />
      </video>
    </div>

    <div class="col-span-3 flex flex-col justify-between p-8">
      <div>
        <div class="flex flex-col gap-6">
          <div class="inline-flex items-center gap-2">
            <div class="text-sm text-text-primary">
              {{
                reason === 'out_of_credits'
                  ? $t('credits.topUp.insufficientTitle')
                  : $t('subscription.required.title')
              }}
            </div>
            <CloudBadge
              reverse-order
              no-padding
              background-color="var(--p-dialog-background)"
              use-subscription
            />
          </div>

          <p
            v-if="reason === 'out_of_credits'"
            class="m-0 text-sm text-text-secondary"
          >
            {{ $t('credits.topUp.insufficientMessage') }}
          </p>

          <div class="flex items-baseline gap-2">
            <span class="text-4xl font-bold">{{ formattedMonthlyPrice }}</span>
            <span class="text-xl">{{ $t('subscription.perMonth') }}</span>
          </div>
        </div>

        <SubscriptionBenefits class="mt-6 text-muted" />
      </div>

      <div class="flex flex-col pt-8">
        <SubscribeButton
          class="py-2 px-4 rounded-lg"
          :pt="{
            root: {
              style: 'background: var(--color-accent-blue, #0B8CE9);'
            },
            label: {
              class: 'font-inter font-[700] text-sm'
            }
          }"
          @subscribed="handleSubscribed"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, watch } from 'vue'

import CloudBadge from '@/components/topbar/CloudBadge.vue'
import Button from '@/components/ui/button/Button.vue'
import { MONTHLY_SUBSCRIPTION_PRICE } from '@/config/subscriptionPricesConfig'
import PricingTable from '@/platform/cloud/subscription/components/PricingTable.vue'
import SubscribeButton from '@/platform/cloud/subscription/components/SubscribeButton.vue'
import SubscriptionBenefits from '@/platform/cloud/subscription/components/SubscriptionBenefits.vue'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { isCloud } from '@/platform/distribution/types'
import { useTelemetry } from '@/platform/telemetry'
import { useCommandStore } from '@/stores/commandStore'
import type { SubscriptionDialogReason } from '@/platform/cloud/subscription/composables/useSubscriptionDialog'

const { onClose, reason } = defineProps<{
  onClose: () => void
  reason?: SubscriptionDialogReason
}>()

const emit = defineEmits<{
  close: [subscribed: boolean]
}>()

const { fetchStatus, isActiveSubscription } = useBillingContext()

const isSubscriptionEnabled = (): boolean =>
  Boolean(isCloud && window.__CONFIG__?.subscription_required)

// Legacy price for non-tier flow with locale-aware formatting
const formattedMonthlyPrice = new Intl.NumberFormat(
  navigator.language || 'en-US',
  {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }
).format(MONTHLY_SUBSCRIPTION_PRICE)
const commandStore = useCommandStore()
const telemetry = useTelemetry()

// Always show custom pricing table for cloud subscriptions
const showCustomPricingTable = computed(() => isSubscriptionEnabled())

const POLL_INTERVAL_MS = 3000
const MAX_POLL_ATTEMPTS = 3
let pollInterval: number | null = null
let pollAttempts = 0

const stopPolling = () => {
  if (pollInterval) {
    clearInterval(pollInterval)
    pollInterval = null
  }
}

const startPolling = () => {
  stopPolling()
  pollAttempts = 0

  const poll = async () => {
    try {
      await fetchStatus()
      pollAttempts++

      if (pollAttempts >= MAX_POLL_ATTEMPTS) {
        stopPolling()
      }
    } catch (error) {
      console.error(
        '[SubscriptionDialog] Failed to poll subscription status',
        error
      )
      stopPolling()
    }
  }

  void poll()
  pollInterval = window.setInterval(() => {
    void poll()
  }, POLL_INTERVAL_MS)
}

const handleWindowFocus = () => {
  if (showCustomPricingTable.value) {
    startPolling()
  }
}

watch(
  showCustomPricingTable,
  (enabled) => {
    if (enabled) {
      window.addEventListener('focus', handleWindowFocus)
    } else {
      window.removeEventListener('focus', handleWindowFocus)
      stopPolling()
    }
  },
  { immediate: true }
)

watch(
  () => isActiveSubscription.value,
  (isActive) => {
    if (isActive && showCustomPricingTable.value) {
      emit('close', true)
    }
  }
)

const handleSubscribed = () => {
  emit('close', true)
}

const handleClose = () => {
  stopPolling()
  onClose()
}

const handleContactUs = async () => {
  telemetry?.trackHelpResourceClicked({
    resource_type: 'help_feedback',
    is_external: true,
    source: 'subscription'
  })
  await commandStore.execute('Comfy.ContactSupport')
}

const handleViewEnterprise = () => {
  telemetry?.trackHelpResourceClicked({
    resource_type: 'docs',
    is_external: true,
    source: 'subscription'
  })
  window.open('https://www.comfy.org/cloud/enterprise', '_blank')
}

onBeforeUnmount(() => {
  stopPolling()
  window.removeEventListener('focus', handleWindowFocus)
})
</script>

<style scoped>
.legacy-dialog :deep(.bg-comfy-menu-secondary) {
  background-color: transparent;
}

.legacy-dialog :deep(.p-button) {
  color: white;
}
</style>
