<template>
  <div
    class="relative flex flex-col p-4 pt-8 md:p-16 !overflow-y-auto h-full gap-8"
  >
    <Button
      v-if="checkoutStep === 'preview'"
      size="icon"
      variant="muted-textonly"
      class="rounded-full shrink-0 text-text-secondary hover:bg-white/10 absolute left-2.5 top-2.5"
      :aria-label="$t('g.back')"
      @click="handleBackToPricing"
    >
      <i class="pi pi-arrow-left text-xl" />
    </Button>

    <Button
      size="icon"
      variant="muted-textonly"
      class="rounded-full shrink-0 text-text-secondary hover:bg-white/10 absolute right-2.5 top-2.5"
      :aria-label="$t('g.close')"
      @click="handleClose"
    >
      <i class="pi pi-times text-xl" />
    </Button>

    <div v-if="reason === 'out_of_credits'" class="text-center">
      <h2 class="text-xl lg:text-2xl text-muted-foreground m-0">
        {{ $t('credits.topUp.insufficientTitle') }}
      </h2>
      <p class="m-0 mt-2 text-sm text-text-secondary">
        {{ $t('credits.topUp.insufficientMessage') }}
      </p>
    </div>

    <!-- Pricing Table Step -->
    <PricingTableWorkspace
      v-if="checkoutStep === 'pricing'"
      class="flex-1"
      :is-loading="isLoadingPreview || isResubscribing"
      :loading-tier="loadingTier"
      @subscribe="handleSubscribeClick"
      @resubscribe="handleResubscribe"
    />

    <!-- Subscription Preview Step - New Subscription -->
    <SubscriptionAddPaymentPreviewWorkspace
      v-else-if="
        checkoutStep === 'preview' &&
        previewData &&
        previewData.transition_type === 'new_subscription'
      "
      :preview-data="previewData"
      :tier-key="selectedTierKey!"
      :billing-cycle="selectedBillingCycle"
      :is-loading="isSubscribing || isPolling"
      @add-credit-card="handleAddCreditCard"
      @back="handleBackToPricing"
    />

    <!-- Subscription Preview Step - Plan Transition -->
    <SubscriptionTransitionPreviewWorkspace
      v-else-if="
        checkoutStep === 'preview' &&
        previewData &&
        previewData.transition_type !== 'new_subscription'
      "
      :preview-data="previewData"
      :is-loading="isSubscribing || isPolling"
      @confirm="handleConfirmTransition"
      @back="handleBackToPricing"
    />
  </div>
</template>

<script setup lang="ts">
import { useToast } from 'primevue/usetoast'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import type { TierKey } from '@/platform/cloud/subscription/constants/tierPricing'
import type { BillingCycle } from '@/platform/cloud/subscription/utils/subscriptionTierRank'
import type { PreviewSubscribeResponse } from '@/platform/workspace/api/workspaceApi'
import { workspaceApi } from '@/platform/workspace/api/workspaceApi'
import { useBillingOperationStore } from '@/platform/workspace/stores/billingOperationStore'
import type { SubscriptionDialogReason } from '@/platform/cloud/subscription/composables/useSubscriptionDialog'

import PricingTableWorkspace from './PricingTableWorkspace.vue'
import SubscriptionAddPaymentPreviewWorkspace from './SubscriptionAddPaymentPreviewWorkspace.vue'
import SubscriptionTransitionPreviewWorkspace from './SubscriptionTransitionPreviewWorkspace.vue'

type CheckoutStep = 'pricing' | 'preview'
type CheckoutTierKey = Exclude<TierKey, 'free' | 'founder'>

const { onClose, reason } = defineProps<{
  onClose: () => void
  reason?: SubscriptionDialogReason
}>()

const emit = defineEmits<{
  close: [subscribed: boolean]
}>()

const { t } = useI18n()
const toast = useToast()
const { subscribe, previewSubscribe, plans, fetchStatus, fetchBalance } =
  useBillingContext()

const billingOperationStore = useBillingOperationStore()
const isPolling = computed(() => billingOperationStore.hasPendingOperations)

const checkoutStep = ref<CheckoutStep>('pricing')
const isLoadingPreview = ref(false)
const loadingTier = ref<CheckoutTierKey | null>(null)
const isSubscribing = ref(false)
const isResubscribing = ref(false)
const previewData = ref<PreviewSubscribeResponse | null>(null)
const selectedTierKey = ref<CheckoutTierKey | null>(null)
const selectedBillingCycle = ref<BillingCycle>('yearly')

function getApiPlanSlug(
  tierKey: CheckoutTierKey,
  billingCycle: BillingCycle
): string | null {
  const apiDuration = billingCycle === 'yearly' ? 'ANNUAL' : 'MONTHLY'
  const apiTier = tierKey.toUpperCase()
  const plan = plans.value.find(
    (p) => p.tier === apiTier && p.duration === apiDuration
  )
  return plan?.slug ?? null
}

async function handleSubscribeClick(payload: {
  tierKey: CheckoutTierKey
  billingCycle: BillingCycle
}) {
  const { tierKey, billingCycle } = payload

  isLoadingPreview.value = true
  loadingTier.value = tierKey
  selectedTierKey.value = tierKey
  selectedBillingCycle.value = billingCycle

  try {
    const planSlug = getApiPlanSlug(tierKey, billingCycle)
    if (!planSlug) {
      toast.add({
        severity: 'error',
        summary: 'Unable to subscribe',
        detail: 'This plan is not available',
        life: 5000
      })
      return
    }
    const response = await previewSubscribe(planSlug)

    if (!response || !response.allowed) {
      toast.add({
        severity: 'error',
        summary: 'Unable to subscribe',
        detail: response?.reason || 'This plan is not available',
        life: 5000
      })
      return
    }

    previewData.value = response
    checkoutStep.value = 'preview'
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : 'Failed to load subscription preview'
    toast.add({
      severity: 'error',
      summary: 'Error',
      detail: message,
      life: 5000
    })
  } finally {
    isLoadingPreview.value = false
    loadingTier.value = null
  }
}

function handleBackToPricing() {
  checkoutStep.value = 'pricing'
  previewData.value = null
}

async function handleAddCreditCard() {
  if (!selectedTierKey.value) return

  isSubscribing.value = true
  try {
    const planSlug = getApiPlanSlug(
      selectedTierKey.value,
      selectedBillingCycle.value
    )
    if (!planSlug) return
    const response = await subscribe(
      planSlug,
      'https://www.comfy.org/payment/success',
      'https://www.comfy.org/payment/failed'
    )

    if (!response) return

    if (response.status === 'subscribed') {
      toast.add({
        severity: 'success',
        summary: t('subscription.required.pollingSuccess'),
        life: 5000
      })
      await Promise.all([fetchStatus(), fetchBalance()])
      emit('close', true)
    } else if (
      response.status === 'needs_payment_method' &&
      response.payment_method_url
    ) {
      window.open(response.payment_method_url, '_blank')
      billingOperationStore.startOperation(
        response.billing_op_id,
        'subscription'
      )
    } else if (response.status === 'pending_payment') {
      billingOperationStore.startOperation(
        response.billing_op_id,
        'subscription'
      )
    }
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'Failed to subscribe'
    toast.add({
      severity: 'error',
      summary: 'Error',
      detail: message,
      life: 5000
    })
  } finally {
    isSubscribing.value = false
  }
}

async function handleConfirmTransition() {
  if (!selectedTierKey.value) return

  isSubscribing.value = true
  try {
    const planSlug = getApiPlanSlug(
      selectedTierKey.value,
      selectedBillingCycle.value
    )
    if (!planSlug) return
    const response = await subscribe(
      planSlug,
      'https://www.comfy.org/payment/success',
      'https://www.comfy.org/payment/failed'
    )

    if (!response) return

    if (response.status === 'subscribed') {
      toast.add({
        severity: 'success',
        summary: t('subscription.required.pollingSuccess'),
        life: 5000
      })
      await Promise.all([fetchStatus(), fetchBalance()])
      emit('close', true)
    } else if (
      response.status === 'needs_payment_method' &&
      response.payment_method_url
    ) {
      window.open(response.payment_method_url, '_blank')
      billingOperationStore.startOperation(
        response.billing_op_id,
        'subscription'
      )
    } else if (response.status === 'pending_payment') {
      billingOperationStore.startOperation(
        response.billing_op_id,
        'subscription'
      )
    }
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'Failed to update subscription'
    toast.add({
      severity: 'error',
      summary: 'Error',
      detail: message,
      life: 5000
    })
  } finally {
    isSubscribing.value = false
  }
}

async function handleResubscribe() {
  isResubscribing.value = true
  try {
    await workspaceApi.resubscribe()
    toast.add({
      severity: 'success',
      summary: t('subscription.resubscribeSuccess'),
      life: 5000
    })
    await Promise.all([fetchStatus(), fetchBalance()])
    emit('close', true)
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'Failed to resubscribe'
    toast.add({
      severity: 'error',
      summary: 'Error',
      detail: message,
      life: 5000
    })
  } finally {
    isResubscribing.value = false
  }
}

function handleClose() {
  onClose()
}
</script>

<style scoped>
.legacy-dialog :deep(.bg-comfy-menu-secondary) {
  background-color: transparent;
}

.legacy-dialog :deep(.p-button) {
  color: white;
}
</style>
