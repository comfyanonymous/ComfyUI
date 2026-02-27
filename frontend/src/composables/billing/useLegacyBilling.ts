import { computed, ref } from 'vue'

import { useSubscription } from '@/platform/cloud/subscription/composables/useSubscription'
import type {
  PreviewSubscribeResponse,
  SubscribeResponse
} from '@/platform/workspace/api/workspaceApi'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'

import type {
  BalanceInfo,
  BillingActions,
  BillingState,
  SubscriptionInfo
} from './types'

/**
 * Adapter for legacy user-scoped billing via /customers/* endpoints.
 * Used for personal workspaces.
 * @internal - Use useBillingContext() instead of importing directly.
 */
export function useLegacyBilling(): BillingState & BillingActions {
  const {
    isActiveSubscription: legacyIsActiveSubscription,
    subscriptionTier,
    subscriptionDuration,
    formattedRenewalDate,
    formattedEndDate,
    isCancelled,
    fetchStatus: legacyFetchStatus,
    manageSubscription: legacyManageSubscription,
    subscribe: legacySubscribe,
    showSubscriptionDialog: legacyShowSubscriptionDialog
  } = useSubscription()

  const firebaseAuthStore = useFirebaseAuthStore()

  const isInitialized = ref(false)
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  const isActiveSubscription = computed(() => legacyIsActiveSubscription.value)
  const isFreeTier = computed(() => subscriptionTier.value === 'FREE')

  const subscription = computed<SubscriptionInfo | null>(() => {
    if (!legacyIsActiveSubscription.value && !subscriptionTier.value) {
      return null
    }

    return {
      isActive: legacyIsActiveSubscription.value,
      tier: subscriptionTier.value,
      duration: subscriptionDuration.value,
      planSlug: null, // Legacy doesn't use plan slugs
      renewalDate: formattedRenewalDate.value || null,
      endDate: formattedEndDate.value || null,
      isCancelled: isCancelled.value,
      hasFunds: (firebaseAuthStore.balance?.amount_micros ?? 0) > 0
    }
  })

  const balance = computed<BalanceInfo | null>(() => {
    const legacyBalance = firebaseAuthStore.balance
    if (!legacyBalance) return null

    return {
      amountMicros: legacyBalance.amount_micros ?? 0,
      currency: legacyBalance.currency ?? 'usd',
      effectiveBalanceMicros:
        legacyBalance.effective_balance_micros ??
        legacyBalance.amount_micros ??
        0,
      prepaidBalanceMicros: legacyBalance.prepaid_balance_micros ?? 0,
      cloudCreditBalanceMicros: legacyBalance.cloud_credit_balance_micros ?? 0
    }
  })

  // Legacy billing doesn't have workspace-style plans
  const plans = computed(() => [])
  const currentPlanSlug = computed(() => null)

  async function initialize(): Promise<void> {
    if (isInitialized.value) return

    isLoading.value = true
    error.value = null
    try {
      await Promise.all([fetchStatus(), fetchBalance()])
      // Re-fetch balance if free tier credits were just lazily granted
      if (isFreeTier.value && balance.value?.amountMicros === 0) {
        await fetchBalance()
      }
      isInitialized.value = true
    } catch (err) {
      error.value =
        err instanceof Error ? err.message : 'Failed to initialize billing'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function fetchStatus(): Promise<void> {
    isLoading.value = true
    error.value = null
    try {
      await legacyFetchStatus()
    } catch (err) {
      error.value =
        err instanceof Error ? err.message : 'Failed to fetch subscription'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function fetchBalance(): Promise<void> {
    isLoading.value = true
    error.value = null
    try {
      await firebaseAuthStore.fetchBalance()
    } catch (err) {
      error.value =
        err instanceof Error ? err.message : 'Failed to fetch balance'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function subscribe(
    _planSlug: string,
    _returnUrl?: string,
    _cancelUrl?: string
  ): Promise<SubscribeResponse | void> {
    // Legacy billing uses Stripe checkout flow via useSubscription
    await legacySubscribe()
  }

  async function previewSubscribe(
    _planSlug: string
  ): Promise<PreviewSubscribeResponse | null> {
    // Legacy billing doesn't support preview - returns null
    return null
  }

  async function manageSubscription(): Promise<void> {
    await legacyManageSubscription()
  }

  async function cancelSubscription(): Promise<void> {
    await legacyManageSubscription()
  }

  async function fetchPlans(): Promise<void> {
    // Legacy billing doesn't have workspace-style plans
    // Plans are hardcoded in the UI for legacy subscriptions
  }

  async function requireActiveSubscription(): Promise<void> {
    await fetchStatus()
    if (!isActiveSubscription.value) {
      legacyShowSubscriptionDialog()
    }
  }

  function showSubscriptionDialog(): void {
    legacyShowSubscriptionDialog()
  }

  return {
    // State
    isInitialized,
    subscription,
    balance,
    plans,
    currentPlanSlug,
    isLoading,
    error,
    isActiveSubscription,
    isFreeTier,

    // Actions
    initialize,
    fetchStatus,
    fetchBalance,
    subscribe,
    previewSubscribe,
    manageSubscription,
    cancelSubscription,
    fetchPlans,
    requireActiveSubscription,
    showSubscriptionDialog
  }
}
