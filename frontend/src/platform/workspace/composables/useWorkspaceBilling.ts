import { computed, onBeforeUnmount, ref, shallowRef } from 'vue'

import { useBillingPlans } from '@/platform/cloud/subscription/composables/useBillingPlans'
import { useSubscriptionDialog } from '@/platform/cloud/subscription/composables/useSubscriptionDialog'
import type {
  BillingBalanceResponse,
  BillingStatusResponse,
  PreviewSubscribeResponse,
  SubscribeResponse
} from '@/platform/workspace/api/workspaceApi'
import { workspaceApi } from '@/platform/workspace/api/workspaceApi'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'

import type {
  BalanceInfo,
  BillingActions,
  BillingState,
  SubscriptionInfo
} from '../../../composables/billing/types'

/**
 * Adapter for workspace-scoped billing via /billing/* endpoints.
 * Used for team workspaces.
 * @internal - Use useBillingContext() instead of importing directly.
 */
export function useWorkspaceBilling(): BillingState & BillingActions {
  const billingPlans = useBillingPlans()
  const workspaceStore = useTeamWorkspaceStore()

  const isInitialized = ref(false)
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  const statusData = shallowRef<BillingStatusResponse | null>(null)
  const balanceData = shallowRef<BillingBalanceResponse | null>(null)

  const isActiveSubscription = computed(
    () => statusData.value?.is_active ?? false
  )
  const isFreeTier = computed(
    () => statusData.value?.subscription_tier === 'FREE'
  )

  const subscription = computed<SubscriptionInfo | null>(() => {
    const status = statusData.value
    if (!status) return null

    return {
      isActive: status.is_active,
      tier: status.subscription_tier ?? null,
      duration: status.subscription_duration ?? null,
      planSlug: status.plan_slug ?? null,
      renewalDate: status.renewal_date ?? null,
      endDate: status.cancel_at ?? null,
      isCancelled: status.subscription_status === 'canceled',
      hasFunds: status.has_funds
    }
  })

  const balance = computed<BalanceInfo | null>(() => {
    const data = balanceData.value
    if (!data) return null

    return {
      amountMicros: data.amount_micros,
      currency: data.currency,
      effectiveBalanceMicros: data.effective_balance_micros,
      prepaidBalanceMicros: data.prepaid_balance_micros,
      cloudCreditBalanceMicros: data.cloud_credit_balance_micros
    }
  })

  const plans = computed(() => billingPlans.plans.value)
  const currentPlanSlug = computed(
    () => statusData.value?.plan_slug ?? billingPlans.currentPlanSlug.value
  )

  const pendingCancelOpId = ref<string | null>(null)
  let cancelPollTimeout: number | null = null

  const stopCancelPolling = () => {
    if (cancelPollTimeout !== null) {
      window.clearTimeout(cancelPollTimeout)
      cancelPollTimeout = null
    }
  }

  async function pollCancelStatus(opId: string): Promise<void> {
    stopCancelPolling()

    const maxAttempts = 30
    let attempt = 0
    const poll = async () => {
      if (pendingCancelOpId.value !== opId) return

      try {
        const response = await workspaceApi.getBillingOpStatus(opId)
        if (response.status === 'succeeded') {
          pendingCancelOpId.value = null
          stopCancelPolling()
          await fetchStatus()
          workspaceStore.updateActiveWorkspace({
            isSubscribed: false
          })
          return
        }

        if (response.status === 'failed') {
          pendingCancelOpId.value = null
          stopCancelPolling()
          throw new Error(
            response.error_message ?? 'Failed to cancel subscription'
          )
        }

        attempt += 1
        if (attempt >= maxAttempts) {
          pendingCancelOpId.value = null
          stopCancelPolling()
          await fetchStatus()
          return
        }
      } catch (err) {
        pendingCancelOpId.value = null
        stopCancelPolling()
        throw err
      }

      cancelPollTimeout = window.setTimeout(
        () => {
          void poll()
        },
        Math.min(1000 * 2 ** attempt, 5000)
      )
    }

    await poll()
  }

  async function initialize(): Promise<void> {
    if (isInitialized.value) return

    isLoading.value = true
    error.value = null
    try {
      await Promise.all([fetchStatus(), fetchBalance(), fetchPlans()])
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
      statusData.value = await workspaceApi.getBillingStatus()
    } catch (err) {
      error.value =
        err instanceof Error ? err.message : 'Failed to fetch billing status'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function fetchBalance(): Promise<void> {
    isLoading.value = true
    error.value = null
    try {
      balanceData.value = await workspaceApi.getBillingBalance()
    } catch (err) {
      error.value =
        err instanceof Error ? err.message : 'Failed to fetch balance'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function subscribe(
    planSlug: string,
    returnUrl?: string,
    cancelUrl?: string
  ): Promise<SubscribeResponse> {
    isLoading.value = true
    error.value = null
    try {
      const response = await workspaceApi.subscribe(
        planSlug,
        returnUrl,
        cancelUrl
      )

      // Refresh status and balance after subscription
      await Promise.all([fetchStatus(), fetchBalance()])

      return response
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to subscribe'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function previewSubscribe(
    planSlug: string
  ): Promise<PreviewSubscribeResponse | null> {
    isLoading.value = true
    error.value = null
    try {
      return await workspaceApi.previewSubscribe(planSlug)
    } catch (err) {
      error.value =
        err instanceof Error ? err.message : 'Failed to preview subscription'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function manageSubscription(): Promise<void> {
    isLoading.value = true
    error.value = null
    try {
      const returnUrl = window.location.href
      const response = await workspaceApi.getPaymentPortalUrl(returnUrl)
      if (response.url) {
        window.open(response.url, '_blank')
      }
    } catch (err) {
      error.value =
        err instanceof Error ? err.message : 'Failed to open billing portal'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function cancelSubscription(): Promise<void> {
    isLoading.value = true
    error.value = null
    try {
      const response = await workspaceApi.cancelSubscription()
      pendingCancelOpId.value = response.billing_op_id
      await pollCancelStatus(response.billing_op_id)
    } catch (err) {
      error.value =
        err instanceof Error ? err.message : 'Failed to cancel subscription'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function fetchPlans(): Promise<void> {
    isLoading.value = true
    error.value = null
    try {
      await billingPlans.fetchPlans()
      if (billingPlans.error.value) {
        error.value = billingPlans.error.value
      }
    } finally {
      isLoading.value = false
    }
  }

  const subscriptionDialog = useSubscriptionDialog()

  async function requireActiveSubscription(): Promise<void> {
    await fetchStatus()
    if (!isActiveSubscription.value) {
      subscriptionDialog.show()
    }
  }

  function showSubscriptionDialog(): void {
    subscriptionDialog.show()
  }

  onBeforeUnmount(() => {
    stopCancelPolling()
  })

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
