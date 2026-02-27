import { computed, ref, shallowRef, toValue, watch } from 'vue'
import { createSharedComposable } from '@vueuse/core'

import { useFeatureFlags } from '@/composables/useFeatureFlags'
import {
  KEY_TO_TIER,
  getTierFeatures
} from '@/platform/cloud/subscription/constants/tierPricing'
import type { TierKey } from '@/platform/cloud/subscription/constants/tierPricing'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'

import type {
  BalanceInfo,
  BillingActions,
  BillingContext,
  BillingType,
  BillingState,
  SubscriptionInfo
} from './types'
import { useLegacyBilling } from './useLegacyBilling'
import { useWorkspaceBilling } from '@/platform/workspace/composables/useWorkspaceBilling'

/**
 * Unified billing context that automatically switches between legacy (user-scoped)
 * and workspace billing based on the active workspace type.
 *
 * - Personal workspaces use legacy billing via /customers/* endpoints
 * - Team workspaces use workspace billing via /billing/* endpoints
 *
 * The context automatically initializes when the workspace changes and provides
 * a unified interface for subscription status, balance, and billing actions.
 *
 * @example
 * ```typescript
 * const {
 *   type,
 *   subscription,
 *   balance,
 *   isInitialized,
 *   initialize,
 *   subscribe
 * } = useBillingContext()
 *
 * // Wait for initialization
 * await initialize()
 *
 * // Check subscription status
 * if (subscription.value?.isActive) {
 *   console.log(`Tier: ${subscription.value.tier}`)
 * }
 *
 * // Check balance
 * if (balance.value) {
 *   const dollars = balance.value.amountMicros / 1_000_000
 *   console.log(`Balance: $${dollars.toFixed(2)}`)
 * }
 * ```
 */
function useBillingContextInternal(): BillingContext {
  const store = useTeamWorkspaceStore()
  const { flags } = useFeatureFlags()

  const legacyBillingRef = shallowRef<(BillingState & BillingActions) | null>(
    null
  )
  const workspaceBillingRef = shallowRef<
    (BillingState & BillingActions) | null
  >(null)

  const getLegacyBilling = () => {
    if (!legacyBillingRef.value) {
      legacyBillingRef.value = useLegacyBilling()
    }
    return legacyBillingRef.value
  }

  const getWorkspaceBilling = () => {
    if (!workspaceBillingRef.value) {
      workspaceBillingRef.value = useWorkspaceBilling()
    }
    return workspaceBillingRef.value
  }

  const isInitialized = ref(false)
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  /**
   * Determines which billing type to use:
   * - If team workspaces feature is disabled: always use legacy (/customers)
   * - If team workspaces feature is enabled:
   *   - Personal workspace: use legacy (/customers)
   *   - Team workspace: use workspace (/billing)
   */
  const type = computed<BillingType>(() => {
    if (!flags.teamWorkspacesEnabled) return 'legacy'
    return store.isInPersonalWorkspace ? 'legacy' : 'workspace'
  })

  const activeContext = computed(() =>
    type.value === 'legacy' ? getLegacyBilling() : getWorkspaceBilling()
  )

  // Proxy state from active context
  const subscription = computed<SubscriptionInfo | null>(() =>
    toValue(activeContext.value.subscription)
  )

  const balance = computed<BalanceInfo | null>(() =>
    toValue(activeContext.value.balance)
  )

  const plans = computed(() => toValue(activeContext.value.plans))

  const currentPlanSlug = computed(() =>
    toValue(activeContext.value.currentPlanSlug)
  )

  const isActiveSubscription = computed(() =>
    toValue(activeContext.value.isActiveSubscription)
  )

  const isFreeTier = computed(() => subscription.value?.tier === 'FREE')

  function getMaxSeats(tierKey: TierKey): number {
    if (type.value === 'legacy') return 1

    const apiTier = KEY_TO_TIER[tierKey]
    const plan = plans.value.find(
      (p) => p.tier === apiTier && p.duration === 'MONTHLY'
    )
    return plan?.max_seats ?? getTierFeatures(tierKey).maxMembers
  }

  // Sync subscription info to workspace store for display in workspace switcher
  // A subscription is considered "subscribed" for workspace purposes if it's active AND not cancelled
  // This ensures the delete button is enabled after cancellation, even before the period ends
  watch(
    subscription,
    (sub) => {
      if (!sub || store.isInPersonalWorkspace) return

      store.updateActiveWorkspace({
        isSubscribed: sub.isActive && !sub.isCancelled,
        subscriptionPlan: sub.planSlug
      })
    },
    { immediate: true }
  )

  // Initialize billing when workspace changes
  watch(
    () => store.activeWorkspace?.id,
    async (newWorkspaceId, oldWorkspaceId) => {
      if (!newWorkspaceId) {
        // No workspace selected - reset state
        isInitialized.value = false
        error.value = null
        return
      }

      if (newWorkspaceId !== oldWorkspaceId) {
        // Workspace changed - reinitialize
        isInitialized.value = false
        try {
          await initialize()
        } catch (err) {
          // Error is already captured in error ref
          console.error('Failed to initialize billing context:', err)
        }
      }
    },
    { immediate: true }
  )

  async function initialize(): Promise<void> {
    if (isInitialized.value) return

    isLoading.value = true
    error.value = null
    try {
      await activeContext.value.initialize()
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
    return activeContext.value.fetchStatus()
  }

  async function fetchBalance(): Promise<void> {
    return activeContext.value.fetchBalance()
  }

  async function subscribe(
    planSlug: string,
    returnUrl?: string,
    cancelUrl?: string
  ) {
    return activeContext.value.subscribe(planSlug, returnUrl, cancelUrl)
  }

  async function previewSubscribe(planSlug: string) {
    return activeContext.value.previewSubscribe(planSlug)
  }

  async function manageSubscription() {
    return activeContext.value.manageSubscription()
  }

  async function cancelSubscription() {
    return activeContext.value.cancelSubscription()
  }

  async function fetchPlans() {
    return activeContext.value.fetchPlans()
  }

  async function requireActiveSubscription() {
    return activeContext.value.requireActiveSubscription()
  }

  function showSubscriptionDialog() {
    return activeContext.value.showSubscriptionDialog()
  }

  return {
    type,
    isInitialized,
    subscription,
    balance,
    plans,
    currentPlanSlug,
    isLoading,
    error,
    isActiveSubscription,
    isFreeTier,
    getMaxSeats,

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

export const useBillingContext = createSharedComposable(
  useBillingContextInternal
)
