import type { ComputedRef, Ref } from 'vue'

import type { TierKey } from '@/platform/cloud/subscription/constants/tierPricing'
import type {
  Plan,
  PreviewSubscribeResponse,
  SubscribeResponse,
  SubscriptionDuration,
  SubscriptionTier
} from '@/platform/workspace/api/workspaceApi'

export type BillingType = 'legacy' | 'workspace'

export interface SubscriptionInfo {
  isActive: boolean
  tier: SubscriptionTier | null
  duration: SubscriptionDuration | null
  planSlug: string | null
  renewalDate: string | null
  endDate: string | null
  isCancelled: boolean
  hasFunds: boolean
}

export interface BalanceInfo {
  amountMicros: number
  currency: string
  effectiveBalanceMicros?: number
  prepaidBalanceMicros?: number
  cloudCreditBalanceMicros?: number
}

export interface BillingActions {
  initialize: () => Promise<void>
  fetchStatus: () => Promise<void>
  fetchBalance: () => Promise<void>
  subscribe: (
    planSlug: string,
    returnUrl?: string,
    cancelUrl?: string
  ) => Promise<SubscribeResponse | void>
  previewSubscribe: (
    planSlug: string
  ) => Promise<PreviewSubscribeResponse | null>
  manageSubscription: () => Promise<void>
  cancelSubscription: () => Promise<void>
  fetchPlans: () => Promise<void>
  /**
   * Ensures billing is initialized and subscription is active.
   * Shows subscription dialog if not subscribed.
   * Use this in extensions/entry points that require active subscription.
   */
  requireActiveSubscription: () => Promise<void>
  /**
   * Shows the subscription dialog.
   */
  showSubscriptionDialog: () => void
}

export interface BillingState {
  isInitialized: Ref<boolean>
  subscription: ComputedRef<SubscriptionInfo | null>
  balance: ComputedRef<BalanceInfo | null>
  plans: ComputedRef<Plan[]>
  currentPlanSlug: ComputedRef<string | null>
  isLoading: Ref<boolean>
  error: Ref<string | null>
  /**
   * Convenience computed for checking if subscription is active.
   * Equivalent to `subscription.value?.isActive ?? false`
   */
  isActiveSubscription: ComputedRef<boolean>
  /**
   * Whether the current billing context has a FREE tier subscription.
   * Workspace-aware: reflects the active workspace's tier, not the user's personal tier.
   */
  isFreeTier: ComputedRef<boolean>
}

export interface BillingContext extends BillingState, BillingActions {
  type: ComputedRef<BillingType>
  getMaxSeats: (tierKey: TierKey) => number
}
