import { storeToRefs } from 'pinia'

import { getComfyApiBaseUrl } from '@/config/comfyApi'
import { t } from '@/i18n'
import { isCloud } from '@/platform/distribution/types'
import { useTelemetry } from '@/platform/telemetry'
import {
  FirebaseAuthStoreError,
  useFirebaseAuthStore
} from '@/stores/firebaseAuthStore'
import type { CheckoutAttributionMetadata } from '@/platform/telemetry/types'
import type { TierKey } from '@/platform/cloud/subscription/constants/tierPricing'
import type { BillingCycle } from './subscriptionTierRank'

type CheckoutTier = TierKey | `${TierKey}-yearly`

const getCheckoutTier = (
  tierKey: TierKey,
  billingCycle: BillingCycle
): CheckoutTier => (billingCycle === 'yearly' ? `${tierKey}-yearly` : tierKey)

const getCheckoutAttributionForCloud =
  async (): Promise<CheckoutAttributionMetadata> => {
    if (__DISTRIBUTION__ !== 'cloud') {
      return {}
    }

    const { getCheckoutAttribution } =
      await import('@/platform/telemetry/utils/checkoutAttribution')

    return getCheckoutAttribution()
  }

/**
 * Core subscription checkout logic shared between PricingTable and
 * SubscriptionRedirectView. Handles:
 * - Ensuring the user is authenticated
 * - Calling the backend checkout endpoint
 * - Normalizing error responses
 * - Opening the checkout URL in a new tab when available
 *
 * Callers are responsible for:
 * - Guarding on cloud-only behavior (isCloud)
 * - Managing loading state
 * - Wrapping with error handling (e.g. useErrorHandling)
 */
export async function performSubscriptionCheckout(
  tierKey: TierKey,
  currentBillingCycle: BillingCycle,
  openInNewTab: boolean = true
): Promise<void> {
  if (!isCloud) return

  const firebaseAuthStore = useFirebaseAuthStore()
  const { userId } = storeToRefs(firebaseAuthStore)
  const telemetry = useTelemetry()
  const authHeader = await firebaseAuthStore.getAuthHeader()

  if (!authHeader) {
    throw new FirebaseAuthStoreError(t('toastMessages.userNotAuthenticated'))
  }

  const checkoutTier = getCheckoutTier(tierKey, currentBillingCycle)
  let checkoutAttribution: CheckoutAttributionMetadata = {}
  try {
    checkoutAttribution = await getCheckoutAttributionForCloud()
  } catch (error) {
    console.warn(
      '[SubscriptionCheckout] Failed to collect checkout attribution',
      error
    )
  }
  const checkoutPayload = { ...checkoutAttribution }

  const response = await fetch(
    `${getComfyApiBaseUrl()}/customers/cloud-subscription-checkout/${checkoutTier}`,
    {
      method: 'POST',
      headers: { ...authHeader, 'Content-Type': 'application/json' },
      body: JSON.stringify(checkoutPayload)
    }
  )

  if (!response.ok) {
    let errorMessage = 'Failed to initiate checkout'
    try {
      const errorData = await response.json()
      errorMessage = errorData.message || errorMessage
    } catch {
      // If JSON parsing fails, try to get text response or use HTTP status
      try {
        const errorText = await response.text()
        errorMessage =
          errorText || `HTTP ${response.status} ${response.statusText}`
      } catch {
        errorMessage = `HTTP ${response.status} ${response.statusText}`
      }
    }

    throw new FirebaseAuthStoreError(
      t('toastMessages.failedToInitiateSubscription', {
        error: errorMessage
      })
    )
  }

  const data = await response.json()

  if (data.checkout_url) {
    if (userId.value) {
      telemetry?.trackBeginCheckout({
        user_id: userId.value,
        tier: tierKey,
        cycle: currentBillingCycle,
        checkout_type: 'new',
        ...checkoutAttribution
      })
    }
    if (openInNewTab) {
      window.open(data.checkout_url, '_blank')
    } else {
      globalThis.location.href = data.checkout_url
    }
  }
}
