import { computed, ref, watch } from 'vue'
import { createSharedComposable } from '@vueuse/core'

import { useCurrentUser } from '@/composables/auth/useCurrentUser'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import { useErrorHandling } from '@/composables/useErrorHandling'
import { getComfyApiBaseUrl, getComfyPlatformBaseUrl } from '@/config/comfyApi'
import { t } from '@/i18n'
import { isCloud } from '@/platform/distribution/types'
import { useTelemetry } from '@/platform/telemetry'
import type { SubscriptionDialogReason } from '@/platform/cloud/subscription/composables/useSubscriptionDialog'
import type { CheckoutAttributionMetadata } from '@/platform/telemetry/types'
import {
  FirebaseAuthStoreError,
  useFirebaseAuthStore
} from '@/stores/firebaseAuthStore'
import { useDialogService } from '@/services/dialogService'
import { TIER_TO_KEY } from '@/platform/cloud/subscription/constants/tierPricing'
import type { operations } from '@/types/comfyRegistryTypes'
import { useSubscriptionCancellationWatcher } from './useSubscriptionCancellationWatcher'

type CloudSubscriptionCheckoutResponse = NonNullable<
  operations['createCloudSubscriptionCheckout']['responses']['201']['content']['application/json']
>

export type CloudSubscriptionStatusResponse = NonNullable<
  operations['GetCloudSubscriptionStatus']['responses']['200']['content']['application/json']
>

function useSubscriptionInternal() {
  const subscriptionStatus = ref<CloudSubscriptionStatusResponse | null>(null)
  const telemetry = useTelemetry()
  const isInitialized = ref(false)

  const isSubscribedOrIsNotCloud = computed(() => {
    if (!isCloud || !window.__CONFIG__?.subscription_required) return true

    return subscriptionStatus.value?.is_active ?? false
  })
  const { reportError, accessBillingPortal } = useFirebaseAuthActions()
  const { showSubscriptionRequiredDialog } = useDialogService()

  const firebaseAuthStore = useFirebaseAuthStore()
  const { getAuthHeader } = firebaseAuthStore
  const { wrapWithErrorHandlingAsync } = useErrorHandling()

  const { isLoggedIn } = useCurrentUser()

  const isCancelled = computed(() => {
    return !!subscriptionStatus.value?.end_date
  })

  const formattedRenewalDate = computed(() => {
    if (!subscriptionStatus.value?.renewal_date) return ''

    const renewalDate = new Date(subscriptionStatus.value.renewal_date)

    return renewalDate.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    })
  })

  const formattedEndDate = computed(() => {
    if (!subscriptionStatus.value?.end_date) return ''

    const endDate = new Date(subscriptionStatus.value.end_date)

    return endDate.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    })
  })

  const subscriptionTier = computed(
    () => subscriptionStatus.value?.subscription_tier ?? null
  )

  const isFreeTier = computed(() => subscriptionTier.value === 'FREE')

  const subscriptionDuration = computed(
    () => subscriptionStatus.value?.subscription_duration ?? null
  )

  const isYearlySubscription = computed(
    () => subscriptionDuration.value === 'ANNUAL'
  )

  const subscriptionTierName = computed(() => {
    const tier = subscriptionTier.value
    if (!tier) return ''
    const key = TIER_TO_KEY[tier] ?? 'standard'
    const baseName = t(`subscription.tiers.${key}.name`)
    return isYearlySubscription.value
      ? t('subscription.tierNameYearly', { name: baseName })
      : baseName
  })

  function buildApiUrl(path: string): string {
    return `${getComfyApiBaseUrl()}${path}`
  }

  const getCheckoutAttributionForCloud =
    async (): Promise<CheckoutAttributionMetadata> => {
      if (__DISTRIBUTION__ !== 'cloud') {
        return {}
      }

      const { getCheckoutAttribution } =
        await import('@/platform/telemetry/utils/checkoutAttribution')

      return getCheckoutAttribution()
    }

  const fetchStatus = wrapWithErrorHandlingAsync(
    fetchSubscriptionStatus,
    reportError
  )

  const subscribe = wrapWithErrorHandlingAsync(async () => {
    const response = await initiateSubscriptionCheckout()

    if (!response.checkout_url) {
      throw new Error(
        t('toastMessages.failedToInitiateSubscription', {
          error: 'No checkout URL returned'
        })
      )
    }

    window.open(response.checkout_url, '_blank')
  }, reportError)

  const showSubscriptionDialog = (options?: {
    reason?: SubscriptionDialogReason
  }) => {
    if (isCloud) {
      useTelemetry()?.trackSubscription('modal_opened', {
        current_tier: subscriptionTier.value?.toLowerCase(),
        reason: options?.reason
      })
    }

    void showSubscriptionRequiredDialog(options)
  }

  /**
   * Whether cloud subscription mode is enabled (cloud distribution with subscription_required config).
   * Use to determine which UI to show (SubscriptionPanel vs LegacyCreditsPanel).
   */
  const isSubscriptionEnabled = (): boolean =>
    Boolean(isCloud && window.__CONFIG__?.subscription_required)

  const { startCancellationWatcher, stopCancellationWatcher } =
    useSubscriptionCancellationWatcher({
      fetchStatus,
      isActiveSubscription: isSubscribedOrIsNotCloud,
      subscriptionStatus,
      telemetry,
      shouldWatchCancellation: isSubscriptionEnabled
    })

  const manageSubscription = async () => {
    await accessBillingPortal()
    startCancellationWatcher()
  }

  const requireActiveSubscription = async (): Promise<void> => {
    await fetchSubscriptionStatus()

    if (!isSubscribedOrIsNotCloud.value) {
      showSubscriptionDialog()
    }
  }

  const handleViewUsageHistory = () => {
    window.open(`${getComfyPlatformBaseUrl()}/profile/usage`, '_blank')
  }

  const handleLearnMore = () => {
    window.open('https://docs.comfy.org', '_blank')
  }

  const handleInvoiceHistory = async () => {
    await accessBillingPortal()
  }

  /**
   * Fetch the current cloud subscription status for the authenticated user
   * @returns Subscription status or null if no subscription exists
   */
  async function fetchSubscriptionStatus(): Promise<CloudSubscriptionStatusResponse | null> {
    const authHeader = await getAuthHeader()
    if (!authHeader) {
      throw new FirebaseAuthStoreError(t('toastMessages.userNotAuthenticated'))
    }

    const response = await fetch(
      buildApiUrl('/customers/cloud-subscription-status'),
      {
        headers: {
          ...authHeader,
          'Content-Type': 'application/json'
        }
      }
    )

    if (!response.ok) {
      const errorData = await response.json()
      throw new FirebaseAuthStoreError(
        t('toastMessages.failedToFetchSubscription', {
          error: errorData.message
        })
      )
    }

    const statusData = await response.json()
    subscriptionStatus.value = statusData

    return statusData
  }

  watch(
    () => isLoggedIn.value,
    async (loggedIn) => {
      if (loggedIn && isCloud) {
        try {
          await fetchSubscriptionStatus()
        } catch (error) {
          // Network errors are expected during navigation/component unmount
          // and when offline - log for debugging but don't surface to user
          console.error('Failed to fetch subscription status:', error)
        } finally {
          isInitialized.value = true
        }
      } else {
        subscriptionStatus.value = null
        stopCancellationWatcher()
        isInitialized.value = true
      }
    },
    { immediate: true }
  )

  const initiateSubscriptionCheckout =
    async (): Promise<CloudSubscriptionCheckoutResponse> => {
      const authHeader = await getAuthHeader()
      if (!authHeader) {
        throw new FirebaseAuthStoreError(
          t('toastMessages.userNotAuthenticated')
        )
      }
      const checkoutAttribution = await getCheckoutAttributionForCloud()

      const response = await fetch(
        buildApiUrl('/customers/cloud-subscription-checkout'),
        {
          method: 'POST',
          headers: {
            ...authHeader,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(checkoutAttribution)
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new FirebaseAuthStoreError(
          t('toastMessages.failedToInitiateSubscription', {
            error: errorData.message
          })
        )
      }

      return response.json()
    }

  return {
    // State
    isActiveSubscription: isSubscribedOrIsNotCloud,
    isInitialized,
    isCancelled,
    formattedRenewalDate,
    formattedEndDate,
    subscriptionTier,
    isFreeTier,
    subscriptionDuration,
    isYearlySubscription,
    subscriptionTierName,
    subscriptionStatus,

    // Utilities
    isSubscriptionEnabled,

    // Actions
    subscribe,
    fetchStatus,
    showSubscriptionDialog,
    manageSubscription,
    requireActiveSubscription,
    handleViewUsageHistory,
    handleLearnMore,
    handleInvoiceHistory
  }
}

export const useSubscription = createSharedComposable(useSubscriptionInternal)
