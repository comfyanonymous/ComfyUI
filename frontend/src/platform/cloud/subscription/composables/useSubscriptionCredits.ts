import { computed, toValue } from 'vue'
import { useI18n } from 'vue-i18n'

import { formatCreditsFromCents } from '@/base/credits/comfyCredits'
import { useBillingContext } from '@/composables/billing/useBillingContext'

/**
 * Composable for handling subscription credit calculations and formatting.
 *
 * Uses useBillingContext which automatically selects the correct billing source:
 * - If team workspaces feature is disabled: uses legacy (/customers)
 * - If team workspaces feature is enabled:
 *   - Personal workspace: uses legacy (/customers)
 *   - Team workspace: uses workspace (/billing)
 */
/**
 * Formats a cent value to display credits.
 * Backend returns cents despite the *_micros naming convention.
 */
function formatBalance(maybeCents: number | undefined, locale: string): string {
  const cents = maybeCents ?? 0
  return formatCreditsFromCents({
    cents,
    locale,
    numberOptions: {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }
  })
}

export function useSubscriptionCredits() {
  const billingContext = useBillingContext()
  const { locale } = useI18n()

  const totalCredits = computed(() => {
    const balance = toValue(billingContext.balance)
    return formatBalance(balance?.amountMicros, locale.value)
  })

  const monthlyBonusCredits = computed(() => {
    const balance = toValue(billingContext.balance)
    return formatBalance(balance?.cloudCreditBalanceMicros, locale.value)
  })

  const prepaidCredits = computed(() => {
    const balance = toValue(billingContext.balance)
    return formatBalance(balance?.prepaidBalanceMicros, locale.value)
  })

  const isLoadingBalance = computed(() => toValue(billingContext.isLoading))

  return {
    totalCredits,
    monthlyBonusCredits,
    prepaidCredits,
    isLoadingBalance
  }
}
