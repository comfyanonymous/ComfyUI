<script setup lang="ts">
import { until } from '@vueuse/core'
import Button from 'primevue/button'
import ProgressSpinner from 'primevue/progressspinner'
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'

import { useBillingContext } from '@/composables/billing/useBillingContext'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import { useErrorHandling } from '@/composables/useErrorHandling'
import type { TierKey } from '@/platform/cloud/subscription/constants/tierPricing'
import { performSubscriptionCheckout } from '@/platform/cloud/subscription/utils/subscriptionCheckoutUtil'

import type { BillingCycle } from '../subscription/utils/subscriptionTierRank'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()
const { reportError, accessBillingPortal } = useFirebaseAuthActions()
const { wrapWithErrorHandlingAsync } = useErrorHandling()

const { isActiveSubscription, isInitialized } = useBillingContext()

const selectedTierKey = ref<TierKey | null>(null)

const tierDisplayName = computed(() => {
  if (!selectedTierKey.value) return ''
  const names: Record<TierKey, string> = {
    free: t('subscription.tiers.free.name'),
    standard: t('subscription.tiers.standard.name'),
    creator: t('subscription.tiers.creator.name'),
    pro: t('subscription.tiers.pro.name'),
    founder: t('subscription.tiers.founder.name')
  }
  return names[selectedTierKey.value]
})

const runRedirect = wrapWithErrorHandlingAsync(async () => {
  const rawType = route.query.tier
  const rawCycle = route.query.cycle
  let tierKeyParam: string | null = null
  let cycleParam = 'monthly'

  if (typeof rawType === 'string') {
    tierKeyParam = rawType
  } else if (Array.isArray(rawType) && rawType[0]) {
    tierKeyParam = rawType[0]
  }

  if (typeof rawCycle === 'string') {
    cycleParam = rawCycle
  } else if (Array.isArray(rawCycle) && rawCycle[0]) {
    cycleParam = rawCycle[0]
  }

  if (!tierKeyParam) {
    await router.push('/')
    return
  }

  // Only paid tiers can be checked out via redirect
  const validTierKeys: TierKey[] = ['standard', 'creator', 'pro', 'founder']
  if (!(validTierKeys as string[]).includes(tierKeyParam)) {
    await router.push('/')
    return
  }

  const tierKey = tierKeyParam as TierKey

  selectedTierKey.value = tierKey

  const validCycles: BillingCycle[] = ['monthly', 'yearly']
  if (!cycleParam || !(validCycles as string[]).includes(cycleParam)) {
    cycleParam = 'monthly'
  }

  if (!isInitialized.value) {
    await until(isInitialized).toBe(true)
  }

  if (isActiveSubscription.value) {
    await accessBillingPortal(undefined, false)
  } else {
    await performSubscriptionCheckout(
      tierKey,
      cycleParam as BillingCycle,
      false
    )
  }
}, reportError)

onMounted(() => {
  void runRedirect()
})
</script>

<template>
  <div
    class="flex h-full w-full items-center justify-center bg-comfy-menu-secondary-bg"
  >
    <div class="flex flex-col items-center gap-4">
      <img
        src="/assets/images/comfy-logo-single.svg"
        :alt="t('g.comfyOrgLogoAlt')"
        class="h-16 w-16"
      />
      <p
        v-if="selectedTierKey"
        class="font-inter text-base font-normal leading-normal text-base-foreground"
      >
        {{
          t('subscription.subscribeTo', {
            plan: tierDisplayName
          })
        }}
      </p>
      <ProgressSpinner
        v-if="selectedTierKey"
        class="h-8 w-8"
        stroke-width="4"
      />
      <Button
        v-if="selectedTierKey"
        as="a"
        href="/"
        link
        :label="t('cloudOnboarding.skipToCloudApp')"
      />
    </div>
  </div>
</template>
