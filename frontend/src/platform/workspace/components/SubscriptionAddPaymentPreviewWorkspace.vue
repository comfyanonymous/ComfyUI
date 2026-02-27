<template>
  <h2 class="text-xl lg:text-2xl text-muted-foreground m-0 text-center mb-8">
    {{ $t('subscription.preview.confirmPayment') }}
  </h2>
  <div
    class="flex flex-col justify-between items-stretch max-w-[400px] mx-auto text-sm h-full"
  >
    <div class="">
      <!-- Plan Header -->
      <div class="flex flex-col gap-2">
        <span class="text-base-foreground text-sm">
          {{ tierName }}
        </span>
        <div class="flex items-baseline gap-2">
          <span class="text-4xl font-semibold text-base-foreground">
            ${{ displayPrice }}
          </span>
          <span class="text-xl text-base-foreground">
            {{ $t('subscription.usdPerMonthPerMember') }}
          </span>
        </div>
        <span class="text-muted-foreground">
          {{ $t('subscription.preview.startingToday') }}
        </span>
      </div>

      <!-- Credits Section -->
      <div class="flex flex-col gap-3 pt-16 pb-8">
        <div class="flex items-center justify-between">
          <span class="text-base-foreground">
            {{ $t('subscription.preview.eachMonthCreditsRefill') }}
          </span>
          <div class="flex items-center gap-1">
            <i class="icon-[lucide--component] text-amber-400 text-sm" />
            <span class="font-bold text-base-foreground">
              {{ displayCredits }}
            </span>
            <span class="text-base-foreground">
              {{ $t('subscription.preview.perMember') }}
            </span>
          </div>
        </div>

        <!-- Expandable Features -->
        <button
          class="flex items-center justify-end gap-1 text-sm text-muted-foreground hover:text-base-foreground cursor-pointer bg-transparent border-none p-0"
          @click="isFeaturesCollapsed = !isFeaturesCollapsed"
        >
          <span>
            {{
              isFeaturesCollapsed
                ? $t('subscription.preview.showMoreFeatures')
                : $t('subscription.preview.hideFeatures')
            }}
          </span>
          <i
            :class="
              cn(
                'pi text-xs',
                isFeaturesCollapsed ? 'pi-chevron-down' : 'pi-chevron-up'
              )
            "
          />
        </button>
        <div v-show="!isFeaturesCollapsed" class="flex flex-col gap-2 pt-2">
          <div class="flex items-center justify-between">
            <span class="text-sm text-base-foreground">
              {{ $t('subscription.maxDurationLabel') }}
            </span>
            <span class="text-sm font-bold text-base-foreground">
              {{ maxDuration }}
            </span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-base-foreground">
              {{ $t('subscription.gpuLabel') }}
            </span>
            <i class="pi pi-check text-xs text-success-foreground" />
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-base-foreground">
              {{ $t('subscription.addCreditsLabel') }}
            </span>
            <i class="pi pi-check text-xs text-success-foreground" />
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-base-foreground">
              {{ $t('subscription.customLoRAsLabel') }}
            </span>
            <i
              v-if="hasCustomLoRAs"
              class="pi pi-check text-xs text-success-foreground"
            />
            <i v-else class="pi pi-times text-xs text-muted-foreground" />
          </div>
        </div>
      </div>

      <!-- Total Due Section -->
      <div class="flex flex-col gap-2 border-t border-border-subtle pt-8">
        <div class="flex text-base items-center justify-between">
          <span class="text-base-foreground">
            {{ $t('subscription.preview.totalDueToday') }}
          </span>
          <span class="font-bold text-base-foreground">
            ${{ totalDueToday }}
          </span>
        </div>
        <span class="text-muted-foreground text-sm">
          {{
            $t('subscription.preview.nextPaymentDue', {
              date: nextPaymentDate
            })
          }}
        </span>
      </div>
    </div>
    <!-- Footer -->
    <div class="flex flex-col gap-2 pt-8">
      <!-- Terms Agreement -->
      <p class="text-xs text-muted-foreground text-center">
        <i18n-t keypath="subscription.preview.termsAgreement" tag="span">
          <template #terms>
            <a
              href="https://www.comfy.org/terms"
              target="_blank"
              rel="noopener noreferrer"
              class="underline hover:text-base-foreground"
            >
              {{ $t('subscription.preview.terms') }}
            </a>
          </template>
          <template #privacy>
            <a
              href="https://www.comfy.org/privacy"
              target="_blank"
              rel="noopener noreferrer"
              class="underline hover:text-base-foreground"
            >
              {{ $t('subscription.preview.privacyPolicy') }}
            </a>
          </template>
        </i18n-t>
      </p>

      <!-- Add Credit Card Button -->
      <Button
        variant="secondary"
        size="lg"
        class="w-full rounded-lg"
        :loading="isLoading"
        @click="$emit('addCreditCard')"
      >
        {{ $t('subscription.preview.addCreditCard') }}
      </Button>

      <!-- Back Link -->
      <Button
        variant="textonly"
        class="text-muted-foreground hover:text-base-foreground hover:bg-none text-center cursor-pointer transition-colors text-xs"
        @click="$emit('back')"
      >
        {{ $t('subscription.preview.backToAllPlans') }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import {
  getTierCredits,
  getTierFeatures,
  getTierPrice
} from '@/platform/cloud/subscription/constants/tierPricing'
import type { TierKey } from '@/platform/cloud/subscription/constants/tierPricing'
import type { BillingCycle } from '@/platform/cloud/subscription/utils/subscriptionTierRank'
import type { PreviewSubscribeResponse } from '@/platform/workspace/api/workspaceApi'
import { cn } from '@/utils/tailwindUtil'

interface Props {
  tierKey: Exclude<TierKey, 'free' | 'founder'>
  billingCycle?: BillingCycle
  isLoading?: boolean
  previewData?: PreviewSubscribeResponse | null
}

const {
  tierKey,
  billingCycle = 'monthly',
  isLoading = false,
  previewData = null
} = defineProps<Props>()

defineEmits<{
  addCreditCard: []
  back: []
}>()

const { t, n } = useI18n()

const isFeaturesCollapsed = ref(true)

const tierName = computed(() => t(`subscription.tiers.${tierKey}.name`))

const displayPrice = computed(() => {
  if (previewData?.new_plan) {
    return (previewData.new_plan.price_cents / 100).toFixed(0)
  }
  return getTierPrice(tierKey, billingCycle === 'yearly')
})

const displayCredits = computed(() => n(getTierCredits(tierKey) ?? 0))

const hasCustomLoRAs = computed(() => getTierFeatures(tierKey).customLoRAs)
const maxDuration = computed(() => t(`subscription.maxDuration.${tierKey}`))

const totalDueToday = computed(() => {
  if (previewData) {
    return (previewData.cost_today_cents / 100).toFixed(2)
  }
  const priceValue = getTierPrice(tierKey, billingCycle === 'yearly')
  if (billingCycle === 'yearly') {
    return (priceValue * 12).toFixed(2)
  }
  return priceValue.toFixed(2)
})

const nextPaymentDate = computed(() => {
  if (previewData?.new_plan?.period_end) {
    return new Date(previewData.new_plan.period_end).toLocaleDateString(
      'en-US',
      {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      }
    )
  }
  const date = new Date()
  if (billingCycle === 'yearly') {
    date.setFullYear(date.getFullYear() + 1)
  } else {
    date.setMonth(date.getMonth() + 1)
  }
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  })
})
</script>
