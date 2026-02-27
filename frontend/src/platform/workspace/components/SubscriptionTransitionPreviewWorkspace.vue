<template>
  <h2 class="text-xl lg:text-2xl text-muted-foreground m-0 text-center mb-8">
    {{ $t('subscription.preview.confirmPlanChange') }}
  </h2>
  <div
    class="flex flex-col justify-between items-stretch mx-auto text-sm h-full"
  >
    <div>
      <!-- Plan Comparison Header -->
      <div class="flex items-center gap-4">
        <!-- Current Plan -->
        <div class="flex flex-col gap-1 w-[250px]">
          <span class="text-base-foreground text-sm">
            {{ currentTierName }}
          </span>
          <div class="flex items-baseline gap-1">
            <span class="text-2xl font-semibold text-base-foreground">
              ${{ currentDisplayPrice }}
            </span>
            <span class="text-sm text-base-foreground">
              {{ $t('subscription.usdPerMonthPerMember') }}
            </span>
          </div>
          <div class="flex items-center gap-1 text-muted-foreground text-sm">
            <i class="icon-[lucide--component] text-amber-400 text-xs" />
            <span
              >{{ currentDisplayCredits }}
              {{ $t('subscription.perMonth') }}</span
            >
          </div>
          <span class="text-muted-foreground text-sm inline">
            {{
              $t('subscription.preview.ends', { date: currentPeriodEndDate })
            }}
          </span>
        </div>

        <!-- Arrow -->
        <i class="pi pi-arrow-right text-muted-foreground w-8 h-8" />

        <!-- New Plan -->
        <div class="flex flex-col gap-1">
          <span class="text-base-foreground text-sm font-semibold">
            {{ newTierName }}
          </span>
          <div class="flex items-baseline gap-1">
            <span class="text-2xl font-semibold text-base-foreground">
              ${{ newDisplayPrice }}
            </span>
            <span class="text-sm text-base-foreground">
              {{ $t('subscription.usdPerMonthPerMember') }}
            </span>
          </div>
          <div class="flex items-center gap-1 text-muted-foreground text-sm">
            <i class="icon-[lucide--component] text-amber-400 text-xs" />
            <span
              >{{ newDisplayCredits }} {{ $t('subscription.perMonth') }}</span
            >
          </div>
          <span class="text-muted-foreground text-sm">
            {{ $t('subscription.preview.starting', { date: effectiveDate }) }}
          </span>
        </div>
      </div>

      <!-- Credits Section -->
      <div class="flex flex-col gap-3 pt-12 pb-6">
        <div class="flex items-center justify-between">
          <span class="text-base-foreground">
            {{ $t('subscription.preview.eachMonthCreditsRefill') }}
          </span>
          <div class="flex items-center gap-1">
            <i class="icon-[lucide--component] text-amber-400 text-sm" />
            <span class="font-bold text-base-foreground">
              {{ newDisplayCredits }}
            </span>
          </div>
        </div>
      </div>

      <!-- Proration Section -->
      <div
        v-if="showProration"
        class="flex flex-col gap-2 border-t border-border-subtle pt-6 pb-6"
      >
        <div
          v-if="proratedRefundCents > 0"
          class="flex items-center justify-between"
        >
          <span class="text-muted-foreground">
            {{
              $t('subscription.preview.proratedRefund', {
                plan: currentTierName
              })
            }}
          </span>
          <span class="text-muted-foreground">-${{ proratedRefund }}</span>
        </div>
        <div
          v-if="proratedChargeCents > 0"
          class="flex items-center justify-between"
        >
          <span class="text-muted-foreground">
            {{
              $t('subscription.preview.proratedCharge', { plan: newTierName })
            }}
          </span>
          <span class="text-muted-foreground">${{ proratedCharge }}</span>
        </div>
      </div>

      <!-- Total Due Section -->
      <div class="flex flex-col gap-2 border-t border-border-subtle pt-6">
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
      <Button
        variant="secondary"
        size="lg"
        class="w-full rounded-lg"
        :loading="isLoading"
        @click="$emit('confirm')"
      >
        {{ $t('subscription.preview.confirm') }}
      </Button>

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
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { getTierCredits } from '@/platform/cloud/subscription/constants/tierPricing'
import type { PreviewSubscribeResponse } from '@/platform/workspace/api/workspaceApi'

interface Props {
  previewData: PreviewSubscribeResponse
  isLoading?: boolean
}

const { previewData, isLoading = false } = defineProps<Props>()

defineEmits<{
  confirm: []
  back: []
}>()

const { t, n } = useI18n()

function formatTierName(tier: string): string {
  return t(`subscription.tiers.${tier.toLowerCase()}.name`)
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  })
}

const currentTierName = computed(() =>
  previewData.current_plan ? formatTierName(previewData.current_plan.tier) : ''
)

const newTierName = computed(() => formatTierName(previewData.new_plan.tier))

const currentDisplayPrice = computed(() =>
  previewData.current_plan
    ? (previewData.current_plan.price_cents / 100).toFixed(0)
    : '0'
)

const newDisplayPrice = computed(() =>
  (previewData.new_plan.price_cents / 100).toFixed(0)
)

const currentDisplayCredits = computed(() => {
  if (!previewData.current_plan) return n(0)
  const tierKey = previewData.current_plan.tier.toLowerCase() as
    | 'standard'
    | 'creator'
    | 'pro'
  return n(getTierCredits(tierKey) ?? 0)
})

const newDisplayCredits = computed(() => {
  const tierKey = previewData.new_plan.tier.toLowerCase() as
    | 'standard'
    | 'creator'
    | 'pro'
  return n(getTierCredits(tierKey) ?? 0)
})

const currentPeriodEndDate = computed(() =>
  previewData.current_plan?.period_end
    ? formatDate(previewData.current_plan.period_end)
    : ''
)

const effectiveDate = computed(() => formatDate(previewData.effective_at))

const showProration = computed(() => previewData.is_immediate)

const proratedRefundCents = computed(() => {
  if (!previewData.current_plan || !previewData.is_immediate) return 0
  const chargeToday = previewData.cost_today_cents
  const newPlanCost = previewData.new_plan.price_cents
  if (chargeToday < newPlanCost) {
    return newPlanCost - chargeToday
  }
  return 0
})

const proratedRefund = computed(() =>
  (proratedRefundCents.value / 100).toFixed(2)
)

const proratedChargeCents = computed(() => {
  if (!previewData.is_immediate) return 0
  return previewData.cost_today_cents
})

const proratedCharge = computed(() =>
  (proratedChargeCents.value / 100).toFixed(2)
)

const totalDueToday = computed(() =>
  (previewData.cost_today_cents / 100).toFixed(2)
)

const nextPaymentDate = computed(() =>
  previewData.new_plan.period_end
    ? formatDate(previewData.new_plan.period_end)
    : formatDate(previewData.effective_at)
)
</script>
