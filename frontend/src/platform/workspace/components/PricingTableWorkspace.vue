<template>
  <div class="flex flex-col gap-8">
    <h2 class="text-xl lg:text-2xl text-muted-foreground m-0 text-center">
      {{ t('subscription.chooseBestPlanWorkspace') }}
    </h2>

    <div class="flex justify-center">
      <SelectButton
        v-model="currentBillingCycle"
        :options="billingCycleOptions"
        option-label="label"
        option-value="value"
        :allow-empty="false"
        unstyled
        :pt="{
          root: {
            class: 'flex gap-1 bg-secondary-background rounded-lg p-1.5'
          },
          pcToggleButton: {
            root: ({ context }: ToggleButtonPassThroughMethodOptions) => ({
              class: [
                'w-36  h-8 rounded-md transition-colors cursor-pointer border-none outline-none ring-0 text-sm font-medium flex items-center justify-center',
                context.active
                  ? 'bg-base-foreground text-base-background'
                  : 'bg-transparent text-muted-foreground hover:bg-secondary-background-hover'
              ]
            }),
            label: { class: 'flex items-center gap-2 ' }
          }
        }"
      >
        <template #option="{ option }">
          <div class="flex items-center gap-2">
            <span>{{ option.label }}</span>
            <div
              v-if="option.value === 'yearly'"
              class="bg-primary-background text-white text-[11px] px-1 py-0.5 rounded-full flex items-center font-bold"
            >
              -20%
            </div>
          </div>
        </template>
      </SelectButton>
    </div>
    <div class="flex flex-col xl:flex-row items-stretch gap-6">
      <div
        v-for="tier in tiers"
        :key="tier.id"
        :class="
          cn(
            'flex-1 flex flex-col rounded-2xl border border-border-default bg-base-background shadow-[0_0_12px_rgba(0,0,0,0.1)]',
            tier.isPopular ? 'border-muted-foreground' : ''
          )
        "
      >
        <div class="p-8 pb-0 flex flex-col gap-8">
          <div class="flex flex-row items-center gap-2 justify-between">
            <span
              class="font-inter text-base font-bold leading-normal text-base-foreground"
            >
              {{ tier.name }}
            </span>
            <div
              v-if="tier.isPopular"
              class="rounded-full bg-base-foreground px-1.5 text-[11px] font-bold uppercase text-base-background h-5 tracking-tight flex items-center"
            >
              {{ t('subscription.mostPopular') }}
            </div>
          </div>
          <div class="flex flex-col">
            <div class="flex flex-col gap-2">
              <div class="flex flex-row items-baseline gap-2">
                <span
                  class="font-inter text-[32px] font-semibold leading-normal text-base-foreground"
                >
                  <span
                    v-show="currentBillingCycle === 'yearly'"
                    class="line-through text-2xl text-muted-foreground"
                  >
                    ${{ getMonthlyPrice(tier) }}
                  </span>
                  ${{ getPrice(tier) }}
                </span>
                <span
                  class="font-inter text-sm leading-normal text-base-foreground"
                >
                  {{ t('subscription.usdPerMonthPerMember') }}
                </span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-sm text-muted-foreground">
                  {{
                    currentBillingCycle === 'yearly'
                      ? t('subscription.billedYearly', {
                          total: `$${getAnnualTotal(tier)}`
                        })
                      : t('subscription.billedMonthly')
                  }}
                </span>
              </div>
            </div>
          </div>

          <div class="flex flex-col gap-4 pb-0 flex-1">
            <div class="flex flex-row items-center justify-between">
              <span class="text-sm font-normal text-foreground">
                {{ t('subscription.monthlyCreditsPerMemberLabel') }}
              </span>
              <div class="flex flex-row items-center gap-1">
                <i class="icon-[lucide--component] text-amber-400 text-sm" />
                <span
                  class="font-inter text-sm font-bold leading-normal text-base-foreground"
                >
                  {{ n(getMonthlyCreditsPerMember(tier)) }}
                </span>
              </div>
            </div>

            <div class="flex flex-row items-center justify-between">
              <span class="text-sm font-normal text-foreground">
                {{ t('subscription.maxMembersLabel') }}
              </span>
              <span
                class="font-inter text-sm font-bold leading-normal text-base-foreground"
              >
                {{ getMaxMembers(tier) }}
              </span>
            </div>

            <div class="flex flex-row items-center justify-between">
              <span class="text-sm font-normal text-foreground">
                {{ t('subscription.maxDurationLabel') }}
              </span>
              <span
                class="font-inter text-sm font-bold leading-normal text-base-foreground"
              >
                {{ tier.maxDuration }}
              </span>
            </div>

            <div class="flex flex-row items-center justify-between">
              <span class="text-sm font-normal text-foreground">
                {{ t('subscription.gpuLabel') }}
              </span>
              <i class="pi pi-check text-xs text-success-foreground" />
            </div>

            <div class="flex flex-row items-center justify-between">
              <span class="text-sm font-normal text-foreground">
                {{ t('subscription.addCreditsLabel') }}
              </span>
              <i class="pi pi-check text-xs text-success-foreground" />
            </div>

            <div class="flex flex-row items-center justify-between">
              <span class="text-sm font-normal text-foreground">
                {{ t('subscription.customLoRAsLabel') }}
              </span>
              <i
                v-if="tier.customLoRAs"
                class="pi pi-check text-xs text-success-foreground"
              />
              <i v-else class="pi pi-times text-xs text-foreground" />
            </div>

            <div class="flex flex-col gap-2">
              <div class="flex flex-row items-start justify-between">
                <div class="flex flex-col gap-2">
                  <span
                    class="text-sm font-normal text-foreground leading-relaxed"
                  >
                    {{ t('subscription.videoEstimateLabel') }}
                  </span>
                  <div class="flex flex-row items-center gap-2 group pt-2">
                    <i
                      class="pi pi-question-circle text-xs text-muted-foreground group-hover:text-base-foreground"
                    />
                    <span
                      class="text-sm font-normal text-muted-foreground cursor-pointer group-hover:text-base-foreground"
                      @click="togglePopover"
                    >
                      {{ t('subscription.videoEstimateHelp') }}
                    </span>
                  </div>
                </div>
                <span
                  class="font-inter text-sm font-bold leading-normal text-base-foreground"
                >
                  ~{{ n(tier.pricing.videoEstimate) }}
                </span>
              </div>
            </div>
          </div>
        </div>
        <div class="flex flex-col p-8">
          <Button
            :variant="getButtonSeverity(tier)"
            :disabled="isButtonDisabled(tier)"
            :loading="loadingTier === tier.key"
            :class="
              cn(
                'h-10 w-full',
                getButtonTextClass(tier),
                tier.key === 'creator'
                  ? 'bg-base-foreground border-transparent hover:bg-inverted-background-hover'
                  : 'bg-secondary-background border-transparent hover:bg-secondary-background-hover focus:bg-secondary-background-selected'
              )
            "
            @click="() => handleSubscribe(tier.key)"
          >
            {{ getButtonLabel(tier) }}
          </Button>
        </div>
      </div>
    </div>

    <!-- Video Estimate Help Popover -->
    <Popover
      ref="popover"
      append-to="body"
      :auto-z-index="true"
      :base-z-index="1000"
      :dismissable="true"
      :close-on-escape="true"
      unstyled
      :pt="{
        root: {
          class:
            'rounded-lg border border-interface-stroke bg-interface-panel-surface shadow-lg p-4 max-w-xs'
        }
      }"
    >
      <div class="flex flex-col gap-2">
        <p class="text-sm text-base-foreground leading-normal">
          {{ t('subscription.videoEstimateExplanation') }}
        </p>
        <a
          href="https://cloud.comfy.org/?template=video_wan2_2_14B_i2v"
          target="_blank"
          rel="noopener noreferrer"
          class="text-sm text-azure-600 hover:text-azure-400 no-underline flex gap-1"
        >
          <span class="underline">
            {{ t('subscription.videoEstimateTryTemplate') }}
          </span>
          <span class="no-underline" v-html="'&rarr;'"></span>
        </a>
      </div>
    </Popover>
    <!-- Contact and Enterprise Links -->
    <div class="flex flex-col items-center gap-2">
      <p class="text-sm text-text-secondary m-0">
        {{ $t('subscription.haveQuestions') }}
      </p>
      <div class="flex items-center gap-1.5">
        <Button
          variant="muted-textonly"
          class="h-6 p-1 text-sm text-text-secondary hover:text-base-foreground"
          @click="handleContactUs"
        >
          {{ $t('subscription.contactUs') }}
          <i class="pi pi-comments" />
        </Button>
        <span class="text-sm text-text-secondary">{{ $t('g.or') }}</span>
        <Button
          variant="muted-textonly"
          class="h-6 p-1 text-sm text-text-secondary hover:text-base-foreground"
          @click="handleViewEnterprise"
        >
          {{ $t('subscription.viewEnterprise') }}
          <i class="pi pi-external-link" />
        </Button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { cn } from '@comfyorg/tailwind-utils'
import Popover from 'primevue/popover'
import SelectButton from 'primevue/selectbutton'
import type { ToggleButtonPassThroughMethodOptions } from 'primevue/togglebutton'
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import {
  TIER_PRICING,
  TIER_TO_KEY
} from '@/platform/cloud/subscription/constants/tierPricing'
import type {
  TierKey,
  TierPricing
} from '@/platform/cloud/subscription/constants/tierPricing'
import type { BillingCycle } from '@/platform/cloud/subscription/utils/subscriptionTierRank'
import type { Plan } from '@/platform/workspace/api/workspaceApi'
import type { components } from '@/types/comfyRegistryTypes'

type SubscriptionTier = components['schemas']['SubscriptionTier']
type CheckoutTierKey = Exclude<TierKey, 'free' | 'founder'>

interface Props {
  isLoading?: boolean
  loadingTier?: CheckoutTierKey | null
}

const { isLoading = false, loadingTier = null } = defineProps<Props>()

const emit = defineEmits<{
  subscribe: [payload: { tierKey: CheckoutTierKey; billingCycle: BillingCycle }]
  resubscribe: []
}>()

interface BillingCycleOption {
  label: string
  value: BillingCycle
}

interface PricingTierConfig {
  id: SubscriptionTier
  key: CheckoutTierKey
  name: string
  pricing: TierPricing
  maxDuration: string
  customLoRAs: boolean
  maxMembers: number
  isPopular?: boolean
}

const { t, n } = useI18n()

const billingCycleOptions: BillingCycleOption[] = [
  { label: t('subscription.yearly'), value: 'yearly' },
  { label: t('subscription.monthly'), value: 'monthly' }
]

const tiers: PricingTierConfig[] = [
  {
    id: 'STANDARD',
    key: 'standard',
    name: t('subscription.tiers.standard.name'),
    pricing: TIER_PRICING.standard,
    maxDuration: t('subscription.maxDuration.standard'),
    customLoRAs: false,
    maxMembers: 1,
    isPopular: false
  },
  {
    id: 'CREATOR',
    key: 'creator',
    name: t('subscription.tiers.creator.name'),
    pricing: TIER_PRICING.creator,
    maxDuration: t('subscription.maxDuration.creator'),
    customLoRAs: true,
    maxMembers: 5,
    isPopular: true
  },
  {
    id: 'PRO',
    key: 'pro',
    name: t('subscription.tiers.pro.name'),
    pricing: TIER_PRICING.pro,
    maxDuration: t('subscription.maxDuration.pro'),
    customLoRAs: true,
    maxMembers: 20,
    isPopular: false
  }
]
const {
  plans: apiPlans,
  currentPlanSlug,
  fetchPlans,
  subscription,
  getMaxSeats
} = useBillingContext()

const isCancelled = computed(() => subscription.value?.isCancelled ?? false)

const popover = ref()
const currentBillingCycle = ref<BillingCycle>('yearly')

onMounted(() => {
  void fetchPlans()
})

function getApiPlanForTier(
  tierKey: CheckoutTierKey,
  duration: BillingCycle
): Plan | undefined {
  const apiDuration = duration === 'yearly' ? 'ANNUAL' : 'MONTHLY'
  const apiTier = tierKey.toUpperCase() as Plan['tier']
  return apiPlans.value.find(
    (p) => p.tier === apiTier && p.duration === apiDuration
  )
}

function getPriceFromApi(tier: PricingTierConfig): number | null {
  const plan = getApiPlanForTier(tier.key, currentBillingCycle.value)
  if (!plan) return null
  const price = plan.price_cents / 100
  return currentBillingCycle.value === 'yearly' ? price / 12 : price
}

const currentTierKey = computed<TierKey | null>(() =>
  subscription.value?.tier ? TIER_TO_KEY[subscription.value.tier] : null
)

const isYearlySubscription = computed(
  () => subscription.value?.duration === 'ANNUAL'
)

const isCurrentPlan = (tierKey: CheckoutTierKey): boolean => {
  // Use API current_plan_slug if available
  if (currentPlanSlug.value) {
    const plan = getApiPlanForTier(tierKey, currentBillingCycle.value)
    return plan?.slug === currentPlanSlug.value
  }

  // Fallback to tier-based detection
  if (!currentTierKey.value) return false

  const selectedIsYearly = currentBillingCycle.value === 'yearly'

  return (
    currentTierKey.value === tierKey &&
    isYearlySubscription.value === selectedIsYearly
  )
}

const togglePopover = (event: Event) => {
  popover.value.toggle(event)
}

const getButtonLabel = (tier: PricingTierConfig): string => {
  const planName =
    currentBillingCycle.value === 'yearly'
      ? t('subscription.tierNameYearly', { name: tier.name })
      : tier.name

  if (isCurrentPlan(tier.key)) {
    return isCancelled.value
      ? t('subscription.resubscribeTo', { plan: planName })
      : t('subscription.currentPlan')
  }

  return currentTierKey.value
    ? t('subscription.changeTo', { plan: planName })
    : t('subscription.subscribeTo', { plan: planName })
}

const getButtonSeverity = (
  tier: PricingTierConfig
): 'primary' | 'secondary' => {
  if (isCurrentPlan(tier.key)) {
    return isCancelled.value ? 'primary' : 'secondary'
  }
  if (tier.key === 'creator') return 'primary'
  return 'secondary'
}

const isButtonDisabled = (tier: PricingTierConfig): boolean => {
  if (isLoading) return true
  if (isCurrentPlan(tier.key)) {
    // Allow clicking current plan button when cancelled (for resubscribe)
    return !isCancelled.value
  }
  return false
}

const getButtonTextClass = (tier: PricingTierConfig): string =>
  tier.key === 'creator'
    ? 'font-inter text-sm font-bold leading-normal text-base-background'
    : 'font-inter text-sm font-bold leading-normal text-primary-foreground'

const getPrice = (tier: PricingTierConfig): number =>
  getPriceFromApi(tier) ?? tier.pricing[currentBillingCycle.value]

const getMonthlyPrice = (tier: PricingTierConfig): number => {
  const plan = getApiPlanForTier(tier.key, 'monthly')
  return plan ? plan.price_cents / 100 : tier.pricing.monthly
}

const getAnnualTotal = (tier: PricingTierConfig): number => {
  const plan = getApiPlanForTier(tier.key, 'yearly')
  return plan ? plan.price_cents / 100 : tier.pricing.yearly * 12
}

const getMaxMembers = (tier: PricingTierConfig): number => getMaxSeats(tier.key)

const getMonthlyCreditsPerMember = (tier: PricingTierConfig): number =>
  tier.pricing.credits

function handleSubscribe(tierKey: CheckoutTierKey) {
  if (isLoading) return

  // Handle resubscribe for cancelled subscription on current plan
  if (isCurrentPlan(tierKey)) {
    if (isCancelled.value) {
      emit('resubscribe')
    }
    return
  }

  emit('subscribe', {
    tierKey,
    billingCycle: currentBillingCycle.value
  })
}

function handleContactUs() {
  window.open('https://www.comfy.org/discord', '_blank')
}

function handleViewEnterprise() {
  window.open('https://www.comfy.org/enterprise', '_blank')
}
</script>
