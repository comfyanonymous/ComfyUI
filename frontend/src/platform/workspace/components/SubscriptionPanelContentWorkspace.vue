<template>
  <div class="grow overflow-auto pt-6">
    <!-- Loading state while subscription is being set up -->
    <div
      v-if="isSettingUp"
      class="rounded-2xl border border-interface-stroke p-6"
    >
      <div class="flex items-center gap-2 text-muted-foreground py-4">
        <i class="pi pi-spin pi-spinner" />
        <span>{{ $t('billingOperation.subscriptionProcessing') }}</span>
      </div>
    </div>

    <template v-else>
      <!-- Cancelled subscription info card -->
      <div
        v-if="isCancelled"
        class="mb-6 flex gap-1 rounded-2xl border border-warning-background bg-warning-background/20 p-4"
      >
        <div
          class="flex size-8 shrink-0 items-center justify-center rounded-full text-warning-background"
        >
          <i class="pi pi-info-circle" />
        </div>
        <div class="flex flex-col gap-2">
          <h2 class="text-sm font-bold text-text-primary m-0 pt-1.5">
            {{ $t('subscription.canceledCard.title') }}
          </h2>
          <p class="text-sm text-text-secondary m-0">
            {{
              $t('subscription.canceledCard.description', {
                date: formattedEndDate
              })
            }}
          </p>
        </div>
      </div>

      <div class="rounded-2xl border border-interface-stroke p-6">
        <div>
          <div
            class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between md:gap-2"
          >
            <!-- OWNER Unsubscribed State -->
            <template v-if="showSubscribePrompt">
              <div class="flex flex-col gap-2">
                <div class="text-sm font-bold text-text-primary">
                  {{ $t('subscription.workspaceNotSubscribed') }}
                </div>
                <div class="text-sm text-text-secondary">
                  {{ $t('subscription.subscriptionRequiredMessage') }}
                </div>
              </div>
              <Button
                variant="primary"
                size="lg"
                class="ml-auto rounded-lg px-4 py-2 text-sm font-normal"
                @click="handleSubscribeWorkspace"
              >
                {{ $t('subscription.subscribeNow') }}
              </Button>
            </template>

            <!-- MEMBER View - read-only, workspace not subscribed -->
            <template v-else-if="isMemberView">
              <div class="flex flex-col gap-2">
                <div class="text-sm font-bold text-text-primary">
                  {{ $t('subscription.workspaceNotSubscribed') }}
                </div>
                <div class="text-sm text-text-secondary">
                  {{ $t('subscription.contactOwnerToSubscribe') }}
                </div>
              </div>
            </template>

            <!-- Normal Subscribed State (Owner with subscription, or member viewing subscribed workspace) -->
            <template v-else>
              <div class="flex flex-col gap-2">
                <div class="flex items-center gap-2">
                  <span class="text-sm font-bold text-text-primary">
                    {{ subscriptionTierName }}
                  </span>
                  <StatusBadge
                    v-if="isCancelled"
                    :label="$t('subscription.canceled')"
                    severity="warn"
                  />
                </div>
                <div class="flex items-baseline gap-1 font-inter font-semibold">
                  <span class="text-2xl">${{ tierPrice }}</span>
                  <span class="text-base">
                    {{
                      isInPersonalWorkspace
                        ? $t('subscription.usdPerMonth')
                        : $t('subscription.usdPerMonthPerMember')
                    }}
                  </span>
                </div>
                <div
                  v-if="isActiveSubscription"
                  :class="
                    cn(
                      'text-sm',
                      isCancelled
                        ? 'text-warning-background'
                        : 'text-text-secondary'
                    )
                  "
                >
                  <template v-if="isCancelled">
                    {{
                      $t('subscription.expiresDate', {
                        date: formattedEndDate
                      })
                    }}
                  </template>
                  <template v-else>
                    {{
                      $t('subscription.renewsDate', {
                        date: formattedRenewalDate
                      })
                    }}
                  </template>
                </div>
              </div>

              <div
                v-if="isActiveSubscription && permissions.canManageSubscription"
                class="flex flex-wrap gap-2 md:ml-auto"
              >
                <!-- Cancelled state: show only Resubscribe button -->
                <template v-if="isCancelled">
                  <Button
                    size="lg"
                    variant="primary"
                    class="rounded-lg px-4 text-sm font-normal"
                    :loading="isResubscribing"
                    @click="handleResubscribe"
                  >
                    {{ $t('subscription.resubscribe') }}
                  </Button>
                </template>

                <!-- Active state: show Manage Payment, Upgrade, and menu -->
                <template v-else>
                  <Button
                    v-if="!isFreeTierPlan"
                    size="lg"
                    variant="secondary"
                    class="rounded-lg px-4 text-sm font-normal text-text-primary bg-interface-menu-component-surface-selected"
                    @click="manageSubscription"
                  >
                    {{ $t('subscription.managePayment') }}
                  </Button>
                  <Button
                    size="lg"
                    variant="primary"
                    class="rounded-lg px-4 text-sm font-normal text-text-primary"
                    @click="handleUpgrade"
                  >
                    {{ $t('subscription.upgradePlan') }}
                  </Button>
                  <Button
                    v-if="!isFreeTierPlan"
                    v-tooltip="{ value: $t('g.moreOptions'), showDelay: 300 }"
                    variant="secondary"
                    size="lg"
                    :aria-label="$t('g.moreOptions')"
                    @click="planMenu?.toggle($event)"
                  >
                    <i class="pi pi-ellipsis-h" />
                  </Button>
                  <Menu ref="planMenu" :model="planMenuItems" :popup="true" />
                </template>
              </div>
            </template>
          </div>
        </div>

        <div class="flex flex-col lg:flex-row lg:items-stretch gap-6 pt-6">
          <div class="flex flex-col">
            <div class="flex flex-col gap-3 h-full">
              <div
                class="relative flex flex-col gap-6 rounded-2xl p-5 bg-secondary-background justify-between h-full"
              >
                <Button
                  variant="muted-textonly"
                  size="icon-sm"
                  class="absolute top-4 right-4"
                  :loading="isLoadingBalance"
                  @click="handleRefresh"
                >
                  <i class="pi pi-sync text-text-secondary text-sm" />
                </Button>

                <div class="flex flex-col gap-2">
                  <div class="text-sm text-muted">
                    {{ $t('subscription.totalCredits') }}
                  </div>
                  <Skeleton
                    v-if="isLoadingBalance"
                    width="8rem"
                    height="2rem"
                  />
                  <div v-else class="text-2xl font-bold">
                    {{ showZeroState ? '0' : totalCredits }}
                  </div>
                </div>

                <!-- Credit Breakdown -->
                <table class="text-sm text-muted">
                  <tbody>
                    <tr>
                      <td class="pr-4 font-bold text-left align-middle">
                        <Skeleton
                          v-if="isLoadingBalance"
                          width="5rem"
                          height="1rem"
                        />
                        <span v-else>{{
                          showZeroState ? '0 / 0' : includedCreditsDisplay
                        }}</span>
                      </td>
                      <td class="align-middle" :title="creditsRemainingLabel">
                        {{ creditsRemainingLabel }}
                      </td>
                    </tr>
                    <tr>
                      <td class="pr-4 font-bold text-left align-middle">
                        <Skeleton
                          v-if="isLoadingBalance"
                          width="3rem"
                          height="1rem"
                        />
                        <span v-else>{{
                          showZeroState ? '0' : prepaidCredits
                        }}</span>
                      </td>
                      <td
                        class="align-middle"
                        :title="$t('subscription.creditsYouveAdded')"
                      >
                        {{ $t('subscription.creditsYouveAdded') }}
                      </td>
                    </tr>
                  </tbody>
                </table>

                <div
                  v-if="
                    isActiveSubscription &&
                    !showZeroState &&
                    permissions.canTopUp
                  "
                  class="flex items-center justify-between"
                >
                  <Button
                    variant="secondary"
                    class="p-2 min-h-8 rounded-lg text-sm font-normal text-text-primary bg-interface-menu-component-surface-selected"
                    @click="handleAddApiCredits"
                  >
                    {{ $t('subscription.addCredits') }}
                  </Button>
                </div>
              </div>
            </div>
          </div>

          <div v-if="isActiveSubscription" class="flex flex-col gap-2">
            <div class="text-sm text-text-primary">
              {{ $t('subscription.yourPlanIncludes') }}
            </div>

            <div class="flex flex-col gap-0">
              <div
                v-for="benefit in tierBenefits"
                :key="benefit.key"
                class="flex items-center gap-2 py-2"
              >
                <i
                  v-if="benefit.type === 'feature'"
                  class="pi pi-check text-xs text-text-primary"
                />
                <i
                  v-else-if="benefit.type === 'icon' && benefit.icon"
                  :class="[benefit.icon, 'text-xs text-text-primary']"
                />
                <span
                  v-else-if="benefit.type === 'metric' && benefit.value"
                  class="text-sm font-normal whitespace-nowrap text-text-primary"
                >
                  {{ benefit.value }}
                </span>
                <span class="text-sm text-muted">
                  {{ benefit.label }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Members invoice card -->
      <div
        v-if="
          isActiveSubscription &&
          !isInPersonalWorkspace &&
          permissions.canManageSubscription
        "
        class="mt-6 flex gap-1 rounded-2xl border border-interface-stroke p-6 justify-between items-center text-sm"
      >
        <div class="flex flex-col gap-2">
          <h4 class="text-sm text-text-primary m-0">
            {{ $t('subscription.nextMonthInvoice') }}
          </h4>
          <span
            class="text-muted-foreground underline cursor-pointer"
            @click="manageSubscription"
          >
            {{ $t('subscription.invoiceHistory') }}
          </span>
        </div>
        <div class="flex flex-col gap-2 items-end">
          <h4 class="m-0 font-bold">${{ nextMonthInvoice }}</h4>
          <h5 class="m-0 text-muted-foreground">
            {{ $t('subscription.memberCount', memberCount) }}
          </h5>
        </div>
      </div>

      <!-- View More Details - Outside main content -->
      <div
        v-if="permissions.canManageSubscription"
        class="flex items-center gap-2 py-6"
      >
        <i class="pi pi-external-link text-muted"></i>
        <a
          href="https://www.comfy.org/cloud/pricing"
          target="_blank"
          rel="noopener noreferrer"
          class="text-sm underline hover:opacity-80 text-muted"
        >
          {{ $t('subscription.viewMoreDetailsPlans') }}
        </a>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import Menu from 'primevue/menu'
import Skeleton from 'primevue/skeleton'
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { useToast } from 'primevue/usetoast'

import StatusBadge from '@/components/common/StatusBadge.vue'
import Button from '@/components/ui/button/Button.vue'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { useBillingOperationStore } from '@/platform/workspace/stores/billingOperationStore'
import { useSubscriptionActions } from '@/platform/cloud/subscription/composables/useSubscriptionActions'
import { useSubscriptionCredits } from '@/platform/cloud/subscription/composables/useSubscriptionCredits'
import { workspaceApi } from '@/platform/workspace/api/workspaceApi'
import { useDialogService } from '@/services/dialogService'
import {
  DEFAULT_TIER_KEY,
  TIER_TO_KEY,
  getTierCredits,
  getTierPrice
} from '@/platform/cloud/subscription/constants/tierPricing'
import { useSubscriptionDialog } from '@/platform/cloud/subscription/composables/useSubscriptionDialog'
import type { TierBenefit } from '@/platform/cloud/subscription/utils/tierBenefits'
import { getCommonTierBenefits } from '@/platform/cloud/subscription/utils/tierBenefits'
import { useWorkspaceUI } from '@/platform/workspace/composables/useWorkspaceUI'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'
import { cn } from '@/utils/tailwindUtil'

const workspaceStore = useTeamWorkspaceStore()
const { isWorkspaceSubscribed, isInPersonalWorkspace, members } =
  storeToRefs(workspaceStore)
const { permissions } = useWorkspaceUI()
const { t, n } = useI18n()
const toast = useToast()

const billingOperationStore = useBillingOperationStore()
const isSettingUp = computed(() => billingOperationStore.isSettingUp)

const {
  isActiveSubscription,
  isFreeTier: isFreeTierPlan,
  subscription,
  showSubscriptionDialog,
  manageSubscription,
  fetchStatus,
  fetchBalance,
  getMaxSeats
} = useBillingContext()

const { showCancelSubscriptionDialog } = useDialogService()
const { showPricingTable } = useSubscriptionDialog()

const isResubscribing = ref(false)

async function handleResubscribe() {
  isResubscribing.value = true
  try {
    await workspaceApi.resubscribe()
    toast.add({
      severity: 'success',
      summary: t('subscription.resubscribeSuccess'),
      life: 5000
    })
    await Promise.all([fetchStatus(), fetchBalance()])
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'Failed to resubscribe'
    toast.add({
      severity: 'error',
      summary: t('g.error'),
      detail: message,
      life: 5000
    })
  } finally {
    isResubscribing.value = false
  }
}

// Only show cancelled state for team workspaces (workspace billing)
// Personal workspaces use legacy billing which has different cancellation semantics
const isCancelled = computed(
  () =>
    !isInPersonalWorkspace.value && (subscription.value?.isCancelled ?? false)
)

// Show subscribe prompt to owners without active subscription
// Don't show if subscription is cancelled (still active until end date)
const showSubscribePrompt = computed(() => {
  if (!permissions.value.canManageSubscription) return false
  if (isCancelled.value) return false
  if (isInPersonalWorkspace.value) return !isActiveSubscription.value
  return !isWorkspaceSubscribed.value
})

// MEMBER view without subscription - members can't manage subscription
const isMemberView = computed(
  () =>
    !permissions.value.canManageSubscription &&
    !isActiveSubscription.value &&
    !isWorkspaceSubscribed.value
)

// Show zero state for credits (no real billing data yet)
const showZeroState = computed(
  () => showSubscribePrompt.value || isMemberView.value
)

// Subscribe workspace - opens the subscription dialog (personal or workspace variant)
function handleSubscribeWorkspace() {
  showSubscriptionDialog()
}

function handleUpgrade() {
  isFreeTierPlan.value ? showPricingTable() : showSubscriptionDialog()
}
const subscriptionTier = computed(() => subscription.value?.tier ?? null)
const isYearlySubscription = computed(
  () => subscription.value?.duration === 'ANNUAL'
)

const formattedRenewalDate = computed(() => {
  if (!subscription.value?.renewalDate) return ''
  const renewalDate = new Date(subscription.value.renewalDate)
  return renewalDate.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  })
})

const formattedEndDate = computed(() => {
  if (!subscription.value?.endDate) return ''
  const endDate = new Date(subscription.value.endDate)
  return endDate.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  })
})

const subscriptionTierName = computed(() => {
  const tier = subscriptionTier.value
  if (!tier) return ''
  const key = TIER_TO_KEY[tier] ?? 'standard'
  const baseName = t(`subscription.tiers.${key}.name`)
  return isYearlySubscription.value
    ? t('subscription.tierNameYearly', { name: baseName })
    : baseName
})

const planMenu = ref<InstanceType<typeof Menu> | null>(null)

const planMenuItems = computed(() => [
  {
    label: t('subscription.cancelSubscription'),
    icon: 'pi pi-times',
    command: () => {
      showCancelSubscriptionDialog(subscription.value?.endDate ?? undefined)
    }
  }
])

const tierKey = computed(() => {
  const tier = subscriptionTier.value
  if (!tier) return DEFAULT_TIER_KEY
  return TIER_TO_KEY[tier] ?? DEFAULT_TIER_KEY
})
const tierPrice = computed(() =>
  getTierPrice(tierKey.value, isYearlySubscription.value)
)

const memberCount = computed(() => members.value.length)
const nextMonthInvoice = computed(() => memberCount.value * tierPrice.value)

const refillsDate = computed(() => {
  if (!subscription.value?.renewalDate) return ''
  const date = new Date(subscription.value.renewalDate)
  const day = String(date.getDate()).padStart(2, '0')
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const year = String(date.getFullYear()).slice(-2)
  return `${month}/${day}/${year}`
})

const creditsRemainingLabel = computed(() =>
  isYearlySubscription.value
    ? t(
        'subscription.creditsRemainingThisYear',
        {
          date: refillsDate.value
        },
        {
          escapeParameter: false
        }
      )
    : t(
        'subscription.creditsRemainingThisMonth',
        {
          date: refillsDate.value
        },
        {
          escapeParameter: false
        }
      )
)

const planTotalCredits = computed(() => {
  const credits = getTierCredits(tierKey.value)
  if (credits === null) return '—'
  const total = isYearlySubscription.value ? credits * 12 : credits
  return n(total)
})

const includedCreditsDisplay = computed(
  () => `${monthlyBonusCredits.value} / ${planTotalCredits.value}`
)

const tierBenefits = computed((): TierBenefit[] => {
  const key = tierKey.value
  const benefits: TierBenefit[] = []

  if (!isInPersonalWorkspace.value) {
    benefits.push({
      key: 'members',
      type: 'icon',
      label: t('subscription.membersLabel', { count: getMaxSeats(key) }),
      icon: 'pi pi-user'
    })
  }

  benefits.push(...getCommonTierBenefits(key, t, n))
  return benefits
})

const { totalCredits, monthlyBonusCredits, prepaidCredits, isLoadingBalance } =
  useSubscriptionCredits()

const { handleAddApiCredits, handleRefresh } = useSubscriptionActions()

// Focus-based polling: refresh balance when user returns from Stripe checkout
const PENDING_TOPUP_KEY = 'pending_topup_timestamp'
const TOPUP_EXPIRY_MS = 5 * 60 * 1000 // 5 minutes

function handleWindowFocus() {
  const timestampStr = localStorage.getItem(PENDING_TOPUP_KEY)
  if (!timestampStr) return

  const timestamp = parseInt(timestampStr, 10)

  // Clear expired tracking (older than 5 minutes)
  if (Date.now() - timestamp > TOPUP_EXPIRY_MS) {
    localStorage.removeItem(PENDING_TOPUP_KEY)
    return
  }

  // Refresh and clear tracking to prevent repeated calls
  void handleRefresh()
  localStorage.removeItem(PENDING_TOPUP_KEY)
}

onMounted(() => {
  window.addEventListener('focus', handleWindowFocus)
  void Promise.all([fetchStatus(), fetchBalance()])
})

onBeforeUnmount(() => {
  window.removeEventListener('focus', handleWindowFocus)
})
</script>

<style scoped>
:deep(.bg-comfy-menu-secondary) {
  background-color: transparent;
}
</style>
