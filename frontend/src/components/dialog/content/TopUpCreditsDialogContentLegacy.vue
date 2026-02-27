<template>
  <div
    class="flex min-w-[460px] flex-col rounded-2xl border border-border-default bg-base-background shadow-[1px_1px_8px_0_rgba(0,0,0,0.4)]"
  >
    <!-- Header -->
    <div class="flex py-8 items-center justify-between px-8">
      <h2 class="text-lg font-bold text-base-foreground m-0">
        {{
          isInsufficientCredits
            ? $t('credits.topUp.addMoreCreditsToRun')
            : $t('credits.topUp.addMoreCredits')
        }}
      </h2>
      <button
        class="cursor-pointer rounded border-none bg-transparent p-0 text-muted-foreground transition-colors hover:text-base-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-secondary-foreground"
        @click="() => handleClose()"
      >
        <i class="icon-[lucide--x] size-6" />
      </button>
    </div>
    <p
      v-if="isInsufficientCredits"
      class="text-sm text-muted-foreground m-0 px-8"
    >
      {{ $t('credits.topUp.insufficientWorkflowMessage') }}
    </p>

    <!-- Preset amount buttons -->
    <div class="px-8">
      <h3 class="m-0 text-sm font-normal text-muted-foreground">
        {{ $t('credits.topUp.selectAmount') }}
      </h3>
      <div class="flex gap-2 pt-3">
        <Button
          v-for="amount in PRESET_AMOUNTS"
          :key="amount"
          :autofocus="amount === 50"
          variant="secondary"
          size="lg"
          :class="
            cn(
              'h-10 text-base font-medium w-full focus-visible:ring-secondary-foreground',
              selectedPreset === amount && 'bg-secondary-background-selected'
            )
          "
          @click="handlePresetClick(amount)"
        >
          ${{ amount }}
        </Button>
      </div>
    </div>
    <!-- Amount (USD) / Credits -->
    <div class="flex gap-2 px-8 pt-8">
      <!-- You Pay -->
      <div class="flex flex-1 flex-col gap-3">
        <div class="text-sm text-muted-foreground">
          {{ $t('credits.topUp.youPay') }}
        </div>
        <FormattedNumberStepper
          :model-value="payAmount"
          :min="0"
          :max="MAX_AMOUNT"
          :step="getStepAmount"
          @update:model-value="handlePayAmountChange"
          @max-reached="showCeilingWarning = true"
        >
          <template #prefix>
            <span class="shrink-0 text-base font-semibold text-base-foreground"
              >$</span
            >
          </template>
        </FormattedNumberStepper>
      </div>

      <!-- You Get -->
      <div class="flex flex-1 flex-col gap-3">
        <div class="text-sm text-muted-foreground">
          {{ $t('credits.topUp.youGet') }}
        </div>
        <FormattedNumberStepper
          v-model="creditsModel"
          :min="0"
          :max="usdToCredits(MAX_AMOUNT)"
          :step="getCreditsStepAmount"
          @max-reached="showCeilingWarning = true"
        >
          <template #prefix>
            <i class="icon-[lucide--component] size-4 shrink-0 text-gold-500" />
          </template>
        </FormattedNumberStepper>
      </div>
    </div>

    <!-- Warnings -->

    <p
      v-if="isBelowMin"
      class="text-sm text-red-500 m-0 px-8 pt-4 text-center flex items-center justify-center gap-1"
    >
      <i class="icon-[lucide--component] size-4" />
      {{
        $t('credits.topUp.minRequired', {
          credits: formatNumber(usdToCredits(MIN_AMOUNT))
        })
      }}
    </p>
    <p
      v-if="showCeilingWarning"
      class="text-sm text-gold-500 m-0 px-8 pt-4 text-center flex items-center justify-center gap-1"
    >
      <i class="icon-[lucide--component] size-4" />
      {{
        $t('credits.topUp.maxAllowed', {
          credits: formatNumber(usdToCredits(MAX_AMOUNT))
        })
      }}
      <span>{{ $t('credits.topUp.needMore') }}</span>
      <a
        href="https://www.comfy.org/cloud/enterprise"
        target="_blank"
        class="ml-1 text-inherit"
        >{{ $t('credits.topUp.contactUs') }}</a
      >
    </p>

    <div class="pt-8 pb-8 flex flex-col gap-8 px-8">
      <Button
        :disabled="!isValidAmount || loading"
        :loading="loading"
        variant="primary"
        size="lg"
        class="h-10 justify-center"
        @click="handleBuy"
      >
        {{ $t('credits.topUp.buyCredits') }}
      </Button>
      <div class="flex items-center justify-center gap-1">
        <a
          :href="pricingUrl"
          target="_blank"
          class="flex items-center gap-1 text-sm text-muted-foreground no-underline transition-colors hover:text-base-foreground"
        >
          {{ $t('credits.topUp.viewPricing') }}
          <i class="icon-[lucide--external-link] size-4" />
        </a>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useToast } from 'primevue/usetoast'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { creditsToUsd, usdToCredits } from '@/base/credits/comfyCredits'
import Button from '@/components/ui/button/Button.vue'
import FormattedNumberStepper from '@/components/ui/stepper/FormattedNumberStepper.vue'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import { useExternalLink } from '@/composables/useExternalLink'
import { useFeatureFlags } from '@/composables/useFeatureFlags'
import { useSubscription } from '@/platform/cloud/subscription/composables/useSubscription'
import { useTelemetry } from '@/platform/telemetry'
import { clearTopupTracking } from '@/platform/telemetry/topupTracker'
import { useSettingsDialog } from '@/platform/settings/composables/useSettingsDialog'
import { useDialogStore } from '@/stores/dialogStore'
import { cn } from '@/utils/tailwindUtil'

const { isInsufficientCredits = false } = defineProps<{
  isInsufficientCredits?: boolean
}>()

const { t } = useI18n()
const authActions = useFirebaseAuthActions()
const dialogStore = useDialogStore()
const settingsDialog = useSettingsDialog()
const telemetry = useTelemetry()
const toast = useToast()
const { buildDocsUrl, docsPaths } = useExternalLink()
const { flags } = useFeatureFlags()

const { isSubscriptionEnabled } = useSubscription()
// Constants
const PRESET_AMOUNTS = [10, 25, 50, 100]
const MIN_AMOUNT = 5
const MAX_AMOUNT = 10000

// State
const selectedPreset = ref<number | null>(50)
const payAmount = ref(50)
const showCeilingWarning = ref(false)
const loading = ref(false)

// Computed
const pricingUrl = computed(() =>
  buildDocsUrl(docsPaths.partnerNodesPricing, { includeLocale: true })
)

const creditsModel = computed({
  get: () => usdToCredits(payAmount.value),
  set: (newCredits: number) => {
    payAmount.value = Math.round(creditsToUsd(newCredits))
    selectedPreset.value = null
  }
})

const isValidAmount = computed(
  () => payAmount.value >= MIN_AMOUNT && payAmount.value <= MAX_AMOUNT
)

const isBelowMin = computed(() => payAmount.value < MIN_AMOUNT)

// Utility functions
function formatNumber(num: number): string {
  return num.toLocaleString('en-US')
}

// Step amount functions
function getStepAmount(currentAmount: number): number {
  if (currentAmount < 100) return 5
  if (currentAmount < 1000) return 50
  return 100
}

function getCreditsStepAmount(currentCredits: number): number {
  const usdAmount = creditsToUsd(currentCredits)
  return usdToCredits(getStepAmount(usdAmount))
}

// Event handlers
function handlePayAmountChange(value: number) {
  payAmount.value = value
  selectedPreset.value = null
  showCeilingWarning.value = false
}

function handlePresetClick(amount: number) {
  showCeilingWarning.value = false
  payAmount.value = amount
  selectedPreset.value = amount
}

function handleClose(clearTracking = true) {
  if (clearTracking) {
    clearTopupTracking()
  }
  dialogStore.closeDialog({ key: 'top-up-credits' })
}

async function handleBuy() {
  // Prevent double-clicks
  if (loading.value || !isValidAmount.value) return

  loading.value = true
  try {
    telemetry?.trackApiCreditTopupButtonPurchaseClicked(payAmount.value)
    await authActions.purchaseCredits(payAmount.value)

    // Close top-up dialog (keep tracking) and open credits panel to show updated balance
    handleClose(false)

    // In workspace mode (personal workspace), show workspace settings panel
    // Otherwise, show legacy subscription/credits panel
    const settingsPanel = flags.teamWorkspacesEnabled
      ? 'workspace'
      : isSubscriptionEnabled()
        ? 'subscription'
        : 'credits'
    settingsDialog.show(settingsPanel)
  } catch (error) {
    console.error('Purchase failed:', error)

    const errorMessage =
      error instanceof Error ? error.message : t('credits.topUp.unknownError')
    toast.add({
      severity: 'error',
      summary: t('credits.topUp.purchaseError'),
      detail: t('credits.topUp.purchaseErrorDetail', { error: errorMessage }),
      life: 5000
    })
  } finally {
    loading.value = false
  }
}
</script>
