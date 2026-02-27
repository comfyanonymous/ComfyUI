<!-- A popover that shows current user information and actions -->
<template>
  <div
    class="current-user-popover w-80 -m-3 p-2 rounded-lg border border-border-default bg-base-background shadow-[1px_1px_8px_0_rgba(0,0,0,0.4)]"
  >
    <!-- User Info Section -->
    <div class="flex flex-col items-center px-0 py-3 mb-4">
      <UserAvatar
        class="mb-1"
        :photo-url="userPhotoUrl"
        :pt:icon:class="{
          'text-2xl!': !userPhotoUrl
        }"
        size="large"
      />

      <!-- User Details -->
      <h3 class="my-0 mb-1 truncate text-base font-bold text-base-foreground">
        {{ userDisplayName || $t('g.user') }}
      </h3>
      <p v-if="userEmail" class="my-0 truncate text-sm text-muted">
        {{ userEmail }}
      </p>
      <span
        v-if="subscriptionTierName"
        class="my-0 text-xs text-foreground bg-secondary-background-hover rounded-full uppercase px-2 py-0.5 font-bold mt-2"
      >
        {{ subscriptionTierName }}
      </span>
    </div>

    <!-- Credits Section -->
    <div v-if="isActiveSubscription" class="flex items-center gap-2 px-4 py-2">
      <i class="icon-[lucide--component] text-amber-400 text-sm" />
      <Skeleton
        v-if="authStore.isFetchingBalance"
        width="4rem"
        height="1.25rem"
        class="w-full"
      />
      <span v-else class="text-base font-semibold text-base-foreground">{{
        formattedBalance
      }}</span>
      <i
        v-tooltip="{ value: $t('credits.unified.tooltip'), showDelay: 300 }"
        class="icon-[lucide--circle-help] cursor-help text-base text-muted-foreground mr-auto"
      />
      <Button
        variant="secondary"
        size="sm"
        class="text-base-foreground"
        data-testid="add-credits-button"
        @click="handleTopUp"
      >
        {{ $t('subscription.addCredits') }}
      </Button>
    </div>

    <div v-else class="flex justify-center px-4">
      <SubscribeButton
        :fluid="false"
        :label="$t('subscription.subscribeToComfyCloud')"
        size="sm"
        variant="gradient"
        @subscribed="handleSubscribed"
      />
    </div>

    <Divider class="my-2 mx-0" />

    <div
      v-if="isActiveSubscription"
      class="flex items-center gap-2 px-4 py-2 cursor-pointer hover:bg-secondary-background-hover"
      data-testid="partner-nodes-menu-item"
      @click="handleOpenPartnerNodesInfo"
    >
      <i class="icon-[lucide--tag] text-muted-foreground text-sm" />
      <span class="text-sm text-base-foreground flex-1">{{
        $t('subscription.partnerNodesCredits')
      }}</span>
    </div>

    <div
      class="flex items-center gap-2 px-4 py-2 cursor-pointer hover:bg-secondary-background-hover"
      data-testid="plans-pricing-menu-item"
      @click="handleOpenPlansAndPricing"
    >
      <i class="icon-[lucide--receipt-text] text-muted-foreground text-sm" />
      <span class="text-sm text-base-foreground flex-1">{{
        $t('subscription.plansAndPricing')
      }}</span>
      <span
        v-if="canUpgrade"
        class="text-xs font-bold text-base-background bg-base-foreground px-1.5 py-0.5 rounded-full"
      >
        {{ $t('subscription.upgrade') }}
      </span>
    </div>

    <div
      v-if="isActiveSubscription"
      class="flex items-center gap-2 px-4 py-2 cursor-pointer hover:bg-secondary-background-hover"
      data-testid="manage-plan-menu-item"
      @click="handleOpenPlanAndCreditsSettings"
    >
      <i class="icon-[lucide--file-text] text-muted-foreground text-sm" />
      <span class="text-sm text-base-foreground flex-1">{{
        $t('subscription.managePlan')
      }}</span>
    </div>

    <div
      class="flex items-center gap-2 px-4 py-2 cursor-pointer hover:bg-secondary-background-hover"
      data-testid="user-settings-menu-item"
      @click="handleOpenUserSettings"
    >
      <i class="icon-[lucide--settings-2] text-muted-foreground text-sm" />
      <span class="text-sm text-base-foreground flex-1">{{
        $t('userSettings.accountSettings')
      }}</span>
    </div>

    <Divider class="my-2 mx-0" />

    <div
      class="flex items-center gap-2 px-4 py-2 cursor-pointer hover:bg-secondary-background-hover"
      data-testid="logout-menu-item"
      @click="handleLogout"
    >
      <i class="icon-[lucide--log-out] text-muted-foreground text-sm" />
      <span class="text-sm text-base-foreground flex-1">{{
        $t('auth.signOut.signOut')
      }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import Divider from 'primevue/divider'
import Skeleton from 'primevue/skeleton'
import { computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'

import { formatCreditsFromCents } from '@/base/credits/comfyCredits'
import UserAvatar from '@/components/common/UserAvatar.vue'
import Button from '@/components/ui/button/Button.vue'
import { useCurrentUser } from '@/composables/auth/useCurrentUser'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import { useExternalLink } from '@/composables/useExternalLink'
import SubscribeButton from '@/platform/cloud/subscription/components/SubscribeButton.vue'
import { useSubscription } from '@/platform/cloud/subscription/composables/useSubscription'
import { useSubscriptionDialog } from '@/platform/cloud/subscription/composables/useSubscriptionDialog'
import { isCloud } from '@/platform/distribution/types'
import { useTelemetry } from '@/platform/telemetry'
import { useSettingsDialog } from '@/platform/settings/composables/useSettingsDialog'
import { useDialogService } from '@/services/dialogService'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'

const emit = defineEmits<{
  close: []
}>()

const { buildDocsUrl, docsPaths } = useExternalLink()

const { userDisplayName, userEmail, userPhotoUrl, handleSignOut } =
  useCurrentUser()
const authActions = useFirebaseAuthActions()
const authStore = useFirebaseAuthStore()
const settingsDialog = useSettingsDialog()
const dialogService = useDialogService()
const {
  isActiveSubscription,
  subscriptionTierName,
  subscriptionTier,
  fetchStatus
} = useSubscription()
const subscriptionDialog = useSubscriptionDialog()
const { locale } = useI18n()

const formattedBalance = computed(() => {
  const cents =
    authStore.balance?.effective_balance_micros ??
    authStore.balance?.amount_micros ??
    0
  return formatCreditsFromCents({
    cents,
    locale: locale.value,
    numberOptions: {
      minimumFractionDigits: 0,
      maximumFractionDigits: 2
    }
  })
})

const canUpgrade = computed(() => {
  const tier = subscriptionTier.value
  return (
    tier === 'FREE' ||
    tier === 'FOUNDERS_EDITION' ||
    tier === 'STANDARD' ||
    tier === 'CREATOR'
  )
})

const handleOpenUserSettings = () => {
  settingsDialog.show('user')
  emit('close')
}

const handleOpenPlansAndPricing = () => {
  subscriptionDialog.showPricingTable()
  emit('close')
}

const handleOpenPlanAndCreditsSettings = () => {
  if (isCloud) {
    settingsDialog.show('subscription')
  } else {
    settingsDialog.show('credits')
  }

  emit('close')
}

const handleTopUp = () => {
  // Track purchase credits entry from avatar popover
  useTelemetry()?.trackAddApiCreditButtonClicked()
  dialogService.showTopUpCreditsDialog()
  emit('close')
}

const handleOpenPartnerNodesInfo = () => {
  window.open(
    buildDocsUrl(docsPaths.partnerNodesPricing, { includeLocale: true }),
    '_blank'
  )
  emit('close')
}

const handleLogout = async () => {
  await handleSignOut()
  emit('close')
}

const handleSubscribed = async () => {
  await fetchStatus()
}

onMounted(() => {
  void authActions.fetchBalance()
})
</script>
