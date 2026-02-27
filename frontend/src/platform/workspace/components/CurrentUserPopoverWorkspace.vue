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
    </div>

    <!-- Workspace Selector -->
    <div
      class="flex cursor-pointer items-center justify-between rounded-lg px-4 py-2 hover:bg-secondary-background-hover"
      @click="toggleWorkspaceSwitcher"
    >
      <div class="flex min-w-0 flex-1 items-center gap-2">
        <WorkspaceProfilePic
          class="size-6 shrink-0 text-xs"
          :workspace-name="workspaceName"
        />
        <span class="truncate text-sm text-base-foreground">{{
          workspaceName
        }}</span>
      </div>
      <i class="pi pi-chevron-down shrink-0 text-sm text-muted-foreground" />
    </div>

    <Popover
      ref="workspaceSwitcherPopover"
      append-to="body"
      :pt="{
        content: {
          class: 'p-0'
        }
      }"
    >
      <WorkspaceSwitcherPopover
        @select="workspaceSwitcherPopover?.hide()"
        @create="handleCreateWorkspace"
      />
    </Popover>

    <!-- Credits Section -->

    <div class="flex items-center gap-2 px-4 py-2">
      <i class="icon-[lucide--component] text-sm text-amber-400" />
      <Skeleton
        v-if="isLoadingBalance"
        width="4rem"
        height="1.25rem"
        class="w-full"
      />
      <span v-else class="text-base font-semibold text-base-foreground">{{
        displayedCredits
      }}</span>
      <i
        v-tooltip="{ value: $t('credits.unified.tooltip'), showDelay: 300 }"
        class="icon-[lucide--circle-help] mr-auto cursor-help text-base text-muted-foreground"
      />
      <!-- Add Credits (subscribed + personal or workspace owner only) -->
      <Button
        v-if="isActiveSubscription && permissions.canTopUp"
        variant="secondary"
        size="sm"
        class="text-base-foreground"
        data-testid="add-credits-button"
        @click="handleTopUp"
      >
        {{ $t('subscription.addCredits') }}
      </Button>
      <!-- Subscribe/Resubscribe (only when not subscribed or cancelled) -->
      <SubscribeButton
        v-if="showSubscribeAction && isPersonalWorkspace"
        :fluid="false"
        :label="
          isCancelled
            ? $t('subscription.resubscribe')
            : $t('workspaceSwitcher.subscribe')
        "
        size="sm"
        variant="gradient"
      />
      <Button
        v-if="showSubscribeAction && !isPersonalWorkspace"
        variant="primary"
        size="sm"
        @click="handleOpenPlansAndPricing"
      >
        {{
          isCancelled
            ? $t('subscription.resubscribe')
            : $t('workspaceSwitcher.subscribe')
        }}
      </Button>
    </div>

    <Divider class="mx-0 my-2" />

    <!-- Plans & Pricing (PERSONAL and OWNER only) -->
    <div
      v-if="showPlansAndPricing"
      class="flex cursor-pointer items-center gap-2 px-4 py-2 hover:bg-secondary-background-hover"
      data-testid="plans-pricing-menu-item"
      @click="handleOpenPlansAndPricing"
    >
      <i class="icon-[lucide--receipt-text] text-sm text-muted-foreground" />
      <span class="flex-1 text-sm text-base-foreground">{{
        $t('subscription.plansAndPricing')
      }}</span>
      <span
        v-if="canUpgrade"
        class="rounded-full bg-base-foreground px-1.5 py-0.5 text-xs font-bold text-base-background"
      >
        {{ $t('subscription.upgrade') }}
      </span>
    </div>

    <!-- Manage Plan (PERSONAL and OWNER, only if subscribed) -->
    <div
      v-if="showManagePlan"
      class="flex cursor-pointer items-center gap-2 px-4 py-2 hover:bg-secondary-background-hover"
      data-testid="manage-plan-menu-item"
      @click="handleOpenPlanAndCreditsSettings"
    >
      <i class="icon-[lucide--file-text] text-sm text-muted-foreground" />
      <span class="flex-1 text-sm text-base-foreground">{{
        $t('subscription.managePlan')
      }}</span>
    </div>

    <!-- Partner Nodes Pricing (always shown) -->
    <div
      class="flex cursor-pointer items-center gap-2 px-4 py-2 hover:bg-secondary-background-hover"
      data-testid="partner-nodes-menu-item"
      @click="handleOpenPartnerNodesInfo"
    >
      <i class="icon-[lucide--tag] text-sm text-muted-foreground" />
      <span class="flex-1 text-sm text-base-foreground">{{
        $t('subscription.partnerNodesCredits')
      }}</span>
    </div>

    <Divider class="mx-0 my-2" />

    <!-- Workspace Settings (always shown) -->
    <div
      class="flex cursor-pointer items-center gap-2 px-4 py-2 hover:bg-secondary-background-hover"
      data-testid="workspace-settings-menu-item"
      @click="handleOpenWorkspaceSettings"
    >
      <i class="icon-[lucide--users] text-sm text-muted-foreground" />
      <span class="flex-1 text-sm text-base-foreground">{{
        $t('userSettings.workspaceSettings')
      }}</span>
    </div>

    <!-- Account Settings (always shown) -->
    <div
      class="flex cursor-pointer items-center gap-2 px-4 py-2 hover:bg-secondary-background-hover"
      data-testid="user-settings-menu-item"
      @click="handleOpenUserSettings"
    >
      <i class="icon-[lucide--settings-2] text-sm text-muted-foreground" />
      <span class="flex-1 text-sm text-base-foreground">{{
        $t('userSettings.accountSettings')
      }}</span>
    </div>

    <Divider class="mx-0 my-2" />

    <!-- Logout (always shown) -->
    <div
      class="flex cursor-pointer items-center gap-2 px-4 py-2 hover:bg-secondary-background-hover"
      data-testid="logout-menu-item"
      @click="handleLogout"
    >
      <i class="icon-[lucide--log-out] text-sm text-muted-foreground" />
      <span class="flex-1 text-sm text-base-foreground">{{
        $t('auth.signOut.signOut')
      }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import Divider from 'primevue/divider'
import Popover from 'primevue/popover'
import Skeleton from 'primevue/skeleton'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { formatCreditsFromCents } from '@/base/credits/comfyCredits'
import UserAvatar from '@/components/common/UserAvatar.vue'
import WorkspaceProfilePic from '@/platform/workspace/components/WorkspaceProfilePic.vue'
import WorkspaceSwitcherPopover from '@/platform/workspace/components/WorkspaceSwitcherPopover.vue'
import Button from '@/components/ui/button/Button.vue'
import { useCurrentUser } from '@/composables/auth/useCurrentUser'

import { useExternalLink } from '@/composables/useExternalLink'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import SubscribeButton from '@/platform/cloud/subscription/components/SubscribeButton.vue'
import { useSubscriptionDialog } from '@/platform/cloud/subscription/composables/useSubscriptionDialog'
import { isCloud } from '@/platform/distribution/types'
import { useTelemetry } from '@/platform/telemetry'
import { useWorkspaceUI } from '@/platform/workspace/composables/useWorkspaceUI'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'
import { useSettingsDialog } from '@/platform/settings/composables/useSettingsDialog'
import { useDialogService } from '@/services/dialogService'

const workspaceStore = useTeamWorkspaceStore()
const {
  initState,
  workspaceName,
  isInPersonalWorkspace: isPersonalWorkspace
} = storeToRefs(workspaceStore)
const { permissions } = useWorkspaceUI()
const workspaceSwitcherPopover = ref<InstanceType<typeof Popover> | null>(null)

const emit = defineEmits<{
  close: []
}>()

const { buildDocsUrl, docsPaths } = useExternalLink()

const { userDisplayName, userEmail, userPhotoUrl, handleSignOut } =
  useCurrentUser()
const settingsDialog = useSettingsDialog()
const dialogService = useDialogService()
const { isActiveSubscription, subscription, balance, isLoading, fetchBalance } =
  useBillingContext()

const isCancelled = computed(() => subscription.value?.isCancelled ?? false)
const subscriptionDialog = useSubscriptionDialog()

const { locale } = useI18n()
const isLoadingBalance = isLoading

const displayedCredits = computed(() => {
  if (initState.value !== 'ready') return ''

  // API field is named _micros but contains cents (naming inconsistency)
  const cents =
    balance.value?.effectiveBalanceMicros ?? balance.value?.amountMicros ?? 0
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
  // PRO is currently the only/highest tier, so no upgrades available
  // This will need updating when additional tiers are added
  return false
})

const showPlansAndPricing = computed(
  () => permissions.value.canManageSubscription
)
const showManagePlan = computed(
  () => permissions.value.canManageSubscription && isActiveSubscription.value
)
const showSubscribeAction = computed(
  () =>
    permissions.value.canManageSubscription &&
    (!isActiveSubscription.value || isCancelled.value)
)

const handleOpenUserSettings = () => {
  settingsDialog.show('user')
  emit('close')
}

const handleOpenWorkspaceSettings = () => {
  settingsDialog.show('workspace')
  emit('close')
}

const handleOpenPlansAndPricing = () => {
  subscriptionDialog.showPricingTable()
  emit('close')
}

const handleOpenPlanAndCreditsSettings = () => {
  if (isCloud) {
    settingsDialog.show('workspace')
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

const handleCreateWorkspace = () => {
  workspaceSwitcherPopover.value?.hide()
  dialogService.showCreateWorkspaceDialog()
  emit('close')
}

const toggleWorkspaceSwitcher = (event: MouseEvent) => {
  workspaceSwitcherPopover.value?.toggle(event)
}

const refreshBalance = () => {
  void fetchBalance()
}

defineExpose({ refreshBalance })
</script>
