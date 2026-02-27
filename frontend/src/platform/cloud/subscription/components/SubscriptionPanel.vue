<template>
  <div class="subscription-container h-full">
    <div class="flex h-full flex-col gap-6">
      <div class="flex items-center gap-2">
        <span class="text-2xl font-inter font-semibold leading-tight">
          {{
            isActiveSubscription
              ? $t('subscription.title')
              : $t('subscription.titleUnsubscribed')
          }}
        </span>
        <div class="pt-1">
          <CloudBadge
            reverse-order
            background-color="var(--p-dialog-background)"
          />
        </div>
      </div>

      <!-- Workspace mode: workspace-aware subscription content -->
      <SubscriptionPanelContentWorkspace v-if="teamWorkspacesEnabled" />
      <!-- Legacy mode: user-level subscription content -->
      <SubscriptionPanelContentLegacy v-else />

      <div
        class="flex items-center justify-between border-t border-interface-stroke pt-3"
      >
        <div class="flex gap-2">
          <Button
            variant="muted-textonly"
            class="text-xs text-text-secondary"
            @click="handleLearnMoreClick"
          >
            <i class="pi pi-question-circle text-text-secondary text-xs" />
            {{ $t('subscription.learnMore') }}
          </Button>
          <Button
            variant="muted-textonly"
            class="text-xs text-text-secondary"
            @click="handleOpenPartnerNodesInfo"
          >
            <i class="pi pi-question-circle text-text-secondary text-xs" />
            {{ $t('subscription.partnerNodesCredits') }}
          </Button>
          <Button
            variant="muted-textonly"
            class="text-xs text-text-secondary"
            :loading="isLoadingSupport"
            @click="handleMessageSupport"
          >
            <i class="pi pi-comment text-text-secondary text-xs" />
            {{ $t('subscription.messageSupport') }}
          </Button>
        </div>

        <Button
          variant="muted-textonly"
          class="text-xs text-text-secondary"
          @click="handleInvoiceHistory"
        >
          {{ $t('subscription.invoiceHistory') }}
          <i class="pi pi-external-link text-text-secondary text-xs" />
        </Button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, defineAsyncComponent } from 'vue'

import CloudBadge from '@/components/topbar/CloudBadge.vue'
import Button from '@/components/ui/button/Button.vue'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { useExternalLink } from '@/composables/useExternalLink'
import { useFeatureFlags } from '@/composables/useFeatureFlags'
import SubscriptionPanelContentLegacy from '@/platform/cloud/subscription/components/SubscriptionPanelContentLegacy.vue'
import { useSubscriptionActions } from '@/platform/cloud/subscription/composables/useSubscriptionActions'
import { isCloud } from '@/platform/distribution/types'

const SubscriptionPanelContentWorkspace = defineAsyncComponent(
  () =>
    import('@/platform/workspace/components/SubscriptionPanelContentWorkspace.vue')
)

const { flags } = useFeatureFlags()
const teamWorkspacesEnabled = computed(
  () => isCloud && flags.teamWorkspacesEnabled
)

const { buildDocsUrl, docsPaths } = useExternalLink()

const { isActiveSubscription, manageSubscription } = useBillingContext()

const { isLoadingSupport, handleMessageSupport, handleLearnMoreClick } =
  useSubscriptionActions()

const handleInvoiceHistory = async () => {
  await manageSubscription()
}

const handleOpenPartnerNodesInfo = () => {
  window.open(
    buildDocsUrl(docsPaths.partnerNodesPricing, { includeLocale: true }),
    '_blank'
  )
}
</script>
