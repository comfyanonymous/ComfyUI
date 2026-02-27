<template>
  <div
    class="flex w-full max-w-[512px] flex-col rounded-2xl border border-border-default bg-base-background"
  >
    <!-- Header -->
    <div
      class="flex h-12 items-center justify-between border-b border-border-default px-4"
    >
      <h2 class="m-0 text-sm font-normal text-base-foreground">
        {{
          isActiveSubscription
            ? $t('workspacePanel.inviteUpsellDialog.titleSingleSeat')
            : $t('workspacePanel.inviteUpsellDialog.titleNotSubscribed')
        }}
      </h2>
      <button
        class="cursor-pointer rounded border-none bg-transparent p-0 text-muted-foreground transition-colors hover:text-base-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-secondary-foreground"
        :aria-label="$t('g.close')"
        @click="onDismiss"
      >
        <i class="pi pi-times size-4" />
      </button>
    </div>

    <!-- Body -->
    <div class="flex flex-col gap-4 px-4 py-4">
      <p class="m-0 text-sm text-muted-foreground">
        {{
          isActiveSubscription
            ? $t('workspacePanel.inviteUpsellDialog.messageSingleSeat')
            : $t('workspacePanel.inviteUpsellDialog.messageNotSubscribed')
        }}
      </p>
    </div>

    <!-- Footer -->
    <div class="flex items-center justify-end gap-4 px-4 py-4">
      <Button variant="muted-textonly" @click="onDismiss">
        {{ $t('g.cancel') }}
      </Button>
      <Button variant="primary" size="lg" @click="onUpgrade">
        {{
          isActiveSubscription
            ? $t('workspacePanel.inviteUpsellDialog.upgradeToCreator')
            : $t('workspacePanel.inviteUpsellDialog.viewPlans')
        }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { useDialogStore } from '@/stores/dialogStore'

const dialogStore = useDialogStore()
const { isActiveSubscription, showSubscriptionDialog } = useBillingContext()

function onDismiss() {
  dialogStore.closeDialog({ key: 'invite-member-upsell' })
}

function onUpgrade() {
  dialogStore.closeDialog({ key: 'invite-member-upsell' })
  showSubscriptionDialog()
}
</script>
