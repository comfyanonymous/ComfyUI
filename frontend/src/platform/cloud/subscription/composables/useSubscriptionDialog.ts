import { defineAsyncComponent } from 'vue'
import { useDialogService } from '@/services/dialogService'
import { useDialogStore } from '@/stores/dialogStore'
import { useFeatureFlags } from '@/composables/useFeatureFlags'
import { useSubscription } from '@/platform/cloud/subscription/composables/useSubscription'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'

const DIALOG_KEY = 'subscription-required'
const FREE_TIER_DIALOG_KEY = 'free-tier-info'

export type SubscriptionDialogReason =
  | 'subscription_required'
  | 'out_of_credits'
  | 'top_up_blocked'

export const useSubscriptionDialog = () => {
  const { flags } = useFeatureFlags()
  const dialogService = useDialogService()
  const dialogStore = useDialogStore()
  const workspaceStore = useTeamWorkspaceStore()
  const { isFreeTier } = useSubscription()

  function hide() {
    dialogStore.closeDialog({ key: DIALOG_KEY })
    dialogStore.closeDialog({ key: FREE_TIER_DIALOG_KEY })
  }

  function showPricingTable(options?: { reason?: SubscriptionDialogReason }) {
    const useWorkspaceVariant =
      flags.teamWorkspacesEnabled && !workspaceStore.isInPersonalWorkspace

    const component = useWorkspaceVariant
      ? defineAsyncComponent(
          () =>
            import('@/platform/workspace/components/SubscriptionRequiredDialogContentWorkspace.vue')
        )
      : defineAsyncComponent(
          () =>
            import('@/platform/cloud/subscription/components/SubscriptionRequiredDialogContent.vue')
        )

    dialogService.showLayoutDialog({
      key: DIALOG_KEY,
      component,
      props: {
        onClose: hide,
        reason: options?.reason
      },
      dialogComponentProps: {
        style: 'width: min(1328px, 95vw); max-height: 958px;',
        pt: {
          root: {
            class: 'rounded-2xl bg-transparent h-full'
          },
          content: {
            class:
              '!p-0 rounded-2xl border border-border-default bg-base-background/60 backdrop-blur-md shadow-[0_25px_80px_rgba(5,6,12,0.45)] h-full'
          }
        }
      }
    })
  }

  function show(options?: { reason?: SubscriptionDialogReason }) {
    if (isFreeTier.value && workspaceStore.isInPersonalWorkspace) {
      const component = defineAsyncComponent(
        () =>
          import('@/platform/cloud/subscription/components/FreeTierDialogContent.vue')
      )

      dialogService.showLayoutDialog({
        key: FREE_TIER_DIALOG_KEY,
        component,
        props: {
          reason: options?.reason,
          onClose: hide,
          onUpgrade: () => {
            hide()
            showPricingTable(options)
          }
        },
        dialogComponentProps: {
          style: 'width: min(640px, 95vw);',
          pt: {
            root: {
              class: 'rounded-2xl bg-transparent'
            },
            content: {
              class:
                '!p-0 rounded-2xl border border-border-default bg-base-background/60 backdrop-blur-md shadow-[0_25px_80px_rgba(5,6,12,0.45)]'
            }
          }
        }
      })
      return
    }

    showPricingTable(options)
  }

  return {
    show,
    showPricingTable,
    hide
  }
}
