import { onMounted, ref } from 'vue'

import { useBillingContext } from '@/composables/billing/useBillingContext'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import { isCloud } from '@/platform/distribution/types'
import { useTelemetry } from '@/platform/telemetry'
import { useDialogService } from '@/services/dialogService'
import { useCommandStore } from '@/stores/commandStore'

/**
 * Composable for handling subscription panel actions and loading states
 */
export function useSubscriptionActions() {
  const dialogService = useDialogService()
  const authActions = useFirebaseAuthActions()
  const commandStore = useCommandStore()
  const telemetry = useTelemetry()
  const { fetchStatus } = useBillingContext()

  const isLoadingSupport = ref(false)

  onMounted(() => {
    void handleRefresh()
  })

  const handleAddApiCredits = () => {
    void dialogService.showTopUpCreditsDialog()
  }

  const handleMessageSupport = async () => {
    try {
      isLoadingSupport.value = true
      if (isCloud) {
        telemetry?.trackHelpResourceClicked({
          resource_type: 'help_feedback',
          is_external: true,
          source: 'subscription'
        })
      }
      await commandStore.execute('Comfy.ContactSupport')
    } catch (error) {
      console.error('[useSubscriptionActions] Error contacting support:', error)
    } finally {
      isLoadingSupport.value = false
    }
  }

  const handleRefresh = async () => {
    try {
      await Promise.all([authActions.fetchBalance(), fetchStatus()])
    } catch (error) {
      console.error('[useSubscriptionActions] Error refreshing data:', error)
    }
  }

  const handleLearnMoreClick = () => {
    window.open('https://docs.comfy.org/get_started/cloud', '_blank')
  }

  return {
    isLoadingSupport,
    handleAddApiCredits,
    handleMessageSupport,
    handleRefresh,
    handleLearnMoreClick
  }
}
