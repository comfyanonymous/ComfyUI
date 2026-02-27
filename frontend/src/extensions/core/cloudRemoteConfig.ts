import { watchDebounced } from '@vueuse/core'

import { useCurrentUser } from '@/composables/auth/useCurrentUser'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { refreshRemoteConfig } from '@/platform/remoteConfig/refreshRemoteConfig'
import { useExtensionService } from '@/services/extensionService'

/**
 * Cloud-only extension that polls for remote config updates
 * Initial config load happens in main.ts before any other imports
 */
useExtensionService().registerExtension({
  name: 'Comfy.Cloud.RemoteConfig',

  setup: async () => {
    const { isLoggedIn } = useCurrentUser()
    const { isActiveSubscription } = useBillingContext()

    // Refresh config when auth or subscription status changes
    // Primary auth refresh is handled by WorkspaceAuthGate on mount
    // This watcher handles subscription changes and acts as a backup for auth
    watchDebounced(
      [isLoggedIn, isActiveSubscription],
      () => {
        if (!isLoggedIn.value) return
        void refreshRemoteConfig()
      },
      { debounce: 256, immediate: true }
    )

    // Poll for config updates every 10 minutes (with auth)
    setInterval(() => void refreshRemoteConfig(), 600_000)
  }
})
