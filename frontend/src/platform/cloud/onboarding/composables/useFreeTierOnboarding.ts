import { computed, ref } from 'vue'

import { getTierCredits } from '@/platform/cloud/subscription/constants/tierPricing'
import { remoteConfig } from '@/platform/remoteConfig/remoteConfig'

export function useFreeTierOnboarding() {
  const showEmailForm = ref(false)
  const freeTierCredits = computed(() => getTierCredits('free'))
  const isFreeTierEnabled = computed(
    () => remoteConfig.value.new_free_tier_subscriptions ?? false
  )

  function switchToEmailForm() {
    showEmailForm.value = true
  }

  function switchToSocialLogin() {
    showEmailForm.value = false
  }

  return {
    showEmailForm,
    freeTierCredits,
    isFreeTierEnabled,
    switchToEmailForm,
    switchToSocialLogin
  }
}
