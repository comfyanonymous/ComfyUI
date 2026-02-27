import { describe, expect, it, vi } from 'vitest'

import { useFreeTierOnboarding } from '@/platform/cloud/onboarding/composables/useFreeTierOnboarding'

const mockRemoteConfig = vi.hoisted(() => ({
  value: { free_tier_credits: 50 } as Record<string, unknown>
}))

vi.mock('@/platform/remoteConfig/remoteConfig', () => ({
  remoteConfig: mockRemoteConfig
}))

describe('useFreeTierOnboarding', () => {
  describe('showEmailForm', () => {
    it('starts as false', () => {
      const { showEmailForm } = useFreeTierOnboarding()
      expect(showEmailForm.value).toBe(false)
    })

    it('switchToEmailForm sets it to true', () => {
      const { showEmailForm, switchToEmailForm } = useFreeTierOnboarding()
      switchToEmailForm()
      expect(showEmailForm.value).toBe(true)
    })

    it('switchToSocialLogin sets it back to false', () => {
      const { showEmailForm, switchToEmailForm, switchToSocialLogin } =
        useFreeTierOnboarding()
      switchToEmailForm()
      switchToSocialLogin()
      expect(showEmailForm.value).toBe(false)
    })
  })

  describe('freeTierCredits', () => {
    it('returns value from remote config', () => {
      const { freeTierCredits } = useFreeTierOnboarding()
      expect(freeTierCredits.value).toBe(50)
    })
  })

  describe('isFreeTierEnabled', () => {
    it('returns true when remote config says enabled', () => {
      mockRemoteConfig.value.new_free_tier_subscriptions = true
      const { isFreeTierEnabled } = useFreeTierOnboarding()
      expect(isFreeTierEnabled.value).toBe(true)
    })

    it('returns false when remote config says disabled', () => {
      mockRemoteConfig.value.new_free_tier_subscriptions = false
      const { isFreeTierEnabled } = useFreeTierOnboarding()
      expect(isFreeTierEnabled.value).toBe(false)
    })

    it('defaults to false when not set in remote config', () => {
      mockRemoteConfig.value = { free_tier_credits: 50 }
      const { isFreeTierEnabled } = useFreeTierOnboarding()
      expect(isFreeTierEnabled.value).toBe(false)
    })
  })
})
