import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

describe('useConflictAcknowledgment', () => {
  beforeEach(() => {
    // Set up Pinia for each test
    setActivePinia(createTestingPinia({ stubActions: false }))
    // Clear localStorage before each test
    localStorage.clear()
    // Reset modules to ensure fresh state
    vi.resetModules()
  })

  afterEach(() => {
    localStorage.clear()
  })

  describe('initial state loading', () => {
    it('should load empty state when localStorage is empty', async () => {
      vi.resetModules()
      const { useConflictAcknowledgment } =
        await import('@/workbench/extensions/manager/composables/useConflictAcknowledgment')
      const { acknowledgmentState } = useConflictAcknowledgment()

      expect(acknowledgmentState.value).toEqual({
        modal_dismissed: false,
        red_dot_dismissed: false,
        warning_banner_dismissed: false
      })
    })

    it('should load existing state from localStorage', async () => {
      // Pre-populate localStorage with JSON values (as useStorage expects)
      localStorage.setItem('Comfy.ConflictModalDismissed', JSON.stringify(true))
      localStorage.setItem(
        'Comfy.ConflictRedDotDismissed',
        JSON.stringify(true)
      )
      localStorage.setItem(
        'Comfy.ConflictWarningBannerDismissed',
        JSON.stringify(true)
      )

      // Need to import the module after localStorage is set
      vi.resetModules()
      const { useConflictAcknowledgment } =
        await import('@/workbench/extensions/manager/composables/useConflictAcknowledgment')
      const { acknowledgmentState } = useConflictAcknowledgment()

      expect(acknowledgmentState.value).toEqual({
        modal_dismissed: true,
        red_dot_dismissed: true,
        warning_banner_dismissed: true
      })
    })
  })

  describe('dismissal functions', () => {
    it('should mark conflicts as seen with unified function', async () => {
      vi.resetModules()
      const { useConflictAcknowledgment } =
        await import('@/workbench/extensions/manager/composables/useConflictAcknowledgment')
      const { markConflictsAsSeen, acknowledgmentState } =
        useConflictAcknowledgment()

      markConflictsAsSeen()

      expect(acknowledgmentState.value.modal_dismissed).toBe(true)
    })

    it('should dismiss red dot notification', async () => {
      vi.resetModules()
      const { useConflictAcknowledgment } =
        await import('@/workbench/extensions/manager/composables/useConflictAcknowledgment')
      const { dismissRedDotNotification, acknowledgmentState } =
        useConflictAcknowledgment()

      dismissRedDotNotification()

      expect(acknowledgmentState.value.red_dot_dismissed).toBe(true)
    })

    it('should dismiss warning banner', async () => {
      vi.resetModules()
      const { useConflictAcknowledgment } =
        await import('@/workbench/extensions/manager/composables/useConflictAcknowledgment')
      const { dismissWarningBanner, acknowledgmentState } =
        useConflictAcknowledgment()

      dismissWarningBanner()

      expect(acknowledgmentState.value.warning_banner_dismissed).toBe(true)
    })

    it('should mark all conflicts as seen', async () => {
      vi.resetModules()
      const { useConflictAcknowledgment } =
        await import('@/workbench/extensions/manager/composables/useConflictAcknowledgment')
      const { markConflictsAsSeen, acknowledgmentState } =
        useConflictAcknowledgment()

      markConflictsAsSeen()

      expect(acknowledgmentState.value.modal_dismissed).toBe(true)
      expect(acknowledgmentState.value.red_dot_dismissed).toBe(true)
      expect(acknowledgmentState.value.warning_banner_dismissed).toBe(true)
    })
  })

  describe('computed properties', () => {
    it('should calculate shouldShowConflictModal correctly', async () => {
      // Need fresh module import to ensure clean state
      vi.resetModules()
      const { useConflictAcknowledgment } =
        await import('@/workbench/extensions/manager/composables/useConflictAcknowledgment')
      const { shouldShowConflictModal, markConflictsAsSeen } =
        useConflictAcknowledgment()

      expect(shouldShowConflictModal.value).toBe(true)

      markConflictsAsSeen()
      expect(shouldShowConflictModal.value).toBe(false)
    })

    it('should calculate shouldShowRedDot correctly based on conflicts', async () => {
      vi.resetModules()
      const { useConflictAcknowledgment } =
        await import('@/workbench/extensions/manager/composables/useConflictAcknowledgment')
      const { shouldShowRedDot, dismissRedDotNotification } =
        useConflictAcknowledgment()

      // Initially false because no conflicts exist
      expect(shouldShowRedDot.value).toBe(false)

      dismissRedDotNotification()
      expect(shouldShowRedDot.value).toBe(false)
    })

    it('should calculate shouldShowManagerBanner correctly', async () => {
      vi.resetModules()
      const { useConflictAcknowledgment } =
        await import('@/workbench/extensions/manager/composables/useConflictAcknowledgment')
      const { shouldShowManagerBanner, dismissWarningBanner } =
        useConflictAcknowledgment()

      // Initially false because no conflicts exist
      expect(shouldShowManagerBanner.value).toBe(false)

      dismissWarningBanner()
      expect(shouldShowManagerBanner.value).toBe(false)
    })
  })

  describe('localStorage persistence', () => {
    it('should persist to localStorage automatically', async () => {
      // Need fresh module import to ensure clean state
      vi.resetModules()
      const { useConflictAcknowledgment } =
        await import('@/workbench/extensions/manager/composables/useConflictAcknowledgment')
      const { markConflictsAsSeen, dismissWarningBanner } =
        useConflictAcknowledgment()

      markConflictsAsSeen()
      dismissWarningBanner()

      // Wait a tick for useStorage to sync
      await new Promise((resolve) => setTimeout(resolve, 10))

      // VueUse useStorage should automatically persist to localStorage as JSON
      expect(localStorage.getItem('Comfy.ConflictModalDismissed')).toBe('true')
      expect(localStorage.getItem('Comfy.ConflictWarningBannerDismissed')).toBe(
        'true'
      )
    })
  })
})
