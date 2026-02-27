import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import { useToastStore } from '@/platform/updates/common/toastStore'
import { useFrontendVersionMismatchWarning } from '@/platform/updates/common/useFrontendVersionMismatchWarning'
import { useVersionCompatibilityStore } from '@/platform/updates/common/versionCompatibilityStore'

// Mock globals
//@ts-expect-error Define global for the test
global.__COMFYUI_FRONTEND_VERSION__ = '1.0.0'

// Mock config first - this needs to be before any imports
vi.mock('@/config', () => ({
  default: {
    app_title: 'ComfyUI',
    app_version: '1.0.0'
  }
}))

// Mock app
vi.mock('@/scripts/app', () => ({
  app: {
    ui: {
      settings: {
        dispatchChange: vi.fn()
      }
    }
  }
}))

// Mock api
vi.mock('@/scripts/api', () => ({
  api: {
    getSettings: vi.fn(() => Promise.resolve({})),
    storeSetting: vi.fn(() => Promise.resolve(undefined))
  }
}))

// Mock vue-i18n
vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: (key: string, params?: Record<string, string | number> | unknown) => {
      if (key === 'g.versionMismatchWarning')
        return 'Version Compatibility Warning'
      if (key === 'g.versionMismatchWarningMessage' && params) {
        const p = params as Record<string, string>
        return `${p.warning}: ${p.detail} Visit https://docs.comfy.org/installation/update_comfyui#common-update-issues for update instructions.`
      }
      if (key === 'g.frontendOutdated' && params) {
        const p = params as Record<string, string>
        return `Frontend version ${p.frontendVersion} is outdated. Backend requires ${p.requiredVersion} or higher.`
      }
      if (key === 'g.frontendNewer' && params) {
        const p = params as Record<string, string>
        return `Frontend version ${p.frontendVersion} may not be compatible with backend version ${p.backendVersion}.`
      }
      return key
    }
  }),
  createI18n: vi.fn(() => ({
    global: {
      locale: { value: 'en' },
      t: vi.fn()
    }
  }))
}))

// Mock lifecycle hooks to track their calls
const mockOnMounted = vi.fn()
vi.mock('vue', async () => {
  const actual = await vi.importActual('vue')
  return {
    ...actual,
    onMounted: (fn: () => void) => {
      mockOnMounted()
      fn()
    }
  }
})

describe('useFrontendVersionMismatchWarning', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('should not show warning when there is no version mismatch', () => {
    const toastStore = useToastStore()
    const versionStore = useVersionCompatibilityStore()
    const addAlertSpy = vi.spyOn(toastStore, 'addAlert')

    // Mock no version mismatch
    vi.spyOn(versionStore, 'shouldShowWarning', 'get').mockReturnValue(false)

    useFrontendVersionMismatchWarning()

    expect(addAlertSpy).not.toHaveBeenCalled()
  })

  it('should show warning immediately when immediate option is true and there is a mismatch', async () => {
    const toastStore = useToastStore()
    const versionStore = useVersionCompatibilityStore()
    const addAlertSpy = vi.spyOn(toastStore, 'addAlert')
    const dismissWarningSpy = vi.spyOn(versionStore, 'dismissWarning')

    // Mock version mismatch
    vi.spyOn(versionStore, 'shouldShowWarning', 'get').mockReturnValue(true)
    vi.spyOn(versionStore, 'warningMessage', 'get').mockReturnValue({
      type: 'outdated',
      frontendVersion: '1.0.0',
      requiredVersion: '2.0.0'
    })

    useFrontendVersionMismatchWarning({ immediate: true })

    // For immediate: true, the watcher should fire immediately in onMounted
    await nextTick()

    expect(addAlertSpy).toHaveBeenCalledWith(
      expect.stringContaining('Version Compatibility Warning')
    )
    expect(addAlertSpy).toHaveBeenCalledWith(
      expect.stringContaining('Frontend version 1.0.0 is outdated')
    )
    // Should automatically dismiss the warning
    expect(dismissWarningSpy).toHaveBeenCalled()
  })

  it('should not show warning immediately when immediate option is false', async () => {
    const toastStore = useToastStore()
    const versionStore = useVersionCompatibilityStore()
    const addAlertSpy = vi.spyOn(toastStore, 'addAlert')

    // Mock version mismatch
    vi.spyOn(versionStore, 'shouldShowWarning', 'get').mockReturnValue(true)
    vi.spyOn(versionStore, 'warningMessage', 'get').mockReturnValue({
      type: 'outdated',
      frontendVersion: '1.0.0',
      requiredVersion: '2.0.0'
    })

    const result = useFrontendVersionMismatchWarning({ immediate: false })
    await nextTick()

    // Should not show automatically
    expect(addAlertSpy).not.toHaveBeenCalled()

    // But should show when called manually
    result.showWarning()
    expect(addAlertSpy).toHaveBeenCalledOnce()
  })

  it('should call showWarning method manually', () => {
    const toastStore = useToastStore()
    const versionStore = useVersionCompatibilityStore()
    const addAlertSpy = vi.spyOn(toastStore, 'addAlert')
    const dismissWarningSpy = vi.spyOn(versionStore, 'dismissWarning')

    vi.spyOn(versionStore, 'warningMessage', 'get').mockReturnValue({
      type: 'outdated',
      frontendVersion: '1.0.0',
      requiredVersion: '2.0.0'
    })

    const { showWarning } = useFrontendVersionMismatchWarning()
    showWarning()

    expect(addAlertSpy).toHaveBeenCalledOnce()
    expect(dismissWarningSpy).toHaveBeenCalled()
  })

  it('should expose store methods and computed values', () => {
    const versionStore = useVersionCompatibilityStore()

    const mockDismissWarning = vi.fn()
    vi.spyOn(versionStore, 'dismissWarning').mockImplementation(
      mockDismissWarning
    )
    vi.spyOn(versionStore, 'shouldShowWarning', 'get').mockReturnValue(true)
    vi.spyOn(versionStore, 'hasVersionMismatch', 'get').mockReturnValue(true)

    const result = useFrontendVersionMismatchWarning()

    expect(result.shouldShowWarning.value).toBe(true)
    expect(result.hasVersionMismatch.value).toBe(true)

    void result.dismissWarning()
    expect(mockDismissWarning).toHaveBeenCalled()
  })

  it('should register onMounted hook', () => {
    useFrontendVersionMismatchWarning()

    expect(mockOnMounted).toHaveBeenCalledOnce()
  })

  it('should not show warning when warningMessage is null', () => {
    const toastStore = useToastStore()
    const versionStore = useVersionCompatibilityStore()
    const addAlertSpy = vi.spyOn(toastStore, 'addAlert')

    vi.spyOn(versionStore, 'warningMessage', 'get').mockReturnValue(null)

    const { showWarning } = useFrontendVersionMismatchWarning()
    showWarning()

    expect(addAlertSpy).not.toHaveBeenCalled()
  })

  it('should only show warning once even if called multiple times', () => {
    const toastStore = useToastStore()
    const versionStore = useVersionCompatibilityStore()
    const addAlertSpy = vi.spyOn(toastStore, 'addAlert')

    vi.spyOn(versionStore, 'warningMessage', 'get').mockReturnValue({
      type: 'outdated',
      frontendVersion: '1.0.0',
      requiredVersion: '2.0.0'
    })

    const { showWarning } = useFrontendVersionMismatchWarning()

    // Call showWarning multiple times
    showWarning()
    showWarning()
    showWarning()

    // Should only have been called once
    expect(addAlertSpy).toHaveBeenCalledTimes(1)
  })
})
