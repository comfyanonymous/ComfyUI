import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import WorkspaceAuthGate from './WorkspaceAuthGate.vue'

const mockIsInitialized = ref(false)
const mockCurrentUser = ref<object | null>(null)

vi.mock('@/stores/firebaseAuthStore', () => ({
  useFirebaseAuthStore: () => ({
    isInitialized: mockIsInitialized,
    currentUser: mockCurrentUser
  })
}))

const mockRefreshRemoteConfig = vi.fn()
vi.mock('@/platform/remoteConfig/refreshRemoteConfig', () => ({
  refreshRemoteConfig: (options: unknown) => mockRefreshRemoteConfig(options)
}))

const mockTeamWorkspacesEnabled = vi.hoisted(() => ({ value: false }))
vi.mock('@/composables/useFeatureFlags', () => ({
  useFeatureFlags: () => ({
    flags: {
      get teamWorkspacesEnabled() {
        return mockTeamWorkspacesEnabled.value
      }
    }
  })
}))

const mockWorkspaceStoreInitialize = vi.fn()
const mockWorkspaceStoreInitState = vi.hoisted(() => ({
  value: 'uninitialized' as string
}))
vi.mock('@/platform/workspace/stores/teamWorkspaceStore', () => ({
  useTeamWorkspaceStore: () => ({
    get initState() {
      return mockWorkspaceStoreInitState.value
    },
    initialize: mockWorkspaceStoreInitialize
  })
}))

const mockIsCloud = vi.hoisted(() => ({ value: true }))
vi.mock('@/platform/distribution/types', () => ({
  get isCloud() {
    return mockIsCloud.value
  }
}))

vi.mock('primevue/progressspinner', () => ({
  default: { template: '<div class="progress-spinner" />' }
}))

describe('WorkspaceAuthGate', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockIsCloud.value = true
    mockIsInitialized.value = false
    mockCurrentUser.value = null
    mockTeamWorkspacesEnabled.value = false
    mockWorkspaceStoreInitState.value = 'uninitialized'
    mockRefreshRemoteConfig.mockResolvedValue(undefined)
    mockWorkspaceStoreInitialize.mockResolvedValue(undefined)
  })

  const mountComponent = () =>
    mount(WorkspaceAuthGate, {
      slots: {
        default: '<div data-testid="slot-content">App Content</div>'
      }
    })

  describe('non-cloud builds', () => {
    it('renders slot immediately when isCloud is false', async () => {
      mockIsCloud.value = false

      const wrapper = mountComponent()
      await flushPromises()

      expect(wrapper.find('[data-testid="slot-content"]').exists()).toBe(true)
      expect(wrapper.find('.progress-spinner').exists()).toBe(false)
      expect(mockRefreshRemoteConfig).not.toHaveBeenCalled()
    })
  })

  describe('cloud builds - unauthenticated user', () => {
    it('shows spinner while waiting for Firebase auth', () => {
      mockIsInitialized.value = false

      const wrapper = mountComponent()

      expect(wrapper.find('.progress-spinner').exists()).toBe(true)
      expect(wrapper.find('[data-testid="slot-content"]').exists()).toBe(false)
    })

    it('renders slot when Firebase initializes with no user', async () => {
      mockIsInitialized.value = false

      const wrapper = mountComponent()
      expect(wrapper.find('.progress-spinner').exists()).toBe(true)

      mockIsInitialized.value = true
      mockCurrentUser.value = null
      await flushPromises()

      expect(wrapper.find('[data-testid="slot-content"]').exists()).toBe(true)
      expect(mockRefreshRemoteConfig).not.toHaveBeenCalled()
    })
  })

  describe('cloud builds - authenticated user', () => {
    beforeEach(() => {
      mockIsInitialized.value = true
      mockCurrentUser.value = { uid: 'user-123' }
    })

    it('refreshes remote config with auth after Firebase init', async () => {
      mountComponent()
      await flushPromises()

      expect(mockRefreshRemoteConfig).toHaveBeenCalledWith({ useAuth: true })
    })

    it('renders slot when teamWorkspacesEnabled is false', async () => {
      mockTeamWorkspacesEnabled.value = false

      const wrapper = mountComponent()
      await flushPromises()

      expect(wrapper.find('[data-testid="slot-content"]').exists()).toBe(true)
      expect(mockWorkspaceStoreInitialize).not.toHaveBeenCalled()
    })

    it('initializes workspace store when teamWorkspacesEnabled is true', async () => {
      mockTeamWorkspacesEnabled.value = true

      const wrapper = mountComponent()
      await flushPromises()

      expect(mockWorkspaceStoreInitialize).toHaveBeenCalled()
      expect(wrapper.find('[data-testid="slot-content"]').exists()).toBe(true)
    })

    it('skips workspace init when store is already initialized', async () => {
      mockTeamWorkspacesEnabled.value = true
      mockWorkspaceStoreInitState.value = 'ready'

      const wrapper = mountComponent()
      await flushPromises()

      expect(mockWorkspaceStoreInitialize).not.toHaveBeenCalled()
      expect(wrapper.find('[data-testid="slot-content"]').exists()).toBe(true)
    })
  })

  describe('error handling - graceful degradation', () => {
    beforeEach(() => {
      mockIsInitialized.value = true
      mockCurrentUser.value = { uid: 'user-123' }
    })

    it('renders slot when remote config refresh fails', async () => {
      mockRefreshRemoteConfig.mockRejectedValue(new Error('Network error'))

      const wrapper = mountComponent()
      await flushPromises()

      expect(wrapper.find('[data-testid="slot-content"]').exists()).toBe(true)
    })

    it('renders slot when remote config refresh times out', async () => {
      vi.useFakeTimers()
      // Never-resolving promise simulates a hanging request
      mockRefreshRemoteConfig.mockReturnValue(new Promise(() => {}))

      const wrapper = mountComponent()
      await flushPromises()

      // Still showing spinner before timeout
      expect(wrapper.find('.progress-spinner').exists()).toBe(true)

      // Advance past the 10 second timeout
      await vi.advanceTimersByTimeAsync(10_001)
      await flushPromises()

      // Should render slot after timeout (graceful degradation)
      expect(wrapper.find('[data-testid="slot-content"]').exists()).toBe(true)
      vi.useRealTimers()
    })

    it('renders slot when workspace store initialization fails', async () => {
      mockTeamWorkspacesEnabled.value = true
      mockWorkspaceStoreInitialize.mockRejectedValue(
        new Error('Workspace init failed')
      )

      const wrapper = mountComponent()
      await flushPromises()

      expect(wrapper.find('[data-testid="slot-content"]').exists()).toBe(true)
    })
  })
})
