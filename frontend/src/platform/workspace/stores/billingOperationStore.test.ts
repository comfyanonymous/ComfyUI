import { setActivePinia, createPinia } from 'pinia'
import { beforeEach, describe, expect, it, vi, afterEach } from 'vitest'

import type { BillingOpStatusResponse } from '@/platform/workspace/api/workspaceApi'

const mockFetchStatus = vi.fn()
const mockFetchBalance = vi.fn()

vi.mock('@/composables/billing/useBillingContext', () => ({
  useBillingContext: () => ({
    fetchStatus: mockFetchStatus,
    fetchBalance: mockFetchBalance
  })
}))

const mockToastAdd = vi.fn()
const mockToastRemove = vi.fn()

vi.mock('@/platform/updates/common/toastStore', () => ({
  useToastStore: () => ({
    add: mockToastAdd,
    remove: mockToastRemove
  })
}))

vi.mock('@/platform/workspace/api/workspaceApi', () => ({
  workspaceApi: {
    getBillingOpStatus: vi.fn()
  }
}))

vi.mock('@/i18n', () => ({
  t: (key: string) => key
}))

vi.mock('@/platform/settings/composables/useSettingsDialog', () => ({
  useSettingsDialog: () => ({
    show: vi.fn(),
    hide: vi.fn(),
    showAbout: vi.fn()
  })
}))

vi.mock('@/stores/dialogStore', () => ({
  useDialogStore: () => ({
    closeDialog: vi.fn()
  })
}))

import { workspaceApi } from '@/platform/workspace/api/workspaceApi'

import { useBillingOperationStore } from './billingOperationStore'

describe('billingOperationStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('startOperation', () => {
    it('creates a pending operation', () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      expect(store.operations.size).toBe(1)
      const operation = store.getOperation('op-1')
      expect(operation).toBeDefined()
      expect(operation?.status).toBe('pending')
      expect(operation?.type).toBe('subscription')
      expect(store.hasPendingOperations).toBe(true)
    })

    it('does not create duplicate operations', () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')
      store.startOperation('op-1', 'topup')

      expect(store.operations.size).toBe(1)
      expect(store.getOperation('op-1')?.type).toBe('subscription')
    })

    it('shows immediate processing toast for subscription operations', () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'info',
        summary: 'billingOperation.subscriptionProcessing',
        group: 'billing-operation'
      })
    })

    it('shows immediate processing toast for topup operations', () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'topup')

      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'info',
        summary: 'billingOperation.topupProcessing',
        group: 'billing-operation'
      })
    })
  })

  describe('polling success', () => {
    it('updates status and shows toast on success', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'succeeded',
        started_at: new Date().toISOString(),
        completed_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      await vi.advanceTimersByTimeAsync(0)

      const operation = store.getOperation('op-1')
      expect(operation?.status).toBe('succeeded')
      expect(store.hasPendingOperations).toBe(false)

      expect(mockFetchStatus).toHaveBeenCalled()
      expect(mockFetchBalance).toHaveBeenCalled()

      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'success',
        summary: 'billingOperation.subscriptionSuccess',
        life: 5000
      })
    })

    it('shows topup success message for topup operations', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'succeeded',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'topup')

      await vi.advanceTimersByTimeAsync(0)

      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'success',
        summary: 'billingOperation.topupSuccess',
        life: 5000
      })
    })

    it('removes the received toast when operation succeeds', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'succeeded',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      const receivedToast = mockToastAdd.mock.calls[0][0]

      await vi.advanceTimersByTimeAsync(0)

      expect(mockToastRemove).toHaveBeenCalledWith(receivedToast)
    })
  })

  describe('polling failure', () => {
    it('updates status and shows error toast on failure', async () => {
      const errorMessage = 'Payment declined'
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'failed',
        error_message: errorMessage,
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      await vi.advanceTimersByTimeAsync(0)

      const operation = store.getOperation('op-1')
      expect(operation?.status).toBe('failed')
      expect(operation?.errorMessage).toBe(errorMessage)
      expect(store.hasPendingOperations).toBe(false)

      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'error',
        summary: 'billingOperation.subscriptionFailed',
        detail: errorMessage,
        life: 5000
      })
    })

    it('uses default message when no error_message in response', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'failed',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'topup')

      await vi.advanceTimersByTimeAsync(0)

      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'error',
        summary: 'billingOperation.topupFailed',
        detail: undefined,
        life: 5000
      })
    })
  })

  describe('polling timeout', () => {
    it('times out after 2 minutes and shows error toast', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      await vi.advanceTimersByTimeAsync(0)

      await vi.advanceTimersByTimeAsync(121_000)
      await vi.runAllTimersAsync()

      const operation = store.getOperation('op-1')
      expect(operation?.status).toBe('timeout')
      expect(store.hasPendingOperations).toBe(false)

      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'error',
        summary: 'billingOperation.subscriptionTimeout',
        life: 5000
      })
    })

    it('shows topup timeout message for topup operations', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'topup')

      await vi.advanceTimersByTimeAsync(121_000)
      await vi.runAllTimersAsync()

      expect(mockToastAdd).toHaveBeenCalledWith({
        severity: 'error',
        summary: 'billingOperation.topupTimeout',
        life: 5000
      })
    })
  })

  describe('exponential backoff', () => {
    it('uses exponential backoff for polling intervals', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      await vi.advanceTimersByTimeAsync(0)
      expect(workspaceApi.getBillingOpStatus).toHaveBeenCalledTimes(1)

      await vi.advanceTimersByTimeAsync(1500)
      expect(workspaceApi.getBillingOpStatus).toHaveBeenCalledTimes(2)

      await vi.advanceTimersByTimeAsync(2250)
      expect(workspaceApi.getBillingOpStatus).toHaveBeenCalledTimes(3)
    })

    it('caps polling interval at 8 seconds', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      await vi.advanceTimersByTimeAsync(60_000)

      const callCountBefore = vi.mocked(workspaceApi.getBillingOpStatus).mock
        .calls.length

      await vi.advanceTimersByTimeAsync(8000)

      expect(
        vi.mocked(workspaceApi.getBillingOpStatus).mock.calls.length
      ).toBeGreaterThan(callCountBefore)
    })
  })

  describe('network errors', () => {
    it('continues polling on network errors', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus)
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({
          id: 'op-1',
          status: 'succeeded',
          started_at: new Date().toISOString()
        } satisfies BillingOpStatusResponse)

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      await vi.advanceTimersByTimeAsync(0)
      expect(store.getOperation('op-1')?.status).toBe('pending')

      await vi.advanceTimersByTimeAsync(1500)
      expect(store.getOperation('op-1')?.status).toBe('pending')

      await vi.advanceTimersByTimeAsync(2250)
      expect(store.getOperation('op-1')?.status).toBe('succeeded')
    })
  })

  describe('clearOperation', () => {
    it('removes operation from the store', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'succeeded',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      await vi.advanceTimersByTimeAsync(0)

      expect(store.operations.size).toBe(1)

      store.clearOperation('op-1')

      expect(store.operations.size).toBe(0)
      expect(store.getOperation('op-1')).toBeUndefined()
    })
  })

  describe('multiple operations', () => {
    it('can track multiple operations concurrently', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockImplementation(
        async (opId: string) => ({
          id: opId,
          status: 'pending' as const,
          started_at: new Date().toISOString()
        })
      )

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')
      store.startOperation('op-2', 'topup')

      expect(store.operations.size).toBe(2)
      expect(store.hasPendingOperations).toBe(true)

      vi.mocked(workspaceApi.getBillingOpStatus).mockImplementation(
        async (opId: string) => ({
          id: opId,
          status:
            opId === 'op-1' ? ('succeeded' as const) : ('pending' as const),
          started_at: new Date().toISOString()
        })
      )

      await vi.advanceTimersByTimeAsync(1500)

      expect(store.getOperation('op-1')?.status).toBe('succeeded')
      expect(store.getOperation('op-2')?.status).toBe('pending')
      expect(store.hasPendingOperations).toBe(true)
    })
  })

  describe('isSettingUp', () => {
    it('returns true when there is a pending subscription operation', () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      expect(store.isSettingUp).toBe(true)
    })

    it('returns false when there is no pending subscription operation', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'succeeded',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      await vi.advanceTimersByTimeAsync(0)

      expect(store.isSettingUp).toBe(false)
    })

    it('returns false when only topup operations are pending', () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'topup')

      expect(store.isSettingUp).toBe(false)
    })
  })

  describe('isAddingCredits', () => {
    it('returns true when there is a pending topup operation', () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'topup')

      expect(store.isAddingCredits).toBe(true)
    })

    it('returns false when there is no pending topup operation', async () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'succeeded',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'topup')

      await vi.advanceTimersByTimeAsync(0)

      expect(store.isAddingCredits).toBe(false)
    })

    it('returns false when only subscription operations are pending', () => {
      vi.mocked(workspaceApi.getBillingOpStatus).mockResolvedValue({
        id: 'op-1',
        status: 'pending',
        started_at: new Date().toISOString()
      })

      const store = useBillingOperationStore()
      store.startOperation('op-1', 'subscription')

      expect(store.isAddingCredits).toBe(false)
    })
  })
})
