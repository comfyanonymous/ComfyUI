import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  startTopupTracking,
  checkForCompletedTopup,
  clearTopupTracking
} from '@/platform/telemetry/topupTracker'
import type { AuditLog } from '@/services/customerEventsService'

// Mock localStorage
const mockLocalStorage = vi.hoisted(() => ({
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn()
}))

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true
})

// Mock telemetry
const mockTelemetry = vi.hoisted(() => ({
  trackApiCreditTopupSucceeded: vi.fn()
}))

vi.mock('@/platform/telemetry', () => ({
  useTelemetry: vi.fn(() => mockTelemetry)
}))

describe('topupTracker', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('startTopupTracking', () => {
    it('should save current timestamp to localStorage', () => {
      const beforeTimestamp = Date.now()

      startTopupTracking()

      expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
        'pending_topup_timestamp',
        expect.any(String)
      )

      const savedTimestamp = parseInt(
        mockLocalStorage.setItem.mock.calls[0][1],
        10
      )
      expect(savedTimestamp).toBeGreaterThanOrEqual(beforeTimestamp)
      expect(savedTimestamp).toBeLessThanOrEqual(Date.now())
    })
  })

  describe('checkForCompletedTopup', () => {
    it('should return false if no pending topup exists', () => {
      mockLocalStorage.getItem.mockReturnValue(null)

      const result = checkForCompletedTopup([])

      expect(result).toBe(false)
      expect(mockTelemetry.trackApiCreditTopupSucceeded).not.toHaveBeenCalled()
    })

    it('should return false if events array is empty', () => {
      mockLocalStorage.getItem.mockReturnValue(Date.now().toString())

      const result = checkForCompletedTopup([])

      expect(result).toBe(false)
      expect(mockTelemetry.trackApiCreditTopupSucceeded).not.toHaveBeenCalled()
    })

    it('should return false if events array is null', () => {
      mockLocalStorage.getItem.mockReturnValue(Date.now().toString())

      const result = checkForCompletedTopup(null)

      expect(result).toBe(false)
      expect(mockTelemetry.trackApiCreditTopupSucceeded).not.toHaveBeenCalled()
    })

    it('should auto-cleanup if timestamp is older than 24 hours', () => {
      const oldTimestamp = Date.now() - 25 * 60 * 60 * 1000 // 25 hours ago
      mockLocalStorage.getItem.mockReturnValue(oldTimestamp.toString())

      const events: AuditLog[] = [
        {
          event_id: 'test-1',
          event_type: 'credit_added',
          createdAt: new Date().toISOString(),
          params: { amount: 500 }
        }
      ]

      const result = checkForCompletedTopup(events)

      expect(result).toBe(false)
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith(
        'pending_topup_timestamp'
      )
      expect(mockTelemetry.trackApiCreditTopupSucceeded).not.toHaveBeenCalled()
    })

    it('should detect completed topup and fire telemetry', () => {
      const startTimestamp = Date.now() - 5 * 60 * 1000 // 5 minutes ago
      mockLocalStorage.getItem.mockReturnValue(startTimestamp.toString())

      const events: AuditLog[] = [
        {
          event_id: 'test-1',
          event_type: 'api_usage_completed',
          createdAt: new Date(startTimestamp - 1000).toISOString(),
          params: {}
        },
        {
          event_id: 'test-2',
          event_type: 'credit_added',
          createdAt: new Date(startTimestamp + 1000).toISOString(),
          params: { amount: 500 }
        }
      ]

      const result = checkForCompletedTopup(events)

      expect(result).toBe(true)
      expect(mockTelemetry.trackApiCreditTopupSucceeded).toHaveBeenCalledOnce()
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith(
        'pending_topup_timestamp'
      )
    })

    it('should not detect topup if credit_added event is before tracking started', () => {
      const startTimestamp = Date.now()
      mockLocalStorage.getItem.mockReturnValue(startTimestamp.toString())

      const events: AuditLog[] = [
        {
          event_id: 'test-1',
          event_type: 'credit_added',
          createdAt: new Date(startTimestamp - 1000).toISOString(), // Before tracking
          params: { amount: 500 }
        }
      ]

      const result = checkForCompletedTopup(events)

      expect(result).toBe(false)
      expect(mockTelemetry.trackApiCreditTopupSucceeded).not.toHaveBeenCalled()
      expect(mockLocalStorage.removeItem).not.toHaveBeenCalled()
    })

    it('should ignore events without createdAt timestamp', () => {
      const startTimestamp = Date.now()
      mockLocalStorage.getItem.mockReturnValue(startTimestamp.toString())

      const events: AuditLog[] = [
        {
          event_id: 'test-1',
          event_type: 'credit_added',
          createdAt: undefined,
          params: { amount: 500 }
        }
      ]

      const result = checkForCompletedTopup(events)

      expect(result).toBe(false)
      expect(mockTelemetry.trackApiCreditTopupSucceeded).not.toHaveBeenCalled()
    })

    it('should only match credit_added events, not other event types', () => {
      const startTimestamp = Date.now()
      mockLocalStorage.getItem.mockReturnValue(startTimestamp.toString())

      const events: AuditLog[] = [
        {
          event_id: 'test-1',
          event_type: 'api_usage_completed',
          createdAt: new Date(startTimestamp + 1000).toISOString(),
          params: {}
        },
        {
          event_id: 'test-2',
          event_type: 'account_created',
          createdAt: new Date(startTimestamp + 2000).toISOString(),
          params: {}
        }
      ]

      const result = checkForCompletedTopup(events)

      expect(result).toBe(false)
      expect(mockTelemetry.trackApiCreditTopupSucceeded).not.toHaveBeenCalled()
    })
  })

  describe('clearTopupTracking', () => {
    it('should remove pending topup from localStorage', () => {
      clearTopupTracking()

      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith(
        'pending_topup_timestamp'
      )
    })
  })
})
