import axios from 'axios'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  EventType,
  useCustomerEventsService
} from '@/services/customerEventsService'

// Hoist the mocks to avoid hoisting issues
const mockAxiosInstance = vi.hoisted(() => ({
  get: vi.fn()
}))

const mockFirebaseAuthStore = vi.hoisted(() => ({
  getAuthHeader: vi.fn()
}))

const mockI18n = vi.hoisted(() => ({
  d: vi.fn()
}))

// Mock dependencies
vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => mockAxiosInstance),
    isAxiosError: vi.fn()
  }
}))

vi.mock('@/stores/firebaseAuthStore', () => ({
  useFirebaseAuthStore: vi.fn(() => mockFirebaseAuthStore)
}))

vi.mock('@/i18n', () => ({
  d: mockI18n.d
}))

vi.mock('@/utils/typeGuardUtil', () => ({
  isAbortError: vi.fn()
}))

describe('useCustomerEventsService', () => {
  let service: ReturnType<typeof useCustomerEventsService>

  const mockAuthHeaders = {
    Authorization: 'Bearer mock-token',
    'Content-Type': 'application/json'
  }

  const mockEventsResponse = {
    events: [
      {
        event_id: 'event-1',
        event_type: 'credit_added',
        params: {
          amount: 1000,
          transaction_id: 'txn-123',
          payment_method: 'stripe'
        },
        createdAt: '2024-01-01T10:00:00Z'
      },
      {
        event_id: 'event-2',
        event_type: 'api_usage_completed',
        params: {
          api_name: 'Image Generation',
          model: 'sdxl-base',
          duration: 5000,
          cost: 50
        },
        createdAt: '2024-01-02T10:00:00Z'
      }
    ],
    total: 2,
    page: 1,
    limit: 10,
    totalPages: 1
  }

  beforeEach(() => {
    vi.clearAllMocks()

    // Setup default mocks
    mockFirebaseAuthStore.getAuthHeader.mockResolvedValue(mockAuthHeaders)
    mockI18n.d.mockImplementation((date, options) => {
      // Mock i18n date formatting
      if (options?.month === 'short') {
        return date.toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        })
      }
      return date.toLocaleString()
    })

    service = useCustomerEventsService()
  })

  describe('initialization', () => {
    it('should initialize with default state', () => {
      expect(service.isLoading.value).toBe(false)
      expect(service.error.value).toBeNull()
    })

    it('should initialize i18n date formatter', () => {
      expect(mockI18n.d).toBeDefined()
    })
  })

  describe('getMyEvents', () => {
    it('should fetch events successfully', async () => {
      mockAxiosInstance.get.mockResolvedValue({ data: mockEventsResponse })

      const result = await service.getMyEvents({
        page: 1,
        limit: 10
      })

      expect(mockFirebaseAuthStore.getAuthHeader).toHaveBeenCalled()
      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/customers/events', {
        params: { page: 1, limit: 10 },
        headers: mockAuthHeaders
      })

      expect(result).toEqual(mockEventsResponse)
      expect(service.isLoading.value).toBe(false)
      expect(service.error.value).toBeNull()
    })

    it('should use default parameters when none provided', async () => {
      mockAxiosInstance.get.mockResolvedValue({ data: mockEventsResponse })

      await service.getMyEvents()

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/customers/events', {
        params: { page: 1, limit: 10 },
        headers: mockAuthHeaders
      })
    })

    it('should return null when auth headers are missing', async () => {
      mockFirebaseAuthStore.getAuthHeader.mockResolvedValue(null)

      const result = await service.getMyEvents()

      expect(result).toBeNull()
      expect(service.error.value).toBe('Authentication header is missing')
      expect(mockAxiosInstance.get).not.toHaveBeenCalled()
    })

    it('should handle 400 errors', async () => {
      const errorResponse = {
        response: {
          status: 400,
          data: { message: 'Invalid input' }
        }
      }
      mockAxiosInstance.get.mockRejectedValue(errorResponse)
      vi.mocked(axios.isAxiosError).mockReturnValue(true)

      const result = await service.getMyEvents()

      expect(result).toBeNull()
      expect(service.error.value).toBe('Invalid input, object invalid')
    })

    it('should handle 404 errors', async () => {
      const errorResponse = {
        response: {
          status: 404,
          data: { message: 'Not found' }
        }
      }
      mockAxiosInstance.get.mockRejectedValue(errorResponse)
      vi.mocked(axios.isAxiosError).mockReturnValue(true)

      const result = await service.getMyEvents()

      expect(result).toBeNull()
      expect(service.error.value).toBe('Not found')
    })

    it('should handle network errors', async () => {
      const networkError = new Error('Network Error')
      mockAxiosInstance.get.mockRejectedValue(networkError)
      vi.mocked(axios.isAxiosError).mockReturnValue(false)

      const result = await service.getMyEvents()

      expect(result).toBeNull()
      expect(service.error.value).toBe(
        'Fetching customer events failed: Network Error'
      )
    })
  })

  describe('formatEventType', () => {
    it('should format known event types correctly', () => {
      expect(service.formatEventType(EventType.CREDIT_ADDED)).toBe(
        'Credits Added'
      )
      expect(service.formatEventType(EventType.ACCOUNT_CREATED)).toBe(
        'Account Created'
      )
      expect(service.formatEventType(EventType.API_USAGE_COMPLETED)).toBe(
        'API Usage'
      )
    })

    it('should return the original string for unknown event types', () => {
      expect(service.formatEventType('unknown_event')).toBe('unknown_event')
    })
  })

  describe('getEventSeverity', () => {
    it('should return correct severity for known event types', () => {
      expect(service.getEventSeverity(EventType.CREDIT_ADDED)).toBe('success')
      expect(service.getEventSeverity(EventType.ACCOUNT_CREATED)).toBe('info')
      expect(service.getEventSeverity(EventType.API_USAGE_COMPLETED)).toBe(
        'warning'
      )
    })

    it('should return default severity for unknown event types', () => {
      expect(service.getEventSeverity('unknown_event')).toBe('info')
    })
  })

  describe('formatAmount', () => {
    it('should format amounts correctly', () => {
      expect(service.formatAmount(1000)).toBe('10.00')
      expect(service.formatAmount(2550)).toBe('25.50')
      expect(service.formatAmount(100)).toBe('1.00')
    })

    it('should handle undefined amounts', () => {
      expect(service.formatAmount(undefined)).toBe('0.00')
      expect(service.formatAmount(0)).toBe('0.00')
    })
  })

  describe('formatDate', () => {
    it('should use i18n date formatter', () => {
      const dateString = '2024-01-01T10:00:00Z'

      service.formatDate(dateString)

      expect(mockI18n.d).toHaveBeenCalledWith(new Date(dateString), {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    })

    it('should return formatted date string', () => {
      const dateString = '2024-01-01T10:00:00Z'
      const result = service.formatDate(dateString)

      expect(typeof result).toBe('string')
      expect(result.length).toBeGreaterThan(0)
    })
  })

  describe('hasAdditionalInfo', () => {
    it('should return true when event has additional parameters', () => {
      const event = {
        event_id: 'test',
        event_type: 'api_usage_completed',
        params: {
          api_name: 'test-api',
          model: 'test-model',
          duration: 1000,
          extra_param: 'extra_value'
        },
        createdAt: '2024-01-01T10:00:00Z'
      }

      expect(service.hasAdditionalInfo(event)).toBe(true)
    })

    it('should return false when event only has known parameters', () => {
      const event = {
        event_id: 'test',
        event_type: 'api_usage_completed',
        params: {
          amount: 1000,
          api_name: 'test-api',
          model: 'test-model'
        },
        createdAt: '2024-01-01T10:00:00Z'
      }

      expect(service.hasAdditionalInfo(event)).toBe(false)
    })

    it('should return false when params is undefined', () => {
      const event = {
        event_id: 'test',
        event_type: 'account_created',
        params: undefined,
        createdAt: '2024-01-01T10:00:00Z'
      }

      expect(service.hasAdditionalInfo(event)).toBe(false)
    })
  })

  describe('getTooltipContent', () => {
    it('should generate HTML tooltip content for all parameters', () => {
      const event = {
        event_id: 'test',
        event_type: 'api_usage_completed',
        params: {
          transaction_id: 'txn-123',
          duration: 5000,
          status: 'completed'
        },
        createdAt: '2024-01-01T10:00:00Z'
      }

      const result = service.getTooltipContent(event)

      expect(result).toContain('<strong>Transaction Id:</strong> txn-123')
      expect(result).toContain('<strong>Duration:</strong> 5,000')
      expect(result).toContain('<strong>Status:</strong> completed')
      expect(result).toContain('<br>')
    })

    it('should return empty string when no parameters', () => {
      const event = {
        event_id: 'test',
        event_type: 'account_created',
        params: {},
        createdAt: '2024-01-01T10:00:00Z'
      }

      expect(service.getTooltipContent(event)).toBe('')
    })

    it('should handle undefined params', () => {
      const event = {
        event_id: 'test',
        event_type: 'account_created',
        params: undefined,
        createdAt: '2024-01-01T10:00:00Z'
      }

      expect(service.getTooltipContent(event)).toBe('')
    })
  })

  describe('formatJsonKey', () => {
    it('should format keys correctly', () => {
      expect(service.formatJsonKey('transaction_id')).toBe('Transaction Id')
      expect(service.formatJsonKey('api_name')).toBe('Api Name')
      expect(service.formatJsonKey('simple')).toBe('Simple')
    })
  })

  describe('formatJsonValue', () => {
    it('should format numbers with commas', () => {
      expect(service.formatJsonValue(1000)).toBe('1,000')
      expect(service.formatJsonValue(1234567)).toBe('1,234,567')
    })

    it('should format date strings', () => {
      const dateString = '2024-01-01T10:00:00Z'
      const result = service.formatJsonValue(dateString)
      expect(typeof result).toBe('string')
      expect(result).not.toBe(dateString) // Should be formatted
    })
  })

  describe('error handling edge cases', () => {
    it('should handle non-Error objects', async () => {
      const stringError = 'String error'
      mockAxiosInstance.get.mockRejectedValue(stringError)
      vi.mocked(axios.isAxiosError).mockReturnValue(false)

      const result = await service.getMyEvents()

      expect(result).toBeNull()
      expect(service.error.value).toBe(
        'Fetching customer events failed: String error'
      )
    })

    it('should reset error state on new request', async () => {
      // First request fails
      mockAxiosInstance.get.mockRejectedValueOnce(new Error('First error'))
      await service.getMyEvents()
      expect(service.error.value).toBeTruthy()

      // Second request succeeds
      mockAxiosInstance.get.mockResolvedValueOnce({ data: mockEventsResponse })
      await service.getMyEvents()
      expect(service.error.value).toBeNull()
    })
  })

  describe('EventType enum', () => {
    it('should have correct enum values', () => {
      expect(EventType.CREDIT_ADDED).toBe('credit_added')
      expect(EventType.ACCOUNT_CREATED).toBe('account_created')
      expect(EventType.API_USAGE_STARTED).toBe('api_usage_started')
      expect(EventType.API_USAGE_COMPLETED).toBe('api_usage_completed')
    })
  })

  describe('edge cases for formatting functions', () => {
    it('formatJsonKey should handle empty strings', () => {
      expect(service.formatJsonKey('')).toBe('')
    })

    it('formatJsonKey should handle single words', () => {
      expect(service.formatJsonKey('test')).toBe('Test')
    })

    it('formatAmount should handle very large numbers', () => {
      expect(service.formatAmount(999999999)).toBe('9999999.99')
    })
  })
})
