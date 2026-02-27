import type { Mock } from 'vitest'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { computed, nextTick } from 'vue'

import { api } from '@/scripts/api'

interface MockWebSocket {
  readyState: number
  send: Mock
  close: Mock
  addEventListener: Mock
  removeEventListener: Mock
}

describe('API Feature Flags', () => {
  let mockWebSocket: MockWebSocket
  const wsEventHandlers: { [key: string]: (event: unknown) => void } = {}

  beforeEach(() => {
    // Use fake timers
    vi.useFakeTimers()

    // Mock WebSocket
    mockWebSocket = {
      readyState: 1, // WebSocket.OPEN
      send: vi.fn(),
      close: vi.fn(),
      addEventListener: vi.fn(
        (event: string, handler: (event: unknown) => void) => {
          wsEventHandlers[event] = handler
        }
      ),
      removeEventListener: vi.fn()
    }

    // Mock WebSocket constructor
    vi.stubGlobal('WebSocket', function (this: WebSocket) {
      Object.assign(this, mockWebSocket)
    })

    // Reset API state
    api.serverFeatureFlags.value = {}

    // Mock getClientFeatureFlags to return test feature flags
    vi.spyOn(api, 'getClientFeatureFlags').mockReturnValue({
      supports_preview_metadata: true,
      api_version: '1.0.0',
      capabilities: ['bulk_operations', 'async_nodes']
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  describe('Feature flags negotiation', () => {
    it('should send client feature flags as first message on connection', async () => {
      // Initialize API connection
      const initPromise = api.init()

      // Simulate connection open
      wsEventHandlers['open'](new Event('open'))

      // Check that feature flags were sent as first message
      expect(mockWebSocket.send).toHaveBeenCalledTimes(1)
      const sentMessage = JSON.parse(mockWebSocket.send.mock.calls[0][0])
      expect(sentMessage).toEqual({
        type: 'feature_flags',
        data: {
          supports_preview_metadata: true,
          api_version: '1.0.0',
          capabilities: ['bulk_operations', 'async_nodes']
        }
      })

      // Simulate server response with status message
      wsEventHandlers['message']({
        data: JSON.stringify({
          type: 'status',
          data: {
            status: { exec_info: { queue_remaining: 0 } },
            sid: 'test-sid'
          }
        })
      })

      // Simulate server feature flags response
      wsEventHandlers['message']({
        data: JSON.stringify({
          type: 'feature_flags',
          data: {
            supports_preview_metadata: true,
            async_execution: true,
            supported_formats: ['webp', 'jpeg', 'png'],
            api_version: '1.0.0',
            max_upload_size: 104857600,
            capabilities: ['isolated_nodes', 'dynamic_models']
          }
        })
      })

      await initPromise

      // Check that server features were stored
      expect(api.serverFeatureFlags.value).toEqual({
        supports_preview_metadata: true,
        async_execution: true,
        supported_formats: ['webp', 'jpeg', 'png'],
        api_version: '1.0.0',
        max_upload_size: 104857600,
        capabilities: ['isolated_nodes', 'dynamic_models']
      })
    })

    it('should handle server without feature flags support', async () => {
      // Initialize API connection
      const initPromise = api.init()

      // Simulate connection open
      wsEventHandlers['open'](new Event('open'))

      // Clear the send mock to reset
      mockWebSocket.send.mockClear()

      // Simulate server response with status but no feature flags
      wsEventHandlers['message']({
        data: JSON.stringify({
          type: 'status',
          data: {
            status: { exec_info: { queue_remaining: 0 } },
            sid: 'test-sid'
          }
        })
      })

      // Simulate some other message (not feature flags)
      wsEventHandlers['message']({
        data: JSON.stringify({
          type: 'execution_start',
          data: {}
        })
      })

      await initPromise

      // Server features should remain empty
      expect(api.serverFeatureFlags.value).toEqual({})
    })
  })

  describe('Feature checking methods', () => {
    beforeEach(() => {
      // Set up some test features
      api.serverFeatureFlags.value = {
        supports_preview_metadata: true,
        async_execution: false,
        capabilities: ['isolated_nodes', 'dynamic_models']
      }
    })

    it('should check if server supports a boolean feature', () => {
      expect(api.serverSupportsFeature('supports_preview_metadata')).toBe(true)
      expect(api.serverSupportsFeature('async_execution')).toBe(false)
      expect(api.serverSupportsFeature('non_existent_feature')).toBe(false)
    })

    it('should get server feature value', () => {
      expect(api.getServerFeature('supports_preview_metadata')).toBe(true)
      expect(api.getServerFeature('capabilities')).toEqual([
        'isolated_nodes',
        'dynamic_models'
      ])
      expect(api.getServerFeature('non_existent_feature')).toBeUndefined()
    })
  })

  describe('Client feature flags configuration', () => {
    it('should use mocked client feature flags', () => {
      // Verify mocked flags are returned
      const clientFlags = api.getClientFeatureFlags()
      expect(clientFlags).toEqual({
        supports_preview_metadata: true,
        api_version: '1.0.0',
        capabilities: ['bulk_operations', 'async_nodes']
      })
    })

    it('should return a copy of client feature flags', () => {
      // Temporarily restore the real implementation for this test
      vi.mocked(api.getClientFeatureFlags).mockRestore()

      // Verify that modifications to returned object don't affect original
      const clientFlags1 = api.getClientFeatureFlags()
      const clientFlags2 = api.getClientFeatureFlags()

      // Should be different objects
      expect(clientFlags1).not.toBe(clientFlags2)

      // But with same content
      expect(clientFlags1).toEqual(clientFlags2)

      // Modifying one should not affect the other
      clientFlags1.test_flag = true
      expect(api.getClientFeatureFlags()).not.toHaveProperty('test_flag')
    })
  })

  describe('Integration with preview messages', () => {
    it('should affect preview message handling based on feature support', () => {
      // Test with metadata support
      api.serverFeatureFlags.value = { supports_preview_metadata: true }
      expect(api.serverSupportsFeature('supports_preview_metadata')).toBe(true)

      // Test without metadata support
      api.serverFeatureFlags.value = {}
      expect(api.serverSupportsFeature('supports_preview_metadata')).toBe(false)
    })
  })

  describe('Reactivity', () => {
    it('should trigger computed updates when serverFeatureFlags changes', async () => {
      api.serverFeatureFlags.value = {}

      const flag = computed(() =>
        api.getServerFeature('supports_preview_metadata', false)
      )
      expect(flag.value).toBe(false)

      api.serverFeatureFlags.value = { supports_preview_metadata: true }
      await nextTick()

      expect(flag.value).toBe(true)
    })
  })

  describe('Dev override via localStorage', () => {
    afterEach(() => {
      localStorage.clear()
    })

    it('getServerFeature returns localStorage override over server value', () => {
      api.serverFeatureFlags.value = { some_flag: false }
      localStorage.setItem('ff:some_flag', 'true')

      expect(api.getServerFeature('some_flag')).toBe(true)
    })

    it('serverSupportsFeature returns localStorage override over server value', () => {
      api.serverFeatureFlags.value = { some_flag: false }
      localStorage.setItem('ff:some_flag', 'true')

      expect(api.serverSupportsFeature('some_flag')).toBe(true)
    })

    it('getServerFeature falls through when no override is set', () => {
      api.serverFeatureFlags.value = { some_flag: 'server_value' }

      expect(api.getServerFeature('some_flag')).toBe('server_value')
    })

    it('getServerFeature override works with numeric values', () => {
      api.serverFeatureFlags.value = { max_upload_size: 100 }
      localStorage.setItem('ff:max_upload_size', '999')

      expect(api.getServerFeature('max_upload_size')).toBe(999)
    })
  })
})
