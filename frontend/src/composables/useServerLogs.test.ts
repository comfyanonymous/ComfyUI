import { useEventListener } from '@vueuse/core'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import { useServerLogs } from '@/composables/useServerLogs'
import type { LogsWsMessage } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'

vi.mock('@/scripts/api', () => ({
  api: {
    subscribeLogs: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn()
  }
}))

vi.mock('@vueuse/core', () => ({
  useEventListener: vi.fn().mockReturnValue(vi.fn())
}))

describe('useServerLogs', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should initialize with empty logs array', () => {
    const { logs } = useServerLogs()
    expect(logs.value).toEqual([])
  })

  it('should not subscribe to logs by default', () => {
    useServerLogs()
    expect(api.subscribeLogs).not.toHaveBeenCalled()
  })

  it('should subscribe to logs when immediate is true', () => {
    useServerLogs({ immediate: true })
    expect(api.subscribeLogs).toHaveBeenCalledWith(true)
  })

  it('should start listening when startListening is called', async () => {
    const { startListening } = useServerLogs()

    await startListening()

    expect(api.subscribeLogs).toHaveBeenCalledWith(true)
  })

  it('should stop listening when stopListening is called', async () => {
    const { startListening, stopListening } = useServerLogs()

    await startListening()
    await stopListening()

    expect(api.subscribeLogs).toHaveBeenCalledWith(false)
  })

  it('should register event listener when starting', async () => {
    const { startListening } = useServerLogs()

    await startListening()

    expect(vi.mocked(useEventListener)).toHaveBeenCalledWith(
      api,
      'logs',
      expect.any(Function)
    )
  })

  it('should handle log messages correctly', async () => {
    const { logs, startListening } = useServerLogs()

    await startListening()

    // Get the callback that was registered with useEventListener
    const eventCallback = vi.mocked(useEventListener).mock.calls[0][2] as (
      event: CustomEvent<LogsWsMessage>
    ) => void

    // Simulate receiving a log event
    const mockEvent = new CustomEvent('logs', {
      detail: {
        type: 'logs',
        entries: [{ m: 'Log message 1' }, { m: 'Log message 2' }]
      } as unknown as LogsWsMessage
    }) as CustomEvent<LogsWsMessage>

    eventCallback(mockEvent)
    await nextTick()

    expect(logs.value).toEqual(['Log message 1', 'Log message 2'])
  })

  it('should use the message filter if provided', async () => {
    const { logs, startListening } = useServerLogs({
      messageFilter: (msg) => msg !== 'remove me'
    })

    await startListening()

    const eventCallback = vi.mocked(useEventListener).mock.calls[0][2] as (
      event: CustomEvent<LogsWsMessage>
    ) => void

    const mockEvent = new CustomEvent('logs', {
      detail: {
        type: 'logs',
        entries: [
          { m: 'Log message 1 dont remove me' },
          { m: 'remove me' },
          { m: '' }
        ]
      } as unknown as LogsWsMessage
    }) as CustomEvent<LogsWsMessage>

    eventCallback(mockEvent)
    await nextTick()

    expect(logs.value).toEqual(['Log message 1 dont remove me', ''])
  })
})
