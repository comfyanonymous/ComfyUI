import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ErrorRecoveryStrategy } from '@/composables/useErrorHandling'
import { useErrorHandling } from '@/composables/useErrorHandling'
import { t } from '@/i18n'
import { useToastStore } from '@/platform/updates/common/toastStore'

describe('useErrorHandling', () => {
  let errorHandler: ReturnType<typeof useErrorHandling>

  beforeEach(() => {
    vi.clearAllMocks()
    setActivePinia(createTestingPinia())
    errorHandler = useErrorHandling()
  })

  describe('wrapWithErrorHandlingAsync', () => {
    it('should execute action successfully', async () => {
      const action = vi.fn(async () => 'success')
      const wrapped = errorHandler.wrapWithErrorHandlingAsync(action)

      const result = await wrapped()

      expect(result).toBe('success')
      expect(action).toHaveBeenCalledOnce()
    })

    it('should call error handler when action throws', async () => {
      const testError = new Error('test error')
      const action = vi.fn(async () => {
        throw testError
      })
      const customErrorHandler = vi.fn()

      const wrapped = errorHandler.wrapWithErrorHandlingAsync(
        action,
        customErrorHandler
      )

      await wrapped()

      expect(customErrorHandler).toHaveBeenCalledWith(testError)
    })

    it('should call finally handler after success', async () => {
      const action = vi.fn(async () => 'success')
      const finallyHandler = vi.fn()

      const wrapped = errorHandler.wrapWithErrorHandlingAsync(
        action,
        undefined,
        finallyHandler
      )

      await wrapped()

      expect(finallyHandler).toHaveBeenCalledOnce()
    })

    it('should call finally handler after error', async () => {
      const action = vi.fn(async () => {
        throw new Error('test error')
      })
      const finallyHandler = vi.fn()

      const wrapped = errorHandler.wrapWithErrorHandlingAsync(
        action,
        vi.fn(),
        finallyHandler
      )

      await wrapped()

      expect(finallyHandler).toHaveBeenCalledOnce()
    })

    describe('error recovery', () => {
      it('should not use recovery strategy when no error occurs', async () => {
        const action = vi.fn(async () => 'success')
        const recoveryStrategy: ErrorRecoveryStrategy = {
          shouldHandle: vi.fn(() => true),
          recover: vi.fn()
        }

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(
          action,
          undefined,
          undefined,
          [recoveryStrategy]
        )

        await wrapped()

        expect(recoveryStrategy.shouldHandle).not.toHaveBeenCalled()
        expect(recoveryStrategy.recover).not.toHaveBeenCalled()
      })

      it('should use recovery strategy when it matches error', async () => {
        const testError = new Error('test error')
        const action = vi.fn(async () => {
          throw testError
        })
        const recoveryStrategy: ErrorRecoveryStrategy = {
          shouldHandle: vi.fn((error) => error === testError),
          recover: vi.fn(async () => {
            // Recovery succeeds, does nothing
          })
        }

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(
          action,
          vi.fn(),
          undefined,
          [recoveryStrategy]
        )

        await wrapped()

        expect(recoveryStrategy.shouldHandle).toHaveBeenCalledWith(testError)
        expect(recoveryStrategy.recover).toHaveBeenCalled()
      })

      it('should pass action and args to recovery strategy', async () => {
        const testError = new Error('test error')
        const action = vi.fn(async (_arg1: string, _arg2: number) => {
          throw testError
        })
        const recoveryStrategy: ErrorRecoveryStrategy<[string, number], void> =
          {
            shouldHandle: vi.fn(() => true),
            recover: vi.fn()
          }

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(
          action,
          vi.fn(),
          undefined,
          [recoveryStrategy]
        )

        await wrapped('test', 123)

        expect(recoveryStrategy.recover).toHaveBeenCalledWith(
          testError,
          action,
          ['test', 123]
        )
      })

      it('should retry operation when recovery succeeds', async () => {
        let attemptCount = 0
        const action = vi.fn(async (value: string) => {
          attemptCount++
          if (attemptCount === 1) {
            throw new Error('first attempt failed')
          }
          return `success: ${value}`
        })

        const recoveryStrategy: ErrorRecoveryStrategy<[string], string> = {
          shouldHandle: vi.fn(() => true),
          recover: vi.fn(async (_error, retry, args) => {
            await retry(...args)
          })
        }

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(
          action,
          vi.fn(),
          undefined,
          [recoveryStrategy]
        )

        await wrapped('test-value')

        expect(action).toHaveBeenCalledTimes(2)
        expect(recoveryStrategy.recover).toHaveBeenCalledOnce()
      })

      it('should not call error handler when recovery succeeds', async () => {
        const action = vi.fn(async () => {
          throw new Error('test error')
        })
        const customErrorHandler = vi.fn()
        const recoveryStrategy: ErrorRecoveryStrategy = {
          shouldHandle: vi.fn(() => true),
          recover: vi.fn(async () => {
            // Recovery succeeds
          })
        }

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(
          action,
          customErrorHandler,
          undefined,
          [recoveryStrategy]
        )

        await wrapped()

        expect(customErrorHandler).not.toHaveBeenCalled()
      })

      it('should call error handler when recovery fails', async () => {
        const originalError = new Error('original error')
        const recoveryError = new Error('recovery error')
        const action = vi.fn(async () => {
          throw originalError
        })
        const customErrorHandler = vi.fn()
        const recoveryStrategy: ErrorRecoveryStrategy = {
          shouldHandle: vi.fn(() => true),
          recover: vi.fn(async () => {
            throw recoveryError
          })
        }

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(
          action,
          customErrorHandler,
          undefined,
          [recoveryStrategy]
        )

        await wrapped()

        expect(customErrorHandler).toHaveBeenCalledWith(originalError)
      })

      it('should try multiple recovery strategies in order', async () => {
        const testError = new Error('test error')
        const action = vi.fn(async () => {
          throw testError
        })

        const strategy1: ErrorRecoveryStrategy = {
          shouldHandle: vi.fn(() => false),
          recover: vi.fn()
        }

        const strategy2: ErrorRecoveryStrategy = {
          shouldHandle: vi.fn(() => true),
          recover: vi.fn(async () => {
            // Recovery succeeds
          })
        }

        const strategy3: ErrorRecoveryStrategy = {
          shouldHandle: vi.fn(() => true),
          recover: vi.fn()
        }

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(
          action,
          vi.fn(),
          undefined,
          [strategy1, strategy2, strategy3]
        )

        await wrapped()

        expect(strategy1.shouldHandle).toHaveBeenCalledWith(testError)
        expect(strategy1.recover).not.toHaveBeenCalled()

        expect(strategy2.shouldHandle).toHaveBeenCalledWith(testError)
        expect(strategy2.recover).toHaveBeenCalled()

        // Strategy 3 should not be checked because strategy 2 handled it
        expect(strategy3.shouldHandle).not.toHaveBeenCalled()
        expect(strategy3.recover).not.toHaveBeenCalled()
      })

      it('should fall back to error handler when no strategy matches', async () => {
        const testError = new Error('test error')
        const action = vi.fn(async () => {
          throw testError
        })
        const customErrorHandler = vi.fn()

        const strategy: ErrorRecoveryStrategy = {
          shouldHandle: vi.fn(() => false),
          recover: vi.fn()
        }

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(
          action,
          customErrorHandler,
          undefined,
          [strategy]
        )

        await wrapped()

        expect(strategy.shouldHandle).toHaveBeenCalledWith(testError)
        expect(strategy.recover).not.toHaveBeenCalled()
        expect(customErrorHandler).toHaveBeenCalledWith(testError)
      })

      it('should work with synchronous actions', async () => {
        const testError = new Error('test error')
        const action = vi.fn(() => {
          throw testError
        })
        const recoveryStrategy: ErrorRecoveryStrategy = {
          shouldHandle: vi.fn(() => true),
          recover: vi.fn(async () => {
            // Recovery succeeds
          })
        }

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(
          action,
          vi.fn(),
          undefined,
          [recoveryStrategy]
        )

        await wrapped()

        expect(recoveryStrategy.recover).toHaveBeenCalled()
      })
    })

    describe('network error detection', () => {
      it.each([
        ['Failed to fetch', 'Chrome/Edge'],
        ['NetworkError when attempting to fetch resource.', 'Firefox'],
        ['Load failed', 'Safari']
      ])('should show disconnected toast for "%s" (%s)', async (message) => {
        const action = vi.fn(async () => {
          throw new TypeError(message)
        })

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(action)
        await wrapped()

        const toastStore = useToastStore()
        expect(toastStore.add).toHaveBeenCalledWith(
          expect.objectContaining({
            severity: 'error',
            detail: t('g.disconnectedFromBackend')
          })
        )
      })

      it('should not treat non-TypeError as network error', async () => {
        const action = vi.fn(async () => {
          throw new Error('Failed to fetch')
        })

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(action)
        await wrapped()

        const toastStore = useToastStore()
        expect(toastStore.add).toHaveBeenCalledWith(
          expect.objectContaining({
            detail: 'Failed to fetch'
          })
        )
      })
    })

    describe('backward compatibility', () => {
      it('should work without recovery strategies parameter', async () => {
        const action = vi.fn(async () => 'success')
        const wrapped = errorHandler.wrapWithErrorHandlingAsync(action)

        const result = await wrapped()

        expect(result).toBe('success')
      })

      it('should work with empty recovery strategies array', async () => {
        const testError = new Error('test error')
        const action = vi.fn(async () => {
          throw testError
        })
        const customErrorHandler = vi.fn()

        const wrapped = errorHandler.wrapWithErrorHandlingAsync(
          action,
          customErrorHandler,
          undefined,
          []
        )

        await wrapped()

        expect(customErrorHandler).toHaveBeenCalledWith(testError)
      })
    })
  })
})
