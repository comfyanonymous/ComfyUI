import { t } from '@/i18n'
import { useToastStore } from '@/platform/updates/common/toastStore'

/**
 * Strategy for recovering from specific error conditions.
 * Allows operations to be retried after resolving the error condition.
 *
 * @template TArgs - The argument types of the operation to be retried
 * @template TReturn - The return type of the operation
 *
 * @example
 * ```typescript
 * const networkRecovery: ErrorRecoveryStrategy = {
 *   shouldHandle: (error) => error instanceof NetworkError,
 *   recover: async (error, retry) => {
 *     await waitForNetwork()
 *     await retry()
 *   }
 * }
 * ```
 */
export interface ErrorRecoveryStrategy<
  TArgs extends unknown[] = unknown[],
  TReturn = unknown
> {
  /**
   * Determines if this strategy should handle the given error.
   * @param error - The error to check
   * @returns true if this strategy can handle the error
   */
  shouldHandle: (error: unknown) => boolean

  /**
   * Attempts to recover from the error and retry the operation.
   * This function is responsible for:
   * 1. Resolving the error condition (e.g., reauthentication, network reconnect)
   * 2. Calling retry() to re-execute the original operation
   * 3. Handling the retry result (success or failure)
   *
   * @param error - The error that occurred
   * @param retry - Function to retry the original operation
   * @param args - Original arguments passed to the operation
   * @returns Promise that resolves when recovery completes (whether successful or not)
   */
  recover: (
    error: unknown,
    retry: (...args: TArgs) => Promise<TReturn> | TReturn,
    args: TArgs
  ) => Promise<void>
}

export function useErrorHandling() {
  const toast = useToastStore()
  const toastErrorHandler = (error: unknown) => {
    const isNetworkError =
      error instanceof TypeError &&
      /failed to fetch|networkerror|load failed/i.test(error.message)
    const message = isNetworkError
      ? t('g.disconnectedFromBackend')
      : error instanceof Error
        ? error.message
        : t('g.unknownError')
    toast.add({
      severity: 'error',
      summary: t('g.error'),
      detail: message
    })
    console.error(error)
  }

  const wrapWithErrorHandling =
    <TArgs extends unknown[], TReturn>(
      action: (...args: TArgs) => TReturn,
      errorHandler?: (error: unknown) => void,
      finallyHandler?: () => void
    ) =>
    (...args: TArgs): TReturn | undefined => {
      try {
        return action(...args)
      } catch (e) {
        ;(errorHandler ?? toastErrorHandler)(e)
      } finally {
        finallyHandler?.()
      }
    }

  const wrapWithErrorHandlingAsync =
    <TArgs extends unknown[], TReturn>(
      action: (...args: TArgs) => Promise<TReturn> | TReturn,
      errorHandler?: (error: unknown) => void,
      finallyHandler?: () => void,
      recoveryStrategies: ErrorRecoveryStrategy<TArgs, TReturn>[] = []
    ) =>
    async (...args: TArgs): Promise<TReturn | undefined> => {
      try {
        return await action(...args)
      } catch (e) {
        for (const strategy of recoveryStrategies) {
          if (strategy.shouldHandle(e)) {
            try {
              await strategy.recover(e, action, args)
              return
            } catch (recoveryError) {
              console.error('Recovery strategy failed:', recoveryError)
            }
          }
        }

        ;(errorHandler ?? toastErrorHandler)(e)
      } finally {
        finallyHandler?.()
      }
    }

  return {
    wrapWithErrorHandling,
    wrapWithErrorHandlingAsync,
    toastErrorHandler
  }
}
