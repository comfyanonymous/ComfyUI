/**
 * Chain multiple callbacks together.
 *
 * @param originalCallback - The original callback to chain.
 * @param callbacks - The callbacks to chain.
 * @returns A new callback that chains the original callback with the callbacks.
 */
export function useChainCallback<O, T>(
  originalCallback: T | undefined,
  ...callbacks: NonNullable<T> extends (this: O, ...args: infer P) => unknown
    ? ((this: O, ...args: P) => void)[]
    : never
) {
  type Args = NonNullable<T> extends (...args: infer P) => unknown ? P : never
  type Ret = NonNullable<T> extends (...args: unknown[]) => infer R ? R : never

  return function (this: O, ...args: Args) {
    if (typeof originalCallback === 'function') {
      ;(originalCallback as (this: O, ...args: Args) => Ret).call(this, ...args)
    }
    for (const callback of callbacks) {
      callback.call(this, ...args)
    }
  } as (this: O, ...args: Args) => Ret
}
