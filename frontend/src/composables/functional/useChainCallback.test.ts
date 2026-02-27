import { describe, expect, it } from 'vitest'

import { useChainCallback } from '@/composables/functional/useChainCallback'

describe('useChainCallback', () => {
  it('preserves "this" context in original callback and chained callbacks', () => {
    class TestClass {
      value = 'test'

      constructor() {
        this.method = useChainCallback(this.method, function (this: TestClass) {
          expect(this.value).toBe('test')
        })
      }

      method() {
        expect(this.value).toBe('test')
      }
    }

    const instance = new TestClass()
    instance.method()
  })

  it('handles undefined original callback', () => {
    const context = { value: 'test' }
    const chainedFn = useChainCallback(
      undefined,
      function (this: typeof context) {
        expect(this.value).toBe('test')
      }
    )

    chainedFn.call(context)
  })

  it('passes arguments to all callbacks', () => {
    const originalCalls: number[] = []
    const chainedCalls: number[] = []

    const original = function (this: unknown, num: number) {
      originalCalls.push(num)
    }

    const chained = function (this: unknown, num: number) {
      chainedCalls.push(num)
    }

    const chainedFn = useChainCallback(original, chained)
    chainedFn(42)

    expect(originalCalls).toEqual([42])
    expect(chainedCalls).toEqual([42])
  })
})
