import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useCachedRequest } from '@/composables/useCachedRequest'

describe('useCachedRequest', () => {
  let mockRequestFn: (
    params: unknown,
    signal?: AbortSignal
  ) => Promise<unknown | null>
  let abortSpy: () => void

  beforeEach(() => {
    vi.clearAllMocks()

    // Create a spy for the AbortController.abort method
    abortSpy = vi.fn()

    // Mock AbortController
    vi.stubGlobal(
      'AbortController',
      class MockAbortController {
        signal = { aborted: false }
        abort = abortSpy
      }
    )

    // Create a mock request function that returns different results based on params
    mockRequestFn = vi.fn(async (params: unknown) => {
      // Simulate a request that takes some time
      await new Promise((resolve) => setTimeout(resolve, 8))

      if (params === null) return null

      // Return a result based on the params
      return { data: `Result for ${JSON.stringify(params)}` }
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('should cache results and not repeat calls with the same params', async () => {
    const cachedRequest = useCachedRequest(mockRequestFn)

    // First call should make the request
    const result1 = await cachedRequest.call({ id: 1 })
    expect(result1).toEqual({ data: 'Result for {"id":1}' })
    expect(mockRequestFn).toHaveBeenCalledTimes(1)

    // Second call with the same params should use the cache
    const result2 = await cachedRequest.call({ id: 1 })
    expect(result2).toEqual({ data: 'Result for {"id":1}' })
    expect(mockRequestFn).toHaveBeenCalledTimes(1) // Still only called once

    // Call with different params should make a new request
    const result3 = await cachedRequest.call({ id: 2 })
    expect(result3).toEqual({ data: 'Result for {"id":2}' })
    expect(mockRequestFn).toHaveBeenCalledTimes(2)
  })

  it('should deduplicate in-flight requests with the same params', async () => {
    const cachedRequest = useCachedRequest(mockRequestFn)

    // Start two requests with the same params simultaneously
    const promise1 = cachedRequest.call({ id: 1 })
    const promise2 = cachedRequest.call({ id: 1 })

    // Wait for both to complete
    const [result1, result2] = await Promise.all([promise1, promise2])

    // Both should have the same result
    expect(result1).toEqual({ data: 'Result for {"id":1}' })
    expect(result2).toEqual({ data: 'Result for {"id":1}' })

    // But the request function should only be called once
    expect(mockRequestFn).toHaveBeenCalledTimes(1)
  })

  it('should not repeat requests that throw errors', async () => {
    // Create a mock function that throws an error
    const errorMockFn = vi.fn(async () => {
      throw new Error('Test error')
    })

    const cachedRequest = useCachedRequest(errorMockFn)

    // Make a request that will throw
    const result = await cachedRequest.call({ id: 1 })

    // The result should be null
    expect(result).toBeNull()
    expect(errorMockFn).toHaveBeenCalledTimes(1)

    // Make the same request again
    const result2 = await cachedRequest.call({ id: 1 })
    expect(result2).toBeNull()

    // Verify error result is cached and not called again
    expect(errorMockFn).toHaveBeenCalledTimes(1)
  })

  it('should evict least recently used entries when cache exceeds maxSize', async () => {
    // Create a cached request with a small max size
    const cachedRequest = useCachedRequest(mockRequestFn, { maxSize: 2 })

    // Make 3 different requests to exceed the cache size
    await cachedRequest.call({ id: 1 })
    await cachedRequest.call({ id: 2 })
    await cachedRequest.call({ id: 3 })
    await cachedRequest.call({ id: 4 })

    expect(mockRequestFn).toHaveBeenCalledTimes(4)

    // Request id:1 again - it should have been evicted
    await cachedRequest.call({ id: 1 })
    expect(mockRequestFn).toHaveBeenCalledTimes(5)

    // Request least recently used entries
    await cachedRequest.call({ id: 1 })
    await cachedRequest.call({ id: 4 })
    expect(mockRequestFn).toHaveBeenCalledTimes(5) // No new calls
  })

  it('should not repeat calls with same params in different order', async () => {
    const cachedRequest = useCachedRequest(mockRequestFn)

    // First call with params in one order
    await cachedRequest.call({ a: 1, b: 2 })
    expect(mockRequestFn).toHaveBeenCalledTimes(1)

    // Params in different order should still share cache key
    await cachedRequest.call({ b: 2, a: 1 })

    // Verify request function not called again (cache hit)
    expect(mockRequestFn).toHaveBeenCalledTimes(1)
  })

  it('should use custom cache key function if provided', async () => {
    // Create a cache key function that sorts object keys
    const cacheKeyFn = (params: unknown) => {
      if (typeof params !== 'object' || params === null) return String(params)
      return JSON.stringify(
        Object.keys(params as Record<string, unknown>)
          .sort()
          .reduce(
            (acc, key) => ({
              ...acc,
              [key]: (params as Record<string, unknown>)[key]
            }),
            {}
          )
      )
    }

    const cachedRequest = useCachedRequest(mockRequestFn, { cacheKeyFn })

    // First call with params in one order
    const result1 = await cachedRequest.call({ a: 1, b: 2 })
    expect(result1).toEqual({ data: 'Result for {"a":1,"b":2}' })
    expect(mockRequestFn).toHaveBeenCalledTimes(1)

    // Second call with same params in different order should use cache
    const result2 = await cachedRequest.call({ b: 2, a: 1 })
    expect(result2).toEqual({ data: 'Result for {"a":1,"b":2}' })
    expect(mockRequestFn).toHaveBeenCalledTimes(1) // Still only called once
  })

  it('should abort requests when cancel is called', async () => {
    const cachedRequest = useCachedRequest(mockRequestFn)

    // Start a request but don't await it
    const promise = cachedRequest.call({ id: 1 })

    // Cancel all requests
    cachedRequest.cancel()

    // The abort method should have been called
    expect(abortSpy).toHaveBeenCalled()

    // The promise should still resolve (our mock doesn't actually abort)
    const result = await promise
    expect(result).toEqual({ data: 'Result for {"id":1}' })
  })

  it('should clear the cache when clear is called', async () => {
    const cachedRequest = useCachedRequest(mockRequestFn)

    // Make a request to populate the cache
    await cachedRequest.call({ id: 1 })
    expect(mockRequestFn).toHaveBeenCalledTimes(1)

    // Clear the cache
    cachedRequest.clear()

    // Make the same request again
    await cachedRequest.call({ id: 1 })

    // The request function should be called again
    expect(mockRequestFn).toHaveBeenCalledTimes(2)
  })

  it('should handle null results correctly', async () => {
    const cachedRequest = useCachedRequest(mockRequestFn)

    // Make a request that returns null
    const result = await cachedRequest.call(null)
    expect(result).toBeNull()
    expect(mockRequestFn).toHaveBeenCalledTimes(1)

    // Make the same request again
    const result2 = await cachedRequest.call(null)
    expect(result2).toBeNull()

    // Verify null result is treated as any other result (doesn't cause infinite cache miss)
    expect(mockRequestFn).toHaveBeenCalledTimes(1)
  })

  it('should handle string params correctly', async () => {
    const cachedRequest = useCachedRequest(mockRequestFn)

    // Make requests with string params
    await cachedRequest.call('string-param')
    expect(mockRequestFn).toHaveBeenCalledTimes(1)

    // Verify cache hit
    await cachedRequest.call('string-param')
    expect(mockRequestFn).toHaveBeenCalledTimes(1)
  })

  it('should handle number params correctly', async () => {
    const cachedRequest = useCachedRequest(mockRequestFn)

    // Make request with number param
    await cachedRequest.call(123)
    expect(mockRequestFn).toHaveBeenCalledTimes(1)

    // Verify cache hit
    await cachedRequest.call(123)
    expect(mockRequestFn).toHaveBeenCalledTimes(1)
  })
})
